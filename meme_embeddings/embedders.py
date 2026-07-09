"""
embedders.py
============
Классы генерации эмбеддингов мемов (Этап 2 пайплайна).

Публичный API:
    create_embedder(config_or_preset_name) -> BaseEmbedder
    BaseEmbedder.encode(texts=..., images=..., ocr_texts=...) -> np.ndarray [N, target_dim], L2-normalized

Иерархия:
    BaseEmbedder (ABC)
      ├── ClipTextEmbedder          — Вариант A: text-башня CLIP-подобной модели
      ├── TextEmbedder              — специализированный text-encoder (E5/BGE/...)
      ├── MultimodalDescriptionEmbedder — Вариант B: VLM-описание -> TextEmbedder
      └── CustomEmbedder            — обёртка над произвольной функцией/моделью пользователя

Все тяжёлые зависимости (torch, transformers, sentence-transformers, open_clip)
импортируются лениво внутри _load_model(), чтобы сам модуль можно было
импортировать без установки всего стека сразу.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from . import env  # noqa: F401  (побочный эффект импорта: подгружает .env до загрузки моделей)
from .config import EmbedderConfig, EmbedderType, ProjectionConfig
from .logging_setup import get_logger

logger = get_logger(__name__)

ImageLike = Union[str, "PIL.Image.Image", bytes]  # noqa: F821  (PIL опционален)


# ============================================================================
# Базовый абстрактный класс
# ============================================================================

class BaseEmbedder(ABC):
    """
    Общий контракт для всех энкодеров мемов.

    Наследники реализуют только _load_model() и _encode_raw() — вся логика
    нормализации, приведения размерности и батчинга уже реализована здесь,
    чтобы поведение (и качество) было одинаковым между реализациями
    (п.2.4 репорта: консистентность препроцессинга).
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._model = None
        self._projection_matrix: Optional[np.ndarray] = None
        logger.info(
            "Инициализация %s: model=%s trust_remote_code=%s device=%s",
            self.__class__.__name__,
            config.model.name,
            config.model.trust_remote_code,
            config.model.device,
        )
        self._load_model()
        self._init_projection()
        logger.info("%s готов, output_dim=%d", self.__class__.__name__, self.output_dim)

    # ---- обязательные к реализации методы ---------------------------------

    @abstractmethod
    def _load_model(self) -> None:
        """Загрузить веса/клиент модели в self._model (и вспомогательные объекты)."""
        raise NotImplementedError

    @abstractmethod
    def _encode_raw(
        self,
        texts: Optional[Sequence[str]] = None,
        images: Optional[Sequence[ImageLike]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Вернуть НЕнормализованные эмбеддинги нативной размерности модели, shape [N, native_dim]."""
        raise NotImplementedError

    # ---- публичный API ------------------------------------------------------

    def encode(
        self,
        texts: Optional[Union[str, Sequence[str]]] = None,
        images: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        is_query: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Основной метод. Принимает текст(ы) и/или картинку(и) в зависимости от типа энкодера.

        is_query=True  -> добавляется query_prefix (для запроса пользователя)
        is_query=False -> добавляется passage_prefix (для индексации мемов)
        Это важно для e5-подобных моделей, где запрос и документ кодируются с разными префиксами.
        """
        texts = self._as_list(texts)
        images = self._as_list(images)
        logger.debug(
            "encode(): n_texts=%s n_images=%s is_query=%s",
            len(texts) if texts else 0,
            len(images) if images else 0,
            is_query,
        )

        if texts is not None:
            prefix = self.config.model.query_prefix if is_query else self.config.model.passage_prefix
            if prefix:
                texts = [f"{prefix}{t}" for t in texts]

        raw = self._encode_raw(texts=texts, images=images, **kwargs)
        vec = self._project(raw)

        if self.config.normalize:
            vec = self._l2_normalize(vec)

        return vec.astype(np.float32)

    def encode_query(self, text: Union[str, Sequence[str]]) -> np.ndarray:
        """Удобный шорткат для кодирования пользовательского запроса на этапе поиска."""
        return self.encode(texts=text, is_query=True)

    @property
    def output_dim(self) -> int:
        if self.config.projection.enabled and self.config.projection.method != "none":
            return self.config.projection.target_dim
        return self.config.model.native_dim

    # ---- приведение размерности (256/512) ------------------------------------

    def _init_projection(self) -> None:
        proj: ProjectionConfig = self.config.projection
        if not proj.enabled or proj.method in ("none", "truncate"):
            return  # truncate не требует матрицы, делается срезом в _project()

        if proj.method == "random":
            rng = np.random.default_rng(proj.seed)
            native_dim = self.config.model.native_dim
            # случайная гауссова матрица + QR для ортонормированности (Johnson–Lindenstrauss)
            mat = rng.standard_normal((native_dim, proj.target_dim))
            q, _ = np.linalg.qr(mat)
            self._projection_matrix = q[:, : proj.target_dim].astype(np.float32)

        elif proj.method == "pca":
            if not proj.pca_artifact_path:
                raise ValueError("projection.method='pca' требует projection.pca_artifact_path")
            self._projection_matrix = np.load(proj.pca_artifact_path)

    def _project(self, raw: np.ndarray) -> np.ndarray:
        proj = self.config.projection
        if not proj.enabled or proj.method == "none":
            return raw
        if proj.method == "truncate":
            return raw[:, : proj.target_dim]
        if proj.method in ("random", "pca"):
            return raw @ self._projection_matrix
        raise ValueError(f"Неизвестный метод проекции: {proj.method}")

    # ---- утилиты --------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return vec / norms

    @staticmethod
    def _as_list(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={self.config.model.name}, "
            f"output_dim={self.output_dim}, version={self.config.model_version})"
        )


# ============================================================================
# Вариант A: CLIP text encoder
# ============================================================================

class ClipTextEmbedder(BaseEmbedder):
    """
    Использует только текстовую башню CLIP-подобной мультиязычной модели
    (jina-clip-v2 / OpenCLIP multilingual / AltCLIP). Image-энкодер не используется
    в основном пути поиска (см. п.9 контекста репорта), но доступен через
    encode_image() на будущее (image-to-image).
    """

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        spec = self.config.model
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code, revision=spec.revision
        )
        try:
            self._model = AutoModel.from_pretrained(
                spec.hf_id, trust_remote_code=spec.trust_remote_code, revision=spec.revision
            ).to(spec.device)
        except ImportError as e:
            # Известная проблема: модели с trust_remote_code=True (например jina-clip-v2)
            # тянут с HF Hub кастомный modeling_*.py, который импортирует приватные
            # хелперы transformers (например clip_loss из transformers.models.clip.modeling_clip).
            # В свежих версиях transformers эти хелперы были убраны/переименованы в рамках
            # рефакторинга CLIP-моделей, из-за чего remote-код модели ломается на импорте.
            # Это НЕ баг в самом модуле — несовместимость версии transformers с кодом модели.
            logger.error(
                "Не удалось загрузить '%s' (trust_remote_code=%s): %s. "
                "Похоже на несовместимость версии transformers с кастомным кодом модели "
                "(например removed helper вроде clip_loss в новых transformers).",
                spec.hf_id, spec.trust_remote_code, e,
            )
            raise ImportError(
                f"Не удалось загрузить модель '{spec.hf_id}' из-за несовместимости версии "
                f"transformers с её кастомным remote-кодом (trust_remote_code=True).\n"
                f"Оригинальная ошибка: {e}\n\n"
                "Как починить:\n"
                "  1) Откатите transformers на версию, под которую писался remote-код модели, "
                "например: pip install \"transformers==4.44.2\" (см. requirements.txt модуля), "
                "либо посмотрите Model Card на HF Hub — там обычно указана проверенная версия.\n"
                "  2) Либо зафиксируйте конкретный revision модели в models.yaml (поле model.revision) — "
                "если проблема не в версии transformers, а в том, что автор модели обновил remote-код.\n"
                "  3) Либо используйте вариант без trust_remote_code (например открытую OpenCLIP "
                "мультиязычную модель через open_clip вместо AutoModel)."
            ) from e
        self._model.eval()

    def _encode_raw(self, texts=None, images=None, **kwargs) -> np.ndarray:
        if texts is None:
            raise ValueError("ClipTextEmbedder.encode() требует texts (image-путь через encode_image)")

        spec = self.config.model
        batches = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            with self._torch.no_grad():
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=spec.max_length,
                    return_tensors="pt",
                ).to(spec.device)
                # Большинство CLIP-реализаций в HF предоставляют get_text_features
                feats = self._model.get_text_features(**inputs)
                batches.append(feats.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    def encode_image(self, images: Sequence[ImageLike]) -> np.ndarray:
        """
        Отдельный вектор изображения (задел на будущее image-to-image поиск,
        не участвует в основном text-to-text пути, п.9 репорта).
        """
        spec = self.config.model
        with self._torch.no_grad():
            inputs = self._tokenizer.image_processor(images, return_tensors="pt").to(spec.device)  # type: ignore
            feats = self._model.get_image_features(**inputs)
            vec = feats.float().cpu().numpy()
        return self._l2_normalize(vec).astype(np.float32)


# ============================================================================
# Специализированный text-embedding encoder (E5 / BGE и т.п.)
# ============================================================================

class TextEmbedder(BaseEmbedder):
    """
    Отдельная text-embedding модель, оптимизированная под чистую текстовую семантику
    (а не под alignment с картинкой, как text-башня CLIP). Используется как:
      - самостоятельный вариант (если rich_description уже есть из другого источника),
      - "текстовый хвост" MultimodalDescriptionEmbedder (вариант B).

    Реализовано через sentence-transformers как самый простой универсальный backend
    для E5/BGE/GTE и любых совместимых моделей с HuggingFace.
    """

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        spec = self.config.model
        self._model = SentenceTransformer(
            spec.hf_id,
            device=spec.device,
            trust_remote_code=spec.trust_remote_code,
            revision=spec.revision,
        )
        if spec.max_length:
            self._model.max_seq_length = spec.max_length

    def _encode_raw(self, texts=None, images=None, **kwargs) -> np.ndarray:
        if texts is None:
            raise ValueError("TextEmbedder.encode() требует texts")
        vec = self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # нормализация делается централизованно в BaseEmbedder
            show_progress_bar=False,
        )
        return vec


# ============================================================================
# Вариант B: Multimodal LLM -> rich_description -> TextEmbedder
# ============================================================================

class MultimodalDescriptionEmbedder(BaseEmbedder):
    """
    Рекомендованный в репорте вариант B:
      1) картинка + OCR-текст -> мультимодальная LLM (Gemma-vision / Qwen-VL / LLaVA)
         генерирует rich_description по фиксированному промпту;
      2) rich_description кодируется отдельным специализированным TextEmbedder
         (см. config.caption_model / config.model).

    encode() принимает images + ocr_texts (списки одинаковой длины) и сам прогоняет
    оба шага. Если rich_description уже посчитан заранее (офлайн-кеш) — можно
    передать его напрямую через texts=[...], тогда шаг генерации описания пропускается.
    """

    def _load_model(self) -> None:
        # 1) капшнер (мультимодальная LLM)
        caption_spec = self.config.caption_model
        if caption_spec is None:
            raise ValueError("MultimodalDescriptionEmbedder требует config.caption_model")

        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._torch = torch
        self._caption_processor = AutoProcessor.from_pretrained(
            caption_spec.hf_id,
            trust_remote_code=caption_spec.trust_remote_code,
            revision=caption_spec.revision,
        )
        self._caption_model = AutoModelForImageTextToText.from_pretrained(
            caption_spec.hf_id,
            trust_remote_code=caption_spec.trust_remote_code,
            revision=caption_spec.revision,
            torch_dtype=getattr(torch, caption_spec.dtype, torch.float32),
        ).to(caption_spec.device)
        self._caption_model.eval()

        # 2) текстовый энкодер описания — независимый TextEmbedder на том же config.model
        text_only_config = EmbedderConfig(
            embedder_type=EmbedderType.TEXT,
            model=self.config.model,
            projection=self.config.projection,
            normalize=False,  # нормализация всё равно делается один раз в BaseEmbedder.encode
            batch_size=self.config.batch_size,
            model_version=self.config.model_version,
            preprocessing_version=self.config.preprocessing_version,
        )
        self._text_embedder = TextEmbedder(text_only_config)

    def generate_description(self, image: ImageLike, ocr_text: str) -> str:
        """Один прогон VLM: картинка + OCR -> rich_description (2-4 предложения)."""
        prompt = self.config.caption_prompt_template.format(ocr_text=ocr_text or "")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._caption_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.config.caption_model.device)

        with self._torch.no_grad():
            do_sample = self.config.caption_temperature > 0
            output_ids = self._caption_model.generate(
                **inputs,
                max_new_tokens=self.config.caption_max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.caption_temperature if do_sample else None,
            )
        text = self._caption_processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )[0]
        return text.strip()

    def _encode_raw(self, texts=None, images=None, ocr_texts: Optional[Sequence[str]] = None, **kwargs) -> np.ndarray:
        if texts is None:
            # rich_description ещё не посчитан — генерируем через VLM
            if images is None:
                raise ValueError("Нужно передать либо texts=rich_description, либо images(+ocr_texts)")
            ocr_texts = ocr_texts or [""] * len(images)
            texts = [self.generate_description(img, ocr) for img, ocr in zip(images, ocr_texts)]

        # прогоняем готовый текст через text-энкодер (без повторной нормализации/проекции —
        # это уже сделает внешний encode() у self, см. BaseEmbedder.encode)
        return self._text_embedder._encode_raw(texts=texts)

    def last_descriptions(self, images: Sequence[ImageLike], ocr_texts: Sequence[str]) -> List[str]:
        """Вспомогательный метод — получить сами тексты описаний (для логов/QA, п.3.4 репорта)."""
        return [self.generate_description(img, ocr) for img, ocr in zip(images, ocr_texts)]


# ============================================================================
# Кастомная модель пользователя
# ============================================================================

class CustomEmbedder(BaseEmbedder):
    """
    Обёртка для любой пользовательской модели/API, которая не укладывается
    в готовые реализации. Пользователь передаёт encode_fn: List[str] -> np.ndarray
    (и опционально encode_image_fn), всё остальное (нормализация, приведение
    размерности, батчинг по конфигу) отрабатывает BaseEmbedder как обычно.

    Пример:
        def my_encode(texts: list[str]) -> np.ndarray:
            return my_model.embed(texts)

        embedder = CustomEmbedder(config, encode_fn=my_encode)
    """

    def __init__(
        self,
        config: EmbedderConfig,
        encode_fn: Callable[[Sequence[str]], np.ndarray],
        encode_image_fn: Optional[Callable[[Sequence[ImageLike]], np.ndarray]] = None,
    ):
        self._encode_fn = encode_fn
        self._encode_image_fn = encode_image_fn
        super().__init__(config)

    def _load_model(self) -> None:
        pass  # модель уже загружена и обёрнута в encode_fn пользователем

    def _encode_raw(self, texts=None, images=None, **kwargs) -> np.ndarray:
        if texts is not None:
            return np.asarray(self._encode_fn(texts), dtype=np.float32)
        if images is not None and self._encode_image_fn is not None:
            return np.asarray(self._encode_image_fn(images), dtype=np.float32)
        raise ValueError("CustomEmbedder: нужно передать texts, или images + encode_image_fn")


# ============================================================================
# Фабрика
# ============================================================================

_REGISTRY = {
    EmbedderType.CLIP: ClipTextEmbedder,
    EmbedderType.TEXT: TextEmbedder,
    EmbedderType.MULTIMODAL: MultimodalDescriptionEmbedder,
}


def create_embedder(
    config: Union[EmbedderConfig, str],
    encode_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    encode_image_fn: Optional[Callable[[Sequence[ImageLike]], np.ndarray]] = None,
) -> BaseEmbedder:
    """
    Единая точка входа в модуль.

        embedder = create_embedder("multilingual-e5-large")   # пресет по имени
        embedder = create_embedder(my_config)                 # свой EmbedderConfig
        embedder = create_embedder(custom_config, encode_fn=my_fn)  # своя модель

    Если config передан строкой — ищется в PRESETS (config.py).
    Если config.embedder_type == CUSTOM — обязателен encode_fn.
    """
    if isinstance(config, str):
        from .config import get_preset

        logger.info("create_embedder(): загружаю пресет '%s'", config)
        config = get_preset(config)

    if config.embedder_type == EmbedderType.CUSTOM:
        if encode_fn is None:
            raise ValueError("Для EmbedderType.CUSTOM нужно передать encode_fn")
        return CustomEmbedder(config, encode_fn=encode_fn, encode_image_fn=encode_image_fn)

    cls = _REGISTRY.get(config.embedder_type)
    if cls is None:
        raise ValueError(f"Неизвестный embedder_type: {config.embedder_type}")
    return cls(config)
