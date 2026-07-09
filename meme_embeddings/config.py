"""
config.py
=========
Конфигурация для модуля генерации эмбеддингов мемов (Этап 2).

Здесь описаны:
- ModelSpec       — параметры конкретной модели (HF id, размерность, устройство и т.д.)
- ProjectionConfig — как native_dim модели сжимается до целевой размерности (256/512)
- EmbedderConfig  — полный конфиг одного эмбеддера (тип + модель + препроцессинг)
- PRESETS         — готовые конфиги под модели из репорта (jina-clip-v2, multilingual-e5-large,
                     bge-m3, вариант B с Gemma-vision + e5)

Конфиг сознательно сделан "плоским" и сериализуемым (dataclasses -> dict -> json),
чтобы model_version/preprocessing_version можно было один в один сохранить
в метаданные векторной БД (см. п.2.4 репорта).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from . import env  # noqa: F401  (побочный эффект импорта: подгружает .env)
from .logging_setup import get_logger

logger = get_logger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODELS_YAML = _THIS_DIR / "config" / "models.yaml"


class EmbedderType(str, Enum):
    """Какую архитектуру построения вектора использовать (см. п.1 репорта)."""

    CLIP = "clip"                 # Вариант A: text-башня CLIP-подобной модели
    TEXT = "text"                 # Отдельный text-embedding энкодер (E5/BGE), без описания
    MULTIMODAL = "multimodal"     # Вариант B: VLM-описание -> text-embedding энкодер
    CUSTOM = "custom"             # Произвольная пользовательская модель/функция


@dataclass
class ModelSpec:
    """Параметры конкретной модели (энкодера или капшнера)."""

    name: str                                  # человекочитаемое имя
    hf_id: str                                 # id в HuggingFace / open_clip
    native_dim: int                            # размерность эмбеддинга модели "as is"
    max_length: int = 128                      # макс. длина в токенах
    device: str = "cpu"                        # "cpu" | "cuda" | "cuda:0" | "mps"
    dtype: str = "float32"                     # "float32" | "float16" | "bfloat16"
    trust_remote_code: bool = False             # обязателен для моделей с кастомным кодом на HF Hub
    revision: Optional[str] = None              # пин на commit/tag модели (воспроизводимость, п.2.4)
    query_prefix: str = ""                     # e5-подобные модели требуют "query: "/"passage: "
    passage_prefix: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)  # любые доп. kwargs под конкретный backend

    def __post_init__(self) -> None:
        # MEME_EMB_DEVICE из .env / окружения имеет приоритет над значением из yaml/кода —
        # удобно гонять один и тот же конфиг локально (cpu) и на проде (cuda) без правки yaml.
        override = os.getenv("MEME_EMB_DEVICE")
        if override:
            self.device = override


@dataclass
class ProjectionConfig:
    """
    Приведение native_dim модели к целевой размерности вектора мема.

    method:
      - "none"       — без изменений, target_dim игнорируется
      - "truncate"   — обрезка до первых target_dim компонент (валидно для моделей,
                        обученных с Matryoshka Representation Learning: e5-mrl, bge-m3,
                        jina-embeddings-v3/clip-v2 — у них первые компоненты уже
                        самодостаточны, обрезка не требует пересчёта индекса)
      - "random"     — фиксированная случайная ортогональная проекция (Johnson–Lindenstrauss),
                        подходит для моделей без MRL, если важна скорость/память,
                        качество почти не теряется при target_dim >= 256
      - "pca"        — заранее обученная PCA-проекция (артефакт fit на корпусе, см. Whitening)
    """

    enabled: bool = True
    target_dim: int = 512                      # 256 или 512, как просили в задаче
    method: Literal["none", "truncate", "random", "pca"] = "truncate"
    seed: int = 42
    pca_artifact_path: Optional[str] = None     # путь к сохранённой PCA-матрице, если method="pca"


@dataclass
class EmbedderConfig:
    """Полный конфиг одного эмбеддера — то, что передаётся в create_embedder()."""

    embedder_type: EmbedderType
    model: ModelSpec
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    normalize: bool = True                      # L2-normalize (п.2.1 репорта, обязательно)
    batch_size: int = 32

    # используется только для embedder_type == MULTIMODAL (вариант B)
    caption_model: Optional[ModelSpec] = None
    caption_prompt_template: str = (
        "Ты анализируешь мем. На изображении присутствует текст (OCR): \"{ocr_text}\".\n"
        "Опиши:\n"
        "1) Что происходит на изображении (сцена, персонажи, эмоции).\n"
        "2) В чём заключается юмор/ирония/отсылка мема.\n"
        "3) К какой категории мемов он относится (сарказм, абсурд, политика, отношения, работа и т.д.).\n"
        "Ответ дай в 2-4 предложениях, без вводных фраз."
    )
    caption_max_new_tokens: int = 200
    caption_temperature: float = 0.0             # 0.0 => детерминированная генерация (стабильность, п.3.4)

    # версии для метаданных индекса (п.2.4 репорта) — обязательны к логированию
    model_version: str = "v1"
    preprocessing_version: str = "v1"

    def fingerprint(self) -> str:
        """Короткий хэш конфига — удобно писать в metadata вектора вместе с version-полями."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Пресеты моделей грузятся из config/models.yaml (см. также ModelSpec.trust_remote_code
# и .revision — задаются явно на каждую модель в yaml).
#
# Путь к yaml можно переопределить через .env: MEME_EMB_MODELS_YAML=/path/to/models.yaml
#
# Использование:
#   from meme_embeddings.config import PRESETS
#   cfg = PRESETS["multilingual-e5-large"]
# --------------------------------------------------------------------------------------

def _model_spec_from_dict(d: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=d["name"],
        hf_id=d["hf_id"],
        native_dim=d.get("native_dim", 0),
        max_length=d.get("max_length", 128),
        device=d.get("device", "cpu"),
        dtype=d.get("dtype", "float32"),
        trust_remote_code=bool(d.get("trust_remote_code", False)),
        revision=d.get("revision"),
        query_prefix=d.get("query_prefix", ""),
        passage_prefix=d.get("passage_prefix", ""),
        extra=d.get("extra", {}) or {},
    )


def _embedder_config_from_dict(d: Dict[str, Any]) -> EmbedderConfig:
    proj_dict = d.get("projection", {}) or {}
    projection = ProjectionConfig(
        enabled=proj_dict.get("enabled", True),
        target_dim=proj_dict.get("target_dim", 512),
        method=proj_dict.get("method", "truncate"),
        seed=proj_dict.get("seed", 42),
        pca_artifact_path=proj_dict.get("pca_artifact_path"),
    )
    caption_model = _model_spec_from_dict(d["caption_model"]) if d.get("caption_model") else None

    kwargs: Dict[str, Any] = dict(
        embedder_type=EmbedderType(d["embedder_type"]),
        model=_model_spec_from_dict(d["model"]),
        projection=projection,
        caption_model=caption_model,
        model_version=d.get("model_version", "v1"),
        preprocessing_version=d.get("preprocessing_version", "v1"),
    )
    if "normalize" in d:
        kwargs["normalize"] = d["normalize"]
    if "batch_size" in d:
        kwargs["batch_size"] = d["batch_size"]
    if "caption_prompt_template" in d:
        kwargs["caption_prompt_template"] = d["caption_prompt_template"]
    if "caption_max_new_tokens" in d:
        kwargs["caption_max_new_tokens"] = d["caption_max_new_tokens"]
    if "caption_temperature" in d:
        kwargs["caption_temperature"] = d["caption_temperature"]

    return EmbedderConfig(**kwargs)


def load_presets_from_yaml(path: Optional[Path] = None) -> Dict[str, EmbedderConfig]:
    """
    Читает config/models.yaml и собирает словарь {preset_name: EmbedderConfig}.

    Порядок определения пути:
      1. явный аргумент path
      2. .env / окружение: MEME_EMB_MODELS_YAML
      3. встроенный meme_embeddings/config/models.yaml
    """
    if path is None:
        override = env.MODELS_YAML_OVERRIDE
        path = Path(override) if override else _DEFAULT_MODELS_YAML

    if not Path(path).exists():
        logger.warning("models.yaml не найден по пути %s — PRESETS будет пустым", path)
        return {}

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Для загрузки пресетов из yaml нужен пакет PyYAML: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    presets_raw = raw.get("presets", {})
    presets: Dict[str, EmbedderConfig] = {}
    for name, cfg_dict in presets_raw.items():
        try:
            presets[name] = _embedder_config_from_dict(cfg_dict)
        except Exception:
            logger.exception("Не удалось разобрать пресет '%s' из %s, пропускаю", name, path)

    logger.info("Загружено %d пресетов из %s: %s", len(presets), path, list(presets.keys()))
    return presets


PRESETS: Dict[str, EmbedderConfig] = load_presets_from_yaml()


def get_preset(name: str) -> EmbedderConfig:
    if name not in PRESETS:
        raise KeyError(
            f"Пресет '{name}' не найден. Доступные: {list(PRESETS.keys())}. "
            "Для кастомной модели создайте EmbedderConfig вручную (embedder_type=CUSTOM)."
        )
    return PRESETS[name]


def reload_presets(path: Optional[Path] = None) -> Dict[str, EmbedderConfig]:
    """Перечитать models.yaml в рантайме (например, после правки конфига) и обновить PRESETS."""
    global PRESETS
    PRESETS = load_presets_from_yaml(path)
    return PRESETS
