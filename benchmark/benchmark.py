"""
benchmark.py
============
Бенчмарк кандидатов text-embedding моделей для Этапа 2 («meme_embeddings»).

ЗАЧЕМ ЭТОТ ФАЙЛ
---------------
Архитектура (Вариант B) зафиксирована: изображение+OCR -> VLM -> rich_description ->
СИЛЬНЫЙ ТЕКСТОВЫЙ ЭНКОДЕР. Открытый вопрос - какой именно текстовый энкодер использовать
на втором шаге. Этот скрипт прогоняет несколько кандидатов на реальном датасете мемов,
считает retrieval-метрики (nDCG@10, Recall@K, MRR) и сравнивает их МЕЖДУ СОБОЙ -
но никогда не сравнивает «сырые» векторы разных моделей друг с другом напрямую.

КАК ИЗБЕГАЕМ ПРОБЛЕМЫ РАЗНЫХ ЭМБЕДДИНГОВЫХ ПРОСТРАНСТВ
-------------------------------------------------------
Каждая модель прогоняется в ПОЛНОЙ изоляции: свои query-векторы, свои doc-векторы,
косинусная близость считается ТОЛЬКО между векторами одной и той же модели. На выходе
получаем только скалярные метрики (Recall@K, MRR, nDCG@10 - все в [0, 1]), и уже ЭТИ
числа кладём в одну таблицу/график. Ни один эмбеддинг модели A никогда не участвует
в вычислениях с эмбеддингом модели B - иначе сравнение было бы бессмысленным (разные
базисы, разная размерность, разная нормировка обучающих данных).

ДАТАСЕТ (простая архитектура, без явного query-list)
------------------------------------------------------
Датасет - metadata.jsonl вида:
    {"id": "...", "title": "...", "image_desc": "...", "meaning": "..."}
плюс images/{id}.webp.

Явного списка запросов нет, поэтому запрос генерируется из самих данных:
  - query  = поле `meaning`   (то, ЧТО пользователь хочет сказать/найти - ближе всего
             к тому, как реальный пользователь опишет мем словами)
  - doc    = title + image_desc + OCR-текст с картинки (то, что реально будет
             индексироваться в проде)
  - relevant(query_i) = doc_i (самоидентификация: мем должен найти сам себя среди
             остальных N-1 мемов корпуса). Это самый простой защитимый способ получить
             retrieval-метрики без ручной разметки релевантности.

Важно: `image_desc` НЕ используется в запросе - иначе запрос содержал бы прямое описание
картинки и retrieval стал бы тривиальным (утечка ответа в вопрос).

КЛАСТЕР «ПЛОХОГО КАЧЕСТВА»
---------------------------
Часть мемов в реальности несколько раз пересохранена как JPEG при репостах - из-за этого
возникает искажение (блочные артефакты, смаз мелкого текста) и, как следствие, отдельный
«кластер» плохих эмбеддингов. Чтобы это измерить, мы:
  1) берём случайное подмножество мемов (augmentation.fraction),
  2) прогоняем их через симуляцию многократного JPEG re-encode (degrade_image_jpeg),
  3) заново гоняем OCR по деградированной картинке,
  4) считаем retrieval-метрики на этом подмножестве в ДВУХ вариантах корпуса -
     чистом и «загрязнённом» - и сравниваем (quality gap),
  5) считаем cluster_leakage_score: не «слипаются» ли деградированные мемы друг
     с другом в эмбеддинг-пространстве сильнее, чем случайные чистые мемы
     (сигнал того самого паразитного кластера).

ИСПОЛЬЗОВАНИЕ
-------------
    python benchmark.py --config models_config.yaml
    python benchmark.py --config models_config.yaml --models multilingual-e5-base bge-m3
    python benchmark.py --config models_config.yaml --export-model bge-m3 --export-path chosen_model.yaml

Кэшируется всё дорогое: OCR (по id+вариант), деградированные картинки (на диске),
эмбеддинги (по модели+id+вид+вариант, .npy на диске) - повторные запуски почти бесплатны.

ЗАВИСИМОСТИ: numpy, pillow, pyyaml, tqdm, pandas, matplotlib + сам пакет meme_embeddings
             + модуль my_ocr с функцией get_ocr(img: PIL.Image.Image) -> str (Этап 1).
"""

import dotenv
dotenv.load_dotenv()

from __future__ import annotations

import argparse
import gc
import io
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

try:
    from my_ocr import get_ocr  # Этап 1: изображение -> распознанный текст
except ImportError as e:
    raise ImportError(
        "Не найден модуль `my_ocr` с функцией get_ocr(img: PIL.Image.Image) -> str.\n"
        "Он должен быть реализован на Этапе 1 (OCR) и лежать в PYTHONPATH рядом с этим скриптом."
    ) from e

from meme_embeddings import (
    EmbedderConfig,
    EmbedderType,
    ModelSpec,
    ProjectionConfig,
    create_embedder,
    get_logger,
)

logger = get_logger(__name__)


# ============================================================================
# 1. Датасет - максимально простая структура
# ============================================================================

@dataclass
class MemeItem:
    id: str
    title: str
    image_desc: str
    meaning: str
    image_path: Path


def load_dataset(
    metadata_path: Path,
    images_dir: Path,
    sample_size: int,
    seed: int,
    image_ext: str = "webp",
) -> List[MemeItem]:
    """Читает metadata.jsonl, проверяет наличие картинки, берёт случайные N штук."""
    all_items: List[MemeItem] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_path = images_dir / f"{rec['id']}.{image_ext}"
            if not img_path.exists():
                continue
            all_items.append(
                MemeItem(
                    id=rec["id"],
                    title=rec.get("title", "") or "",
                    image_desc=rec.get("image_desc", "") or "",
                    meaning=rec.get("meaning", "") or "",
                    image_path=img_path,
                )
            )

    if not all_items:
        raise ValueError(
            f"Не найдено ни одного валидного элемента ({metadata_path}, images_dir={images_dir}, ext={image_ext}). "
            "Проверьте пути и расширение картинок."
        )

    rng = random.Random(seed)
    if sample_size < len(all_items):
        items = rng.sample(all_items, sample_size)
    else:
        items = list(all_items)
        rng.shuffle(items)

    logger.info("Датасет: %d/%d элементов выбрано (seed=%d)", len(items), len(all_items), seed)
    return items


def build_query_text(item: MemeItem) -> str:
    """Запрос = то, что пользователь хочет найти словами. image_desc сюда не подмешиваем."""
    return item.meaning.strip()


def build_doc_text(item: MemeItem, ocr_text: str) -> str:
    """Документ мема = всё, что реально попадёт в индекс: заголовок + описание сцены + OCR."""
    parts = [item.title.strip(), item.image_desc.strip(), (ocr_text or "").strip()]
    return " ".join(p for p in parts if p)


# ============================================================================
# 2. Кэш OCR (JSON на диске)
# ============================================================================

class OcrCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, str] = {}
        if cache_path.exists():
            try:
                self._data = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("OCR-кэш повреждён, начинаю с пустого: %s", cache_path)

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


def run_ocr_cached(image_path: Path, cache_key: str, cache: OcrCache) -> str:
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    img = Image.open(image_path).convert("RGB")
    text = get_ocr(img) or ""
    cache.set(cache_key, text)
    return text


# ============================================================================
# 3. Аугментация «плохого качества» (симуляция многократного JPEG re-encode)
# ============================================================================

def degrade_image_jpeg(
    img: Image.Image,
    cycles: int,
    quality_range: Tuple[int, int],
    rng: random.Random,
) -> Image.Image:
    """
    Имитирует типичную деградацию мема при многократных репостах в мессенджерах/соцсетях:
    каждый цикл - небольшое случайное down/upscale + пересохранение в JPEG с низким
    случайным качеством. После нескольких циклов мелкий текст на картинке (важный для OCR)
    начинает "плыть", появляются блочные артефакты.
    """
    out = img.convert("RGB")
    for _ in range(cycles):
        q = rng.randint(*quality_range)
        scale = rng.uniform(0.7, 0.95)
        w, h = out.size
        w2, h2 = max(1, int(w * scale)), max(1, int(h * scale))
        out = out.resize((w2, h2), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


def ensure_degraded_image(
    item: MemeItem,
    out_dir: Path,
    cycles: int,
    quality_range: Tuple[int, int],
    seed: int,
) -> Path:
    """Генерирует (или берёт из кэша на диске) деградированную версию картинки мема."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item.id}_degraded.jpg"
    if out_path.exists():
        return out_path
    rng = random.Random(f"{item.id}::{seed}")
    img = Image.open(item.image_path).convert("RGB")
    degraded = degrade_image_jpeg(img, cycles=cycles, quality_range=quality_range, rng=rng)
    degraded.save(out_path, format="JPEG", quality=90)
    return out_path


# ============================================================================
# 4. Кэш эмбеддингов (.npy на диске, по модели+id+виду+варианту)
# ============================================================================

def _safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


class EmbeddingCache:
    def __init__(self, cache_dir: Path, model_name: str):
        self.dir = cache_dir / _safe_name(model_name)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, item_id: str, kind: str, variant: str) -> Path:
        return self.dir / f"{_safe_name(item_id)}__{kind}__{variant}.npy"

    def get(self, item_id: str, kind: str, variant: str) -> Optional[np.ndarray]:
        p = self._path(item_id, kind, variant)
        return np.load(p) if p.exists() else None

    def set(self, item_id: str, kind: str, variant: str, vec: np.ndarray) -> None:
        np.save(self._path(item_id, kind, variant), vec.astype(np.float32))


def encode_with_cache(
    embedder,
    items: Sequence[MemeItem],
    texts: Sequence[str],
    kind: str,
    variant: str,
    cache: EmbeddingCache,
    is_query: bool,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """Кодирует тексты батчами, максимально переиспользуя уже посчитанные .npy из кэша."""
    vectors: List[Optional[np.ndarray]] = [None] * len(items)
    todo: List[int] = []
    for i, it in enumerate(items):
        cached = cache.get(it.id, kind, variant)
        if cached is None:
            todo.append(i)
        else:
            vectors[i] = cached

    if todo:
        for start in tqdm(range(0, len(todo), batch_size), desc=desc, leave=False):
            idx_batch = todo[start : start + batch_size]
            batch_texts = [texts[i] for i in idx_batch]
            vecs = embedder.encode(texts=batch_texts, is_query=is_query)
            for j, i in enumerate(idx_batch):
                vectors[i] = vecs[j]
                cache.set(items[i].id, kind, variant, vecs[j])

    return np.stack(vectors, axis=0)  # type: ignore[arg-type]


# ============================================================================
# 5. Метрики retrieval (self-id ground truth, простая и корректная схема)
# ============================================================================

def cosine_sim_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    q = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-9, None)
    d = docs / np.clip(np.linalg.norm(docs, axis=1, keepdims=True), 1e-9, None)
    return q @ d.T


def retrieval_metrics(sims: np.ndarray, relevant_idx: np.ndarray, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    """
    sims: [n_queries, n_docs] косинусная близость (одна и та же модель по обеим осям!)
    relevant_idx[i]: индекс единственного релевантного документа для запроса i.
    """
    n = sims.shape[0]
    ranks = np.empty(n, dtype=int)
    for i in range(n):
        order = np.argsort(-sims[i])
        ranks[i] = int(np.where(order == relevant_idx[i])[0][0]) + 1  # 1-indexed

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks <= k))
    out["mrr"] = float(np.mean(1.0 / ranks))
    ndcg_k = 10
    ndcg = np.where(ranks <= ndcg_k, 1.0 / np.log2(ranks + 1), 0.0)
    out[f"ndcg@{ndcg_k}"] = float(np.mean(ndcg))
    out["mean_rank"] = float(np.mean(ranks))
    out["n"] = n
    return out


def cluster_leakage_score(doc_vectors: np.ndarray, degraded_mask: np.ndarray, rng: random.Random) -> float:
    """
    Проверяет, не "слипаются" ли деградированные (плохое качество) мемы друг с другом
    в эмбеддинг-пространстве сильнее, чем случайные чистые мемы между собой.
    Положительное большое значение = есть паразитный "кластер плохого качества".
    """
    sims = cosine_sim_matrix(doc_vectors, doc_vectors)
    deg_idx = np.where(degraded_mask)[0]
    clean_idx = np.where(~degraded_mask)[0]
    if len(deg_idx) < 2 or len(clean_idx) < 2:
        return float("nan")

    def mean_offdiag(idx: np.ndarray) -> float:
        sub = sims[np.ix_(idx, idx)]
        m = len(idx)
        return float((sub.sum() - np.trace(sub)) / (m * m - m))

    deg_deg = mean_offdiag(deg_idx)
    k = min(len(deg_idx), len(clean_idx))
    control_idx = np.array(rng.sample(list(clean_idx), k))
    control_control = mean_offdiag(control_idx)
    return deg_deg - control_control


# ============================================================================
# 6. Конфигурация (YAML)
# ============================================================================

@dataclass
class DatasetCfg:
    metadata_path: Path
    images_dir: Path
    image_ext: str = "webp"
    sample_size: int = 300
    seed: int = 42


@dataclass
class AugmentationCfg:
    enabled: bool = True
    fraction: float = 0.25
    cycles: int = 4
    quality_min: int = 10
    quality_max: int = 35
    seed: int = 123


@dataclass
class BenchmarkConfig:
    dataset: DatasetCfg
    augmentation: AugmentationCfg
    cache_dir: Path
    output_dir: Path
    batch_size: int = 32


def load_config(path: Path) -> Tuple[BenchmarkConfig, List[dict]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    ds = raw["dataset"]
    dataset_cfg = DatasetCfg(
        metadata_path=Path(ds["metadata_path"]),
        images_dir=Path(ds["images_dir"]),
        image_ext=ds.get("image_ext", "webp"),
        sample_size=int(ds.get("sample_size", 300)),
        seed=int(ds.get("seed", 42)),
    )

    aug = raw.get("augmentation", {})
    aug_cfg = AugmentationCfg(
        enabled=bool(aug.get("enabled", True)),
        fraction=float(aug.get("fraction", 0.25)),
        cycles=int(aug.get("cycles", 4)),
        quality_min=int(aug.get("quality_min", 10)),
        quality_max=int(aug.get("quality_max", 35)),
        seed=int(aug.get("seed", 123)),
    )

    cfg = BenchmarkConfig(
        dataset=dataset_cfg,
        augmentation=aug_cfg,
        cache_dir=Path(raw.get("cache_dir", ".benchmark_cache")),
        output_dir=Path(raw.get("output_dir", "benchmark_output")),
        batch_size=int(raw.get("batch_size", 32)),
    )

    model_entries = raw["models"]
    return cfg, model_entries


# ----------------------------------------------------------------------------
# Построение EmbedderConfig из "сырой" записи YAML.
#
# ВНИМАНИЕ: поля ModelSpec/ProjectionConfig/EmbedderConfig ниже взяты из фактического
# использования атрибутов в meme_embeddings/embedders.py (config.model.hf_id,
# config.projection.target_dim и т.д. - это реально то, что читает BaseEmbedder).
# Если в вашем актуальном config.py сигнатура датаклассов отличается - поправьте
# только эту функцию, остальной бенчмарк не изменится.
# ----------------------------------------------------------------------------

_EMBEDDER_TYPE_MAP = {
    "text": EmbedderType.TEXT,
    "clip": EmbedderType.CLIP,
}


def build_embedder_config_from_entry(entry: dict) -> EmbedderConfig:
    kind = entry.get("embedder_type", "text").lower()
    embedder_type = _EMBEDDER_TYPE_MAP.get(kind)
    if embedder_type is None:
        raise ValueError(
            f"Модель '{entry.get('name')}': embedder_type='{kind}' не поддерживается бенчмарком "
            f"(доступно: {list(_EMBEDDER_TYPE_MAP)}). Multimodal (Вариант B целиком, с VLM-капшнингом) "
            "сюда сознательно не включён - это отдельный дорогой прогон, здесь сравниваются "
            "кандидаты именно на роль текстового энкодера второго шага."
        )

    native_dim = int(entry["native_dim"])
    target_dim = int(entry.get("target_dim", 512))
    target_dim = min(target_dim, native_dim)
    proj_method = entry.get("projection_method", "truncate" if native_dim > target_dim else "none")

    try:
        model_spec = ModelSpec(
            name=entry["name"],
            hf_id=entry["hf_id"],
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
            revision=entry.get("revision"),
            device=entry.get("device", "cpu"),
            native_dim=native_dim,
            max_length=entry.get("max_length"),
            query_prefix=entry.get("query_prefix", "") or "",
            passage_prefix=entry.get("passage_prefix", "") or "",
        )
        projection = ProjectionConfig(
            enabled=proj_method != "none",
            method=proj_method,
            target_dim=target_dim,
            seed=int(entry.get("projection_seed", 42)),
        )
        econfig = EmbedderConfig(
            embedder_type=embedder_type,
            model=model_spec,
            projection=projection,
            normalize=True,
            batch_size=int(entry.get("batch_size", 32)),
            model_version=entry.get("model_version", "benchmark-v1"),
            preprocessing_version=entry.get("preprocessing_version", "benchmark-v1"),
        )
    except TypeError as e:
        raise TypeError(
            f"Не удалось создать EmbedderConfig для модели '{entry.get('name')}': {e}\n"
            "Похоже, реальная сигнатура ModelSpec/ProjectionConfig/EmbedderConfig в вашем "
            "meme_embeddings/config.py отличается от предполагаемой здесь. Проверьте актуальные "
            "поля датаклассов и поправьте build_embedder_config_from_entry()."
        ) from e

    return econfig


# ============================================================================
# 7. Результат по одной модели
# ============================================================================

@dataclass
class ModelResult:
    name: str
    hf_id: str = ""
    output_dim: int = -1
    load_time_s: float = 0.0
    encode_time_s: float = 0.0
    metrics_clean: Dict[str, float] = field(default_factory=dict)
    metrics_degraded_subset_baseline: Dict[str, float] = field(default_factory=dict)  # тот же subset, чистый корпус
    metrics_degraded_subset_mixed: Dict[str, float] = field(default_factory=dict)     # тот же subset, "грязный" корпус
    cluster_leakage: float = float("nan")
    error: Optional[str] = None


def _free_model_memory(embedder) -> None:
    del embedder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ============================================================================
# 8. Основной прогон
# ============================================================================

def run_benchmark(cfg: BenchmarkConfig, model_entries: List[dict], models_to_run: Optional[List[str]] = None) -> List[ModelResult]:
    items = load_dataset(
        cfg.dataset.metadata_path,
        cfg.dataset.images_dir,
        cfg.dataset.sample_size,
        cfg.dataset.seed,
        cfg.dataset.image_ext,
    )

    ocr_cache = OcrCache(cfg.cache_dir / "ocr_cache.json")

    # --- OCR оригиналов ---
    orig_ocr: Dict[str, str] = {}
    for item in tqdm(items, desc="OCR (оригинальные картинки)"):
        orig_ocr[item.id] = run_ocr_cached(item.image_path, f"{item.id}::orig", ocr_cache)
    ocr_cache.save()

    # --- Подвыборка "плохого качества" + деградация + повторный OCR ---
    degraded_ocr: Dict[str, str] = {}
    degraded_ids: set = set()
    if cfg.augmentation.enabled and cfg.augmentation.fraction > 0:
        rng_aug = random.Random(cfg.augmentation.seed)
        n_degrade = max(2, int(len(items) * cfg.augmentation.fraction))
        degraded_items = rng_aug.sample(items, min(n_degrade, len(items)))
        degraded_ids = {it.id for it in degraded_items}

        deg_img_dir = cfg.cache_dir / "degraded_images"
        for item in tqdm(degraded_items, desc="Деградация JPEG + OCR (плохое качество)"):
            deg_path = ensure_degraded_image(
                item, deg_img_dir, cfg.augmentation.cycles,
                (cfg.augmentation.quality_min, cfg.augmentation.quality_max), cfg.augmentation.seed,
            )
            degraded_ocr[item.id] = run_ocr_cached(deg_path, f"{item.id}::degraded", ocr_cache)
        ocr_cache.save()

    queries = [build_query_text(it) for it in items]
    docs_clean = [build_doc_text(it, orig_ocr[it.id]) for it in items]
    docs_mixed = [
        build_doc_text(it, degraded_ocr[it.id] if it.id in degraded_ids else orig_ocr[it.id])
        for it in items
    ]
    degraded_mask = np.array([it.id in degraded_ids for it in items])
    deg_positions = np.where(degraded_mask)[0]

    entries = model_entries
    if models_to_run:
        wanted = set(models_to_run)
        entries = [e for e in entries if e["name"] in wanted]
        missing = wanted - {e["name"] for e in entries}
        if missing:
            logger.warning("Модели не найдены в конфиге и будут пропущены: %s", missing)

    results: List[ModelResult] = []
    relevant_idx = np.arange(len(items))
    rng_leak = random.Random(cfg.augmentation.seed + 1)

    for entry in entries:
        name = entry["name"]
        logger.info("=== Модель: %s (%s) ===", name, entry.get("hf_id"))
        result = ModelResult(name=name, hf_id=entry.get("hf_id", ""))

        try:
            econfig = build_embedder_config_from_entry(entry)
        except Exception as e:
            logger.error("Пропуск '%s': ошибка конфигурации: %s", name, e)
            result.error = f"config error: {e}"
            results.append(result)
            continue

        try:
            t0 = time.time()
            embedder = create_embedder(econfig)
            result.load_time_s = time.time() - t0
            result.output_dim = embedder.output_dim
        except Exception as e:
            logger.error("Пропуск '%s': не удалось загрузить модель: %s", name, e)
            result.error = f"load error: {e}"
            results.append(result)
            continue

        embedder_ref = embedder
        try:
            emb_cache = EmbeddingCache(cfg.cache_dir / "embeddings", name)
            t0 = time.time()
            q_vecs = encode_with_cache(embedder, items, queries, "query", "orig", emb_cache, True, cfg.batch_size, f"{name}: запросы")
            d_clean = encode_with_cache(embedder, items, docs_clean, "doc", "clean", emb_cache, False, cfg.batch_size, f"{name}: документы (чистые)")
            d_mixed = encode_with_cache(embedder, items, docs_mixed, "doc", "mixed", emb_cache, False, cfg.batch_size, f"{name}: документы (со следами деградации)")
            result.encode_time_s = time.time() - t0

            sims_clean = cosine_sim_matrix(q_vecs, d_clean)
            result.metrics_clean = retrieval_metrics(sims_clean, relevant_idx)

            if len(deg_positions) >= 2:
                result.metrics_degraded_subset_baseline = retrieval_metrics(
                    sims_clean[deg_positions], relevant_idx[deg_positions]
                )
                sims_mixed = cosine_sim_matrix(q_vecs, d_mixed)
                result.metrics_degraded_subset_mixed = retrieval_metrics(
                    sims_mixed[deg_positions], relevant_idx[deg_positions]
                )
                result.cluster_leakage = cluster_leakage_score(d_mixed, degraded_mask, rng_leak)
        except Exception as e:
            logger.exception("Модель '%s' упала на этапе кодирования/метрик: %s", name, e)
            result.error = f"encode/metrics error: {e}"
        finally:
            _free_model_memory(embedder_ref)

        results.append(result)

    return results


# ============================================================================
# 9. Сводная таблица, экспорт, графики
# ============================================================================

def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "model": r.name,
            "hf_id": r.hf_id,
            "output_dim": r.output_dim,
            "load_time_s": round(r.load_time_s, 1),
            "encode_time_s": round(r.encode_time_s, 1),
            "error": r.error or "",
        }
        for k, v in r.metrics_clean.items():
            if k != "n":
                row[f"clean_{k}"] = round(v, 4)
        for k, v in r.metrics_degraded_subset_baseline.items():
            if k != "n":
                row[f"degsubset_baseline_{k}"] = round(v, 4)
        for k, v in r.metrics_degraded_subset_mixed.items():
            if k != "n":
                row[f"degsubset_mixed_{k}"] = round(v, 4)
        if r.metrics_degraded_subset_baseline and r.metrics_degraded_subset_mixed:
            row["quality_gap_ndcg10"] = round(
                row.get("degsubset_baseline_ndcg@10", 0) - row.get("degsubset_mixed_ndcg@10", 0), 4
            )
        row["cluster_leakage"] = round(r.cluster_leakage, 4) if r.cluster_leakage == r.cluster_leakage else None
        rows.append(row)
    return pd.DataFrame(rows)


def export_model_config(name: str, model_entries: List[dict], out_path: Path) -> None:
    entry = next((e for e in model_entries if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"Модель '{name}' не найдена в конфиге")
    payload = {"models": [entry]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("Конфиг модели '%s' экспортирован в %s", name, out_path)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df["error"] == ""].copy()
    if ok.empty:
        logger.warning("Нет успешных прогонов - графики не строятся")
        return
    ok = ok.sort_values("clean_ndcg@10", ascending=False)

    # 1) Главная метрика: nDCG@10 по всем моделям
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(ok))))
    ax.barh(ok["model"], ok["clean_ndcg@10"], color="#4C72B0")
    ax.set_xlabel("nDCG@10 (чистый корпус)")
    ax.set_title("Сравнение моделей: nDCG@10")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "ndcg10_comparison.png", dpi=150)
    plt.close(fig)

    # 2) Recall@1/5/10 сгруппировано
    recall_cols = [c for c in ["clean_recall@1", "clean_recall@5", "clean_recall@10"] if c in ok.columns]
    if recall_cols:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(ok))))
        x = np.arange(len(ok))
        width = 0.8 / len(recall_cols)
        for i, col in enumerate(recall_cols):
            ax.bar(x + i * width, ok[col], width=width, label=col.replace("clean_", ""))
        ax.set_xticks(x + width * (len(recall_cols) - 1) / 2)
        ax.set_xticklabels(ok["model"], rotation=45, ha="right")
        ax.set_ylabel("Recall")
        ax.set_title("Recall@K по моделям")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "recall_comparison.png", dpi=150)
        plt.close(fig)

    # 3) Устойчивость к плохому качеству: nDCG@10 baseline vs mixed на том же подмножестве
    gap_cols = ["degsubset_baseline_ndcg@10", "degsubset_mixed_ndcg@10"]
    if all(c in ok.columns for c in gap_cols):
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(ok))))
        x = np.arange(len(ok))
        width = 0.35
        ax.bar(x - width / 2, ok[gap_cols[0]], width=width, label="чистый корпус (тот же subset)")
        ax.bar(x + width / 2, ok[gap_cols[1]], width=width, label="с деградацией (JPEG re-encode)")
        ax.set_xticks(x)
        ax.set_xticklabels(ok["model"], rotation=45, ha="right")
        ax.set_ylabel("nDCG@10 на подвыборке 'плохого качества'")
        ax.set_title("Устойчивость к деградации картинок")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "degradation_robustness.png", dpi=150)
        plt.close(fig)

    # 4) Cluster leakage score
    if "cluster_leakage" in ok.columns and ok["cluster_leakage"].notna().any():
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(ok))))
        colors = ["#C44E52" if v and v > 0.02 else "#55A868" for v in ok["cluster_leakage"]]
        ax.barh(ok["model"], ok["cluster_leakage"].fillna(0), color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("cluster_leakage_score (выше = сильнее 'слипаются' плохие мемы)")
        ax.set_title("Паразитный кластер 'плохого качества'")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(output_dir / "cluster_leakage.png", dpi=150)
        plt.close(fig)

    # 5) Trade-off: размерность vs качество
    if "output_dim" in ok.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(ok["output_dim"], ok["clean_ndcg@10"], s=60)
        for _, row in ok.iterrows():
            ax.annotate(row["model"], (row["output_dim"], row["clean_ndcg@10"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Размерность вектора")
        ax.set_ylabel("nDCG@10")
        ax.set_title("Компромисс: размерность vs качество")
        fig.tight_layout()
        fig.savefig(output_dir / "dim_vs_quality.png", dpi=150)
        plt.close(fig)

    logger.info("Графики сохранены в %s", output_dir)


# ============================================================================
# 10. CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк text-embedding моделей для поиска мемов")
    parser.add_argument("--config", type=Path, default=Path("models_config.yaml"))
    parser.add_argument("--models", nargs="*", default=None, help="Прогнать только указанные модели (по полю name)")
    parser.add_argument("--export-model", type=str, default=None, help="Только экспортировать конфиг одной модели и выйти")
    parser.add_argument("--export-path", type=Path, default=Path("exported_model_config.yaml"))
    args = parser.parse_args()

    cfg, model_entries = load_config(args.config)

    if args.export_model:
        export_model_config(args.export_model, model_entries, args.export_path)
        return

    results = run_benchmark(cfg, model_entries, models_to_run=args.models)
    df = summarize_results(results)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_dir / "benchmark_results.csv", index=False)
    (cfg.output_dir / "benchmark_results.md").write_text(df.to_markdown(index=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False) + "\n")

    make_plots(df, cfg.output_dir)
    print(f"Результаты сохранены в {cfg.output_dir}/ (csv, md, png-графики)")


if __name__ == "__main__":
    main()
