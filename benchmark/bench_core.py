"""
bench_core.py
=============
Building blocks for the meme-retrieval benchmark: a tiny dataset abstraction,
OCR extraction + caching, JPEG-degradation augmentation ("bad quality memes"),
model-agnostic embedding computation with disk caching, retrieval metrics, and -
importantly - **embedding-space safety** so vectors from different models/channels
are never silently compared.

Depends on: numpy, Pillow, tqdm, and the local ``meme_embed`` package.
OCR uses ``from my_ocr import get_ocr``  (get_ocr(img: PIL.Image) -> str).
"""


import hashlib
import io
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **k):  # minimal fallback
        return x if x is not None else iter(())

# meme_embed is the embedding library from part 1.
from meme_embed import l2_normalize

logger = logging.getLogger("meme_embedder.benchmark")


# --------------------------------------------------------------------------- #
# OCR  (from my_ocr import get_ocr)                                           #
# --------------------------------------------------------------------------- #
def _load_ocr_fn() -> Optional[Callable[[Image.Image], str]]:
    try:
        from my_ocr import get_ocr  # type: ignore
        return get_ocr
    except Exception as exc:  # noqa
        logger.warning("Could not import get_ocr from my_ocr (%s). OCR text will be empty.", exc)
        return None


# --------------------------------------------------------------------------- #
# Dataset  (deliberately minimal)                                             #
# --------------------------------------------------------------------------- #
@dataclass
class MemeItem:
    id: str
    image_path: Path
    title: str = ""
    image_desc: str = ""
    meaning: str = ""
    ocr: str = ""  # filled in later by OcrCache

    def query_text(self, field_name: str) -> str:
        return (getattr(self, field_name, "") or "").strip()


class MemeDataset:
    """Reads ``metadata.jsonl`` + ``images/{id}.{ext}`` and samples N items.

    metadata line: {"id","title","image_desc","meaning"}.
    Ground truth is identity: a query built from item *i* should retrieve item *i*.
    """

    def __init__(
        self,
        path: str | Path,
        n_samples: int = 200,
        seed: int = 42,
        query_field: str = "meaning",
        image_dir: str = "images",
        image_ext: str = "webp",
    ):
        self.root = Path(path)
        self.query_field = query_field
        self.image_ext = image_ext.lstrip(".")
        img_root = self.root / image_dir

        meta_file = self.root / "metadata.jsonl"
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.jsonl not found in {self.root}")

        rows = []
        with open(meta_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                img = img_root / f"{d['id']}.{self.image_ext}"
                if not img.exists():
                    continue
                if not (d.get(query_field) or "").strip():  # need a query source
                    continue
                rows.append(MemeItem(
                    id=d["id"], image_path=img, title=d.get("title", ""),
                    image_desc=d.get("image_desc", ""), meaning=d.get("meaning", ""),
                ))

        if not rows:
            raise RuntimeError(f"No usable items (missing images or empty '{query_field}').")

        rng = random.Random(seed)
        rng.shuffle(rows)
        self.items: list[MemeItem] = rows[: min(n_samples, len(rows))]
        logger.info("Dataset: %d usable, sampled %d (seed=%d).", len(rows), len(self.items), seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> MemeItem:
        return self.items[i]

    def load_image(self, item: MemeItem) -> Image.Image:
        return Image.open(item.image_path).convert("RGB")

    @property
    def ids(self) -> list[str]:
        return [it.id for it in self.items]

    @property
    def queries(self) -> list[str]:
        return [it.query_text(self.query_field) for it in self.items]

    def fingerprint(self) -> str:
        h = hashlib.sha1(("|".join(self.ids) + f"#{self.query_field}").encode()).hexdigest()
        return h[:12]


# --------------------------------------------------------------------------- #
# OCR cache                                                                   #
# --------------------------------------------------------------------------- #
class OcrCache:
    """Caches OCR text per image id to a JSON file (OCR is expensive)."""

    def __init__(self, cache_dir: str | Path):
        self.path = Path(cache_dir) / "ocr.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if self.path.exists():
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))

    def attach(self, dataset: MemeDataset, progress: bool = True) -> None:
        ocr_fn = _load_ocr_fn()
        dirty = False
        it = dataset.items
        for item in tqdm(it, desc="OCR", disable=not progress):
            if item.id in self._cache:
                item.ocr = self._cache[item.id]
                continue
            text = ""
            if ocr_fn is not None:
                try:
                    text = ocr_fn(dataset.load_image(item)) or ""
                except Exception as exc:  # noqa
                    logger.debug("OCR failed for %s: %s", item.id, exc)
            item.ocr = text
            self._cache[item.id] = text
            dirty = True
        if dirty:
            self.path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    def content_hash(self, dataset: MemeDataset) -> str:
        blob = "|".join(self._cache.get(i, "") for i in dataset.ids)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Augmentation - "bad quality memes" (repeated JPEG recompression)            #
# --------------------------------------------------------------------------- #
def jpeg_degrade(img: Image.Image, quality: int = 20, n_recompress: int = 3,
                 downscale: float = 1.0) -> Image.Image:
    """Simulate the compression artifacts of memes re-saved as JPEG many times.

    Optionally downscale→upscale first (mimics reposts through chat apps), then
    round-trip through JPEG ``n_recompress`` times at low ``quality``.
    """
    out = img.convert("RGB")
    if downscale and downscale < 1.0:
        w, h = out.size
        small = out.resize((max(1, int(w * downscale)), max(1, int(h * downscale))), Image.BILINEAR)
        out = small.resize((w, h), Image.BILINEAR)
    for _ in range(max(1, n_recompress)):
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


# --------------------------------------------------------------------------- #
# Embeddings + space safety                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Embeddings:
    """A matrix of vectors tagged with the space that produced them.

    ``space`` = f"{model}|{dim}" - the ONLY thing allowed to be compared with
    another Embeddings of the same tag. Channel is informational (image/ocr/...).
    """
    vectors: np.ndarray
    model: str
    dim: int
    channel: str

    @property
    def space(self) -> str:
        return f"{self.model}|{self.dim}"

    def __len__(self) -> int:
        return len(self.vectors)


def check_same_space(a: Embeddings, b: Embeddings) -> None:
    """Guard against comparing vectors from different models/dims (the classic bug)."""
    if a.space != b.space:
        raise ValueError(
            f"Refusing to compare different embedding spaces: "
            f"{a.channel}[{a.space}] vs {b.channel}[{b.space}]. "
            f"Cross-space cosine is meaningless - use the same model & dim."
        )


class EmbeddingStore:
    """Computes and disk-caches embedding matrices (npz) keyed by content."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir) / "emb"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *parts) -> Path:
        raw = "|".join(str(p) for p in parts)
        return self.dir / (hashlib.sha1(raw.encode()).hexdigest()[:16] + ".npz")

    def get_or_compute(self, tag: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        p = self._key(tag)
        if p.exists():
            logger.debug("cache hit  %s -> %s", tag, p.name)
            return np.load(p)["v"]
        logger.debug("cache miss %s", tag)
        v = np.asarray(compute(), dtype=np.float32)
        np.savez_compressed(p, v=v)
        return v


def build_channels(
    embedder,
    dataset: MemeDataset,
    store: EmbeddingStore,
    *,
    model_name: str,
    dim: int,
    ds_fp: str,
    ocr_fp: str,
    channels: Sequence[str],
    fused_alpha: float,
    query_field: str,
    progress: bool = True,
) -> dict[str, Embeddings]:
    """Compute (cached) image / ocr / fused corpus embeddings + query embeddings."""
    images_loader = lambda: [dataset.load_image(it) for it in dataset.items]  # noqa
    ocr_texts = [it.ocr for it in dataset.items]
    queries = dataset.queries
    out: dict[str, Embeddings] = {}

    def emb(tag, fn):
        return store.get_or_compute(tag, fn)

    need_img = "image" in channels or "fused" in channels
    need_ocr = "ocr" in channels or "fused" in channels

    img_v = ocr_v = None
    if need_img:
        img_v = emb(f"{model_name}|{dim}|image|{ds_fp}",
                    lambda: embedder.encode_image(images_loader(), normalize=True))
        out["image"] = Embeddings(np.asarray(img_v), model_name, dim, "image")
    if need_ocr:
        ocr_v = emb(f"{model_name}|{dim}|ocr|{ds_fp}|{ocr_fp}",
                    lambda: embedder.encode_text(ocr_texts, normalize=True))
        out["ocr"] = Embeddings(np.asarray(ocr_v), model_name, dim, "ocr")
    if "fused" in channels:
        fused = l2_normalize(fused_alpha * img_v + (1.0 - fused_alpha) * ocr_v)
        out["fused"] = Embeddings(fused, model_name, dim, "fused")

    # Queries live in the same space (text tower). For CLIP/VLM, encode_query(text)
    # with no prompt == encode_text; batch it for speed. Depends on the query field.
    q_v = emb(f"{model_name}|{dim}|query|{ds_fp}|{query_field}",
              lambda: embedder.encode_text(queries, normalize=True))
    out["query"] = Embeddings(np.asarray(q_v), model_name, dim, "query")
    return out


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _ranks_of_gt(sims: np.ndarray, gt: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """1-based rank of the ground-truth column for each query row."""
    s = sims.copy()
    if exclude_self:
        np.fill_diagonal(s, -np.inf)
    order = np.argsort(-s, axis=1)                    # best first
    ranks = np.empty(len(gt), dtype=np.int64)
    for i, g in enumerate(gt):
        ranks[i] = int(np.where(order[i] == g)[0][0]) + 1
    return ranks


def retrieval_metrics(
    query: Embeddings, corpus: Embeddings, gt: Sequence[int],
    ks: Sequence[int] = (1, 5, 10), ndcg_k: int = 10, exclude_self: bool = False,
) -> dict[str, float]:
    """Recall@k, MRR, nDCG@k for single-relevant-doc retrieval. Space-checked."""
    check_same_space(query, corpus)
    gt = np.asarray(gt)
    sims = query.vectors @ corpus.vectors.T
    ranks = _ranks_of_gt(sims, gt, exclude_self=exclude_self)
    out = {f"recall@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    within = ranks <= ndcg_k
    out[f"ndcg@{ndcg_k}"] = float(np.mean(np.where(within, 1.0 / np.log2(ranks + 1), 0.0)))
    out["median_rank"] = float(np.median(ranks))
    return out


def mean_pair_cosine(a: Embeddings, b: Embeddings) -> float:
    """Mean cosine between aligned rows a[i]·b[i] (same space)."""
    check_same_space(a, b)
    return float(np.mean(np.sum(a.vectors * b.vectors, axis=1)))


def offdiag_mean_cosine(e: Embeddings) -> float:
    """Mean similarity among distinct items - high value = tight (possibly artifact) cluster."""
    s = e.vectors @ e.vectors.T
    n = len(e)
    if n < 2:
        return 0.0
    return float((s.sum() - np.trace(s)) / (n * (n - 1)))


def topk_sets(query: Embeddings, corpus: Embeddings, k: int) -> list[set]:
    check_same_space(query, corpus)
    sims = query.vectors @ corpus.vectors.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return [set(row.tolist()) for row in order]


def mean_jaccard(sets_a: list[set], sets_b: list[set]) -> float:
    vals = []
    for a, b in zip(sets_a, sets_b):
        u = len(a | b)
        vals.append((len(a & b) / u) if u else 0.0)
    return float(np.mean(vals)) if vals else 0.0
