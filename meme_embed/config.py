"""
meme_embed.config
=================
Configuration, model registry, environment/.env loading, logging, and lightweight
(torch-free) image I/O utilities.

Depends only on ``numpy`` / ``Pillow`` (+ optional ``PyYAML`` / ``python-dotenv``),
so it imports instantly and is trivial to unit-test. Heavy work lives in
``meme_embed.embedders``.
"""


import base64
import io
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Optional dependencies                                                        #
# --------------------------------------------------------------------------- #
try:
    from dotenv import load_dotenv, find_dotenv
    _HAS_DOTENV = True
except ImportError:  # pragma: no cover
    _HAS_DOTENV = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False

# A meme image may be a PIL image, numpy array (H,W,C / H,W), or a string
# (file path, URL, or base64 / data-URI).
ImageInput = Union[Image.Image, np.ndarray, str]

ENV_PREFIX = "MEME_EMBEDDER_"

__all__ = [
    "EmbedderConfig",
    "ImageInput",
    "get_logger",
    "setup_logging",
    "env",
    "load_registry",
    "resolve_config",
    "to_pil",
    "to_pil_batch",
    "l2_normalize",
]

# --------------------------------------------------------------------------- #
# .env discovery (BEFORE logging, so MEME_EMBEDDER_LOG_LEVEL from .env applies) #
# --------------------------------------------------------------------------- #
_DOTENV_PATH: Optional[str] = None
_DOTENV_LOADED: bool = False
if _HAS_DOTENV:
    _DOTENV_PATH = find_dotenv(usecwd=True) or None
    _DOTENV_LOADED = load_dotenv(_DOTENV_PATH)

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
_LOGGER_NAME = "meme_embedder"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: Union[str, int, None] = None) -> logging.Logger:
    """Configure the library logger. Level falls back to ``MEME_EMBEDDER_LOG_LEVEL``."""
    log = logging.getLogger(_LOGGER_NAME)
    if level is None:
        level = os.getenv(ENV_PREFIX + "LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    log.setLevel(level)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        log.addHandler(handler)
    log.propagate = False
    return log


logger = setup_logging()

# Report .env discovery result now that the logger exists.
if not _HAS_DOTENV:
    logger.debug("python-dotenv not installed; .env loading is disabled.")
elif _DOTENV_LOADED and _DOTENV_PATH:
    logger.info(".env found and loaded: %s", _DOTENV_PATH)
else:
    logger.info(".env not found; using process environment / defaults.")


# --------------------------------------------------------------------------- #
# Environment helpers                                                          #
# --------------------------------------------------------------------------- #
def env(key: str, default: Any = None, cast: Optional[type] = None) -> Any:
    """Read ``MEME_EMBEDDER_<KEY>`` from the environment (after .env is loaded)."""
    raw = os.getenv(ENV_PREFIX + key, default)
    if raw is not None and cast is not None and not isinstance(raw, cast):
        try:
            if cast is bool:
                return str(raw).strip().lower() in {"1", "true", "yes", "on"}
            return cast(raw)
        except (ValueError, TypeError):
            return default
    return raw


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class EmbedderConfig:
    """Everything needed to build & run one embedder.

    ``embedding_dim`` is the *target* output size. For Matryoshka-trained models
    (jina-clip-v2, nomic, e5-mistral, ...) we slice + re-normalize to that size,
    which is why 256 / 512 come essentially for free.

    ``extra`` may carry model-specific knobs, e.g.
    ``extra={"encode_kwargs": {"task": "retrieval.passage"}}`` for jina-clip.
    """

    name: str = "custom"
    type: str = "clip"                      # "clip" | "multimodal"
    model_name: str = "openai/clip-vit-base-patch32"
    api: str = "clip_features"              # "clip_features" | "encode_methods" | "vlm_pool"

    embedding_dim: Optional[int] = 512      # desired output dim (Matryoshka truncation)
    native_dim: Optional[int] = None        # model's raw dim, if known
    matryoshka: bool = True                 # is truncation semantically safe?
    normalize: bool = True                  # L2-normalize outputs (cosine-ready)

    device: Optional[str] = None            # "cuda" | "cpu" | None(auto)
    dtype: str = "auto"                     # "auto" | "float32" | "float16" | "bfloat16"
    batch_size: int = 32
    max_text_length: int = 77
    padding: str = "longest"                # transformers padding ("max_length" for SigLIP)
    trust_remote_code: bool = False
    compile: bool = False                   # torch.compile the model

    pooling: str = "mean"                   # multimodal token pooling: "mean" | "last"
    query_prompt: Optional[str] = None      # optional instruction prefix for search queries
    image_prompt: str = "A meme image."     # short prompt used by VLM image encoding

    cache_dir: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------- #
    def resolved_dim(self) -> Optional[int]:
        if self.embedding_dim and self.native_dim:
            return min(self.embedding_dim, self.native_dim)
        return self.embedding_dim or self.native_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "EmbedderConfig":
        known = set(cls.__dataclass_fields__)  # noqa
        payload = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        payload.setdefault("name", name)
        cfg = cls(**payload)
        cfg.extra.update(extra)
        return cfg


# --------------------------------------------------------------------------- #
# Model registry                                                               #
# --------------------------------------------------------------------------- #
_BUILTIN_REGISTRY: dict = {
    "default_model": "clip-jina-v2",
    "models": {
        "clip-jina-v2": {
            "type": "clip",
            "model_name": "jinaai/jina-clip-v2",
            "api": "encode_methods",
            "embedding_dim": 512,
            "native_dim": 1024,
            "matryoshka": True,
            "trust_remote_code": True,
            "max_text_length": 77,
        },
        "clip-openai-b32": {
            "type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "api": "clip_features",
            "embedding_dim": 512,
            "native_dim": 512,
            "matryoshka": False,
        },
        "siglip2-256": {
            "type": "clip",
            "model_name": "google/siglip2-base-patch16-256",
            "api": "clip_features",
            "embedding_dim": 256,
            "native_dim": 768,
            "matryoshka": False,
            "padding": "max_length",
            "max_text_length": 64,
        },
        "gemma-mm": {
            "type": "multimodal",
            "model_name": "google/gemma-3-4b-it",
            "api": "vlm_pool",
            "embedding_dim": 512,
            "matryoshka": False,
            "pooling": "mean",
            "max_text_length": 128,
        },
    },
}


def load_registry(path: Union[str, Path, None] = None) -> dict:
    """Load the model registry from YAML, else the built-in one.

    Resolution order: explicit ``path`` -> ``MEME_EMBEDDER_CONFIG`` -> built-in.
    """
    path = path or env("CONFIG")
    if path and Path(path).exists():
        if not _HAS_YAML:
            logger.warning("PyYAML not installed; ignoring %s and using built-in registry.", path)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                reg = yaml.safe_load(fh) or {}
            reg.setdefault("models", {})
            logger.info("Model registry loaded from %s (%d models).", path, len(reg["models"]))
            return reg
    logger.debug("Using built-in model registry (%d models).", len(_BUILTIN_REGISTRY["models"]))
    return _BUILTIN_REGISTRY


def resolve_config(
    name: Optional[str] = None,
    registry: Optional[dict] = None,
    **overrides: Any,
) -> EmbedderConfig:
    """Resolve a named config from the registry and apply env + explicit overrides."""
    registry = registry or load_registry()
    name = name or env("MODEL") or registry.get("default_model")
    models = registry.get("models", {})
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(models)}")

    cfg = EmbedderConfig.from_dict(name, models[name])

    # Environment overrides (logged so the effective runtime is transparent).
    if (d := env("DEVICE")):
        cfg.device = d
        logger.info("Env override MEME_EMBEDDER_DEVICE=%s", d)
    if (dt := env("DTYPE")):
        cfg.dtype = dt
        logger.info("Env override MEME_EMBEDDER_DTYPE=%s", dt)
    if (cd := env("CACHE_DIR")):
        cfg.cache_dir = cd
        logger.debug("Env override MEME_EMBEDDER_CACHE_DIR=%s", cd)

    # Explicit keyword overrides win over everything.
    for key, val in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            cfg.extra[key] = val

    logger.debug("Resolved config '%s': %s", cfg.name, cfg.to_dict())
    return cfg


# --------------------------------------------------------------------------- #
# Math                                                                         #
# --------------------------------------------------------------------------- #
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization (works on 1-D and 2-D arrays)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


# --------------------------------------------------------------------------- #
# Image I/O (torch-free)                                                       #
# --------------------------------------------------------------------------- #
def _ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
    if np.issubdtype(a.dtype, np.floating):
        a = (a * 255.0) if a.max() <= 1.0 + 1e-6 else a
        a = np.clip(a, 0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return Image.fromarray(a).convert("RGB")


def _load_from_str(s: str) -> Image.Image:
    s = s.strip()
    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    if not s.startswith(("http://", "https://")) and Path(s).exists():
        return Image.open(s).convert("RGB")
    if s.startswith(("http://", "https://")):
        try:
            import requests
            resp = requests.get(s, timeout=20)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:  # noqa - fall back to urllib
            from urllib.request import urlopen
            with urlopen(s, timeout=20) as r:  # nosec
                return Image.open(io.BytesIO(r.read())).convert("RGB")
    try:
        return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    except Exception as exc:  # noqa
        raise ValueError(f"Could not interpret string as image (path/URL/base64): {s[:60]}...") from exc


def to_pil(image: ImageInput) -> Image.Image:
    """Coerce a PIL image / numpy array / string into an RGB ``PIL.Image``."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return _ndarray_to_pil(image)
    if isinstance(image, (str, Path)):
        return _load_from_str(str(image))
    raise TypeError(f"Unsupported image type: {type(image)!r}. Use PIL.Image, np.ndarray, or str.")


def to_pil_batch(images: Iterable[ImageInput]) -> list[Image.Image]:
    return [to_pil(im) for im in images]