"""
meme_embed.embedders
=====================
Meme embedding models. Every embedder produces vectors for **images** and for
**OCR text** in the *same* semantic space (so image and OCR vectors are directly
comparable), plus a dedicated ``encode_query`` for search-time queries.

    BaseMemeEmbedder (ABC)
        ├── ClipEmbedder         # CLIP-family: separate image & text towers, shared space
        └── MultimodalEmbedder   # VLM (Gemma-style): single fused multimodal space (R&D)

Only multimodal image↔text models are used - never a text-only encoder - because
the primary target is the *image* embedding, with OCR handled by the same model's
text tower so both live in one comparable space.

``torch``/``transformers`` import lazily, so this module imports fine without them;
a clear error is raised only when a model is actually run.
"""


import inspect
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import numpy as np

from .config import (
    EmbedderConfig,
    ImageInput,
    get_logger,
    l2_normalize,
    load_registry,
    resolve_config,
    to_pil_batch,
)

logger = get_logger()

__all__ = [
    "BaseMemeEmbedder",
    "ClipEmbedder",
    "MultimodalEmbedder",
    "build_embedder",
    "load_embedder",
    "from_pretrained",
]


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to run embedders. Install torch (and transformers)."
        ) from exc


def _filtered_kwargs(fn, desired: dict) -> dict:
    """Keep only kwargs that ``fn`` explicitly declares as named parameters.

    This makes calls robust across model versions with different signatures
    (e.g. jina-clip's encode_* takes ``normalize_embeddings`` / ``truncate_dim``,
    not ``normalize``).
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in desired.items() if k in params}


def _as_list(x) -> tuple[list, bool]:
    """Return (list, was_single). PIL/str/single-array count as one item."""
    from PIL import Image
    if isinstance(x, (list, tuple)):
        return list(x), False
    if isinstance(x, (str, Image.Image)):
        return [x], True
    if isinstance(x, np.ndarray):
        return ([x], True) if x.ndim <= 3 else (list(x), False)
    return [x], True


# --------------------------------------------------------------------------- #
# Abstract base                                                                #
# --------------------------------------------------------------------------- #
class BaseMemeEmbedder(ABC):
    """Common device/dtype/batch/normalization plumbing + public API.

    Subclasses implement only ``_embed_images`` and ``_embed_texts`` (each
    returning ``[B, D]`` as a torch.Tensor or np.ndarray). Batching, Matryoshka
    truncation, L2-normalization, image coercion and timing are handled here.
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.model = None
        self.processor = None

    # -- lifecycle --------------------------------------------------------- #
    def _resolve_runtime(self):
        torch = _require_torch()
        dev = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.config.dtype != "auto":
            dtype = getattr(torch, self.config.dtype)
        elif dev.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.device, self.dtype = dev, dtype

    @abstractmethod
    def _load(self) -> None:
        """Load model + processor onto ``self.device`` with ``self.dtype``."""

    def _ensure_loaded(self):
        if not self._loaded:
            self._resolve_runtime()
            t0 = time.perf_counter()
            self._load()
            self._loaded = True
            logger.info(
                "Loaded '%s' (%s) on %s/%s in %.2fs",
                self.config.name, self.config.model_name, self.device,
                str(self.dtype).replace("torch.", ""), time.perf_counter() - t0,
            )

    def warmup(self):
        """Eagerly load weights (useful before serving / benchmarking)."""
        self._ensure_loaded()
        return self

    # -- primitives (subclass) -------------------------------------------- #
    @abstractmethod
    def _embed_images(self, pil_images: list): ...

    @abstractmethod
    def _embed_texts(self, texts: list[str]): ...

    # -- shared helpers ---------------------------------------------------- #
    @contextmanager
    def _autocast(self):
        torch = _require_torch()
        if self.device and self.device.startswith("cuda") and self.dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                yield
        else:
            yield

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        if hasattr(x, "detach"):  # torch.Tensor
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def _postprocess(self, arr: np.ndarray, normalize: Optional[bool]) -> np.ndarray:
        arr = np.atleast_2d(self._to_numpy(arr))
        dim = self.config.resolved_dim()
        if dim and arr.shape[1] > dim:
            if not self.config.matryoshka:
                logger.warning(
                    "Truncating %d→%d dims on a non-Matryoshka model ('%s'); "
                    "quality may drop. Set embedding_dim=native_dim to disable.",
                    arr.shape[1], dim, self.config.name,
                )
            arr = arr[:, :dim]
        if normalize is None:
            normalize = self.config.normalize
        if normalize:
            arr = l2_normalize(arr)
        return arr.astype(np.float32)

    def _run_batched(self, items: list, embed_fn) -> np.ndarray:
        bs = max(1, self.config.batch_size)
        chunks = [self._to_numpy(embed_fn(items[i:i + bs])) for i in range(0, len(items), bs)]
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.config.resolved_dim() or 0), np.float32)

    @staticmethod
    def _squeeze(arr: np.ndarray, was_single: bool) -> np.ndarray:
        return arr[0] if was_single and arr.shape[0] == 1 else arr

    # -- public API -------------------------------------------------------- #
    def encode_image(
        self, image: Union[ImageInput, Sequence[ImageInput]], normalize: Optional[bool] = None
    ) -> np.ndarray:
        """Embed one image or a list of images. Returns [D] or [N, D]."""
        self._ensure_loaded()
        items, single = _as_list(image)
        arr = self._run_batched(to_pil_batch(items), self._embed_images)
        return self._squeeze(self._postprocess(arr, normalize), single)

    def encode_text(
        self, text: Union[str, Sequence[str]], normalize: Optional[bool] = None
    ) -> np.ndarray:
        """Embed OCR text (one string or a list). Returns [D] or [N, D]."""
        self._ensure_loaded()
        single = isinstance(text, str)
        items = [text] if single else list(text)
        arr = self._run_batched(items, self._embed_texts)
        return self._squeeze(self._postprocess(arr, normalize), single)

    def encode_query(
        self,
        text: Optional[str] = None,
        image: Optional[ImageInput] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Encode a search query: text, image, or both (late-fused mean). Returns [D]."""
        self._ensure_loaded()
        parts = []
        if text is not None:
            prompt = (self.config.query_prompt or "") + text
            parts.append(self._postprocess(self._to_numpy(self._embed_texts([prompt])), True)[0])
        if image is not None:
            parts.append(self.encode_image(image, normalize=True))
        if not parts:
            raise ValueError("encode_query needs at least one of `text` or `image`.")
        vec = parts[0] if len(parts) == 1 else np.mean(np.stack(parts, 0), axis=0)
        norm = self.config.normalize if normalize is None else normalize
        return l2_normalize(vec[None])[0] if norm else vec.astype(np.float32)

    def encode(self, image: ImageInput, ocr_text: str, normalize: Optional[bool] = None):
        """Core meme interface -> (image_vec, text_vec), kept as *separate* vectors."""
        return self.encode_image(image, normalize), self.encode_text(ocr_text, normalize)

    def batch_encode(
        self,
        images: Sequence[ImageInput],
        ocr_texts: Sequence[str],
        normalize: Optional[bool] = None,
    ):
        """Vectorized meme encoding -> (image_matrix [N,D], text_matrix [N,D])."""
        if len(images) != len(ocr_texts):
            raise ValueError("images and ocr_texts must have equal length.")
        return self.encode_image(list(images), normalize), self.encode_text(list(ocr_texts), normalize)


# --------------------------------------------------------------------------- #
# CLIP-family embedder                                                         #
# --------------------------------------------------------------------------- #
class ClipEmbedder(BaseMemeEmbedder):
    """CLIP / SigLIP / jina-clip. Image & text towers project into a shared space."""

    def _load(self):
        torch = _require_torch()
        from transformers import AutoModel, AutoProcessor

        kw = dict(trust_remote_code=self.config.trust_remote_code, cache_dir=self.config.cache_dir)
        self.model = AutoModel.from_pretrained(
            self.config.model_name, torch_dtype=self.dtype, **kw
        ).to(self.device).eval()
        if self.config.api != "encode_methods":  # jina exposes .encode_* directly
            self.processor = AutoProcessor.from_pretrained(self.config.model_name, **kw)
        if self.config.compile:
            self.model = torch.compile(self.model)

    def _encode_methods_kwargs(self, fn) -> dict:
        """Signature-safe kwargs for jina-style ``encode_image``/``encode_text``."""
        desired = {
            "normalize_embeddings": False,   # we L2-normalize ourselves in _postprocess
            "device": self.device,
            "batch_size": self.config.batch_size,
        }
        if (dim := self.config.resolved_dim()):
            desired["truncate_dim"] = dim    # native Matryoshka truncation, if supported
        desired.update(self.config.extra.get("encode_kwargs", {}))
        return _filtered_kwargs(fn, desired)

    def _embed_images(self, pil_images: list):
        torch = _require_torch()
        if self.config.api == "encode_methods":
            fn = self.model.encode_image
            with torch.inference_mode():
                return fn(pil_images, **self._encode_methods_kwargs(fn))
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.inference_mode(), self._autocast():
            return self.model.get_image_features(**inputs)

    def _embed_texts(self, texts: list[str]):
        torch = _require_torch()
        if self.config.api == "encode_methods":
            fn = self.model.encode_text
            with torch.inference_mode():
                return fn(list(texts), **self._encode_methods_kwargs(fn))
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=self.config.padding,
            truncation=True,
            max_length=self.config.max_text_length,
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            return self.model.get_text_features(**inputs)


# --------------------------------------------------------------------------- #
# Multimodal (VLM) embedder - single fused space, R&D track                    #
# --------------------------------------------------------------------------- #
class MultimodalEmbedder(BaseMemeEmbedder):
    """Gemma-style multimodal LM used as an embedder via masked mean-pooling.

    Image, text, and image+text all map into one shared space. Besides the standard
    ``encode_image``/``encode_text``, it offers ``encode_fused`` (interleaved
    image + "OCR: ...") producing a single meme vector - the Variant-2 approach.
    """

    def _load(self):
        torch = _require_torch()
        import transformers
        from transformers import AutoProcessor

        kw = dict(trust_remote_code=self.config.trust_remote_code, cache_dir=self.config.cache_dir)
        self.processor = AutoProcessor.from_pretrained(self.config.model_name, **kw)

        # VLMs (Gemma-3, etc.) load cleanly as *ImageTextToText* and accept pixel_values;
        # plain AutoModel is a fallback for CLIP-ish backbones. Overridable via extra.model_class.
        wanted = self.config.extra.get("model_class")
        candidates = [wanted] if wanted else ["AutoModelForImageTextToText", "AutoModel"]
        last_err = None
        for cls_name in candidates:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(
                    self.config.model_name, torch_dtype=self.dtype, **kw
                ).to(self.device).eval()
                logger.debug("Loaded multimodal model via %s.", cls_name)
                break
            except Exception as exc:  # noqa - try the next class
                last_err = exc
                logger.debug("Loading via %s failed: %s", cls_name, exc)
        else:
            raise last_err or RuntimeError("Could not load a multimodal model class.")

        if self.config.compile:
            self.model = torch.compile(self.model)

    # -- input building ---------------------------------------------------- #
    def _image_token(self) -> str:
        """Best-effort placeholder token the processor uses to mark an image."""
        for attr in ("image_token", "boi_token"):
            tok = getattr(self.processor, attr, None) or getattr(
                getattr(self.processor, "tokenizer", None), attr, None
            )
            if tok:
                return tok
        return "<start_of_image>"  # Gemma-3 default

    def _build_inputs(self, texts: list[str], images: Optional[list] = None):
        """Tokenize text (+optional images) into model inputs.

        For image inputs we use the chat template so the processor expands the
        image placeholder into the correct number of soft image tokens - passing
        ``text=`` + ``images=`` without that placeholder is what raised
        'Prompt contained 0 image tokens'.
        """
        # Text-only: a plain processor call is fine (no image-token requirement).
        if images is None:
            inputs = self.processor(
                text=list(texts), return_tensors="pt",
                padding=True, truncation=True, max_length=self.config.max_text_length,
            )
            return inputs.to(self.device)

        # Image (+text): prefer the chat template.
        if hasattr(self.processor, "apply_chat_template"):
            conversations = [
                [{"role": "user", "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": tx},
                ]}]
                for im, tx in zip(images, texts)
            ]
            try:
                inputs = self.processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
                return inputs.to(self.device)
            except Exception as exc:  # noqa - fall back to manual token insertion
                logger.debug("apply_chat_template failed (%s); using manual image token.", exc)

        # Fallback: manually prepend the image placeholder token.
        tok = self._image_token()
        texts = [f"{tok}{tx}" for tx in texts]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    # -- pooling / forward ------------------------------------------------- #
    def _pool(self, hidden, mask):
        torch = _require_torch()
        if self.config.pooling == "last":  # last non-pad token
            idx = mask.sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0)), idx]
        m = mask.unsqueeze(-1).to(hidden.dtype)  # masked mean
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _forward(self, inputs):
        torch = _require_torch()
        with torch.inference_mode(), self._autocast():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else out.last_hidden_state
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device)
        return self._pool(hidden, mask)

    def _embed_texts(self, texts: list[str]):
        return self._forward(self._build_inputs(list(texts)))

    def _embed_images(self, pil_images: list):
        prompts = [self.config.image_prompt] * len(pil_images)
        return self._forward(self._build_inputs(prompts, images=pil_images))

    def encode_fused(self, image: ImageInput, ocr_text: str, normalize: Optional[bool] = None) -> np.ndarray:
        """Interleaved image + OCR -> a single fused meme embedding. Returns [D]."""
        self._ensure_loaded()
        pil = to_pil_batch([image])
        prompt = f"{self.config.image_prompt} OCR: {ocr_text}"
        return self._postprocess(self._forward(self._build_inputs([prompt], images=pil)), normalize)[0]


# --------------------------------------------------------------------------- #
# Factory helpers                                                              #
# --------------------------------------------------------------------------- #
_REGISTRY = {"clip": ClipEmbedder, "multimodal": MultimodalEmbedder}


def build_embedder(config: EmbedderConfig) -> BaseMemeEmbedder:
    if config.type not in _REGISTRY:
        raise ValueError(f"Unknown embedder type '{config.type}'. Use one of {sorted(_REGISTRY)}.")
    return _REGISTRY[config.type](config)


def load_embedder(name: Optional[str] = None, *, registry_path=None, **overrides) -> BaseMemeEmbedder:
    """One-liner entry point: ``load_embedder("clip-jina-v2", embedding_dim=256)``."""
    cfg = resolve_config(name, registry=load_registry(registry_path), **overrides)
    return build_embedder(cfg)


def from_pretrained(model_name: str, type: str = "clip", **kwargs) -> BaseMemeEmbedder:
    """Build an embedder for a fully custom HF model with no registry entry."""
    return build_embedder(EmbedderConfig(name="custom", type=type, model_name=model_name, **kwargs))