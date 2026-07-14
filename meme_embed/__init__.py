"""
meme_embed
==========
Importable library for embedding memes (image + OCR text) for semantic search.

Quick start
-----------
    from meme_embed import load_embedder

    emb = load_embedder("clip-jina-v2", embedding_dim=256)   # or load_embedder()
    image_vec, text_vec = emb.encode(image, ocr_text)        # separate vectors
    query_vec = emb.encode_query(text="cat screaming into a phone")

All embedders share one interface: ``encode``, ``encode_image``, ``encode_text``,
``encode_query`` and ``batch_encode``. Images accept PIL / np.ndarray / str.
"""


from .config import (
    EmbedderConfig,
    ImageInput,
    get_logger,
    setup_logging,
    load_registry,
    resolve_config,
    to_pil,
    to_pil_batch,
    l2_normalize,
    env,
)
from .embedders import (
    BaseMemeEmbedder,
    ClipEmbedder,
    MultimodalEmbedder,
    build_embedder,
    load_embedder,
    from_pretrained,
)

__version__ = "0.1.0"

__all__ = [
    # entry points
    "load_embedder",
    "from_pretrained",
    "build_embedder",
    # classes
    "BaseMemeEmbedder",
    "ClipEmbedder",
    "MultimodalEmbedder",
    "EmbedderConfig",
    # config / utils
    "resolve_config",
    "load_registry",
    "setup_logging",
    "get_logger",
    "to_pil",
    "to_pil_batch",
    "l2_normalize",
    "env",
    "ImageInput",
    "__version__",
]