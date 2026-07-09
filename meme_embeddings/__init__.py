"""
meme_embeddings — модуль генерации эмбеддингов для поиска мемов по описанию (Этап 2).

Быстрый старт:
    from meme_embeddings import create_embedder

    embedder = create_embedder("multilingual-e5-large")
    vectors = embedder.encode(texts=["когда понедельник, а работы конца не видно"])
"""

from . import env  # noqa: F401  (побочный эффект: подгружает .env раньше всего остального)
from .logging_setup import get_logger
from .config import (
    EmbedderConfig,
    EmbedderType,
    ModelSpec,
    ProjectionConfig,
    PRESETS,
    get_preset,
    load_presets_from_yaml,
    reload_presets,
)
from .embedders import (
    BaseEmbedder,
    ClipTextEmbedder,
    TextEmbedder,
    MultimodalDescriptionEmbedder,
    CustomEmbedder,
    create_embedder,
)

__all__ = [
    "get_logger",
    "EmbedderConfig",
    "EmbedderType",
    "ModelSpec",
    "ProjectionConfig",
    "PRESETS",
    "get_preset",
    "load_presets_from_yaml",
    "reload_presets",
    "BaseEmbedder",
    "ClipTextEmbedder",
    "TextEmbedder",
    "MultimodalDescriptionEmbedder",
    "CustomEmbedder",
    "create_embedder",
]

__version__ = "0.1.0"
