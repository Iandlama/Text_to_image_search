"""
logging_setup.py
=================
Единая настройка логирования для модуля.

Использование в любом файле модуля:
    from .logging_setup import get_logger
    logger = get_logger(__name__)

Уровень и формат управляются через .env / переменные окружения (см. env.py):
    LOG_LEVEL=DEBUG|INFO|WARNING|ERROR   (по умолчанию INFO)
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger("meme_embeddings")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Вернуть логгер, настроенный под конфигурацию модуля (ленивая инициализация)."""
    _configure_root()
    return logging.getLogger(name)
