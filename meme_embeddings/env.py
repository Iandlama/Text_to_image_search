"""
env.py
======
Загрузка переменных окружения из .env при импорте модуля.

Поддерживаемые переменные (все опциональны):
    LOG_LEVEL              — уровень логирования (см. logging_setup.py)
    MEME_EMB_DEVICE         — устройство по умолчанию, переопределяет "cpu" в yaml (cuda/cpu/mps)
    MEME_EMB_MODELS_YAML    — путь к своему models.yaml вместо встроенного config/models.yaml
    HF_HOME / HF_TOKEN      — стандартные переменные HuggingFace (кеш моделей, приватные веса)

Если пакет python-dotenv не установлен — модуль не падает, просто пропускает подгрузку
.env файла и полагается на переменные, уже выставленные в окружении ОС.
"""

from __future__ import annotations

import os


def load_env(dotenv_path: str | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv не установлен — работаем только с окружением ОС, это не ошибка
        return

    load_dotenv(dotenv_path=dotenv_path, override=False)


# Загружаем .env сразу при первом импорте пакета
load_env()

DEVICE_OVERRIDE = os.getenv("MEME_EMB_DEVICE")
MODELS_YAML_OVERRIDE = os.getenv("MEME_EMB_MODELS_YAML")
