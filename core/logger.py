"""
GAK-WaveCAD — центральный логгер.

Каждый модуль получает логгер через get_logger(name).
Логи пишутся в консоль и в файл logs/gak_wavecad.log.
"""

import os
import logging
from datetime import datetime


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Возвращает настроенный логгер по имени модуля."""
    logger = logging.getLogger(name)

    # Не настраиваем повторно, если уже есть хендлеры
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Консоль
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Файл
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gak_wavecad.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
