"""
core/save_results.py — Сохранение результатов симуляции и оптимизации.
"""

import os
import json
import csv
import numpy as np
from datetime import datetime


def save_csv_results(filepath, params, metrics):
    """
    Сохранение результатов в CSV (как в текущем app.py).
    
    Параметры:
        filepath — путь к файлу
        params — dict с параметрами (R, h, T, n, B, ...)
        metrics — dict с метриками (Q, Lawson, P_fusion, ...)
    """
    rows = []
    
    # Параметры
    for key, val in params.items():
        rows.append(("Параметр", key, val))
    
    rows.append(("=", "=", "="))
    
    # Метрики
    for key, val in metrics.items():
        rows.append(("Метрика", key, val))
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Категория", "Параметр", "Значение"])
        for row in rows:
            writer.writerow(row)


def save_json_results(filepath, params, metrics, config=None):
    """
    Сохранение результатов в JSON.
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "metrics": metrics,
    }
    if config:
        # Конвертируем numpy-типы
        clean_config = {}
        for k, v in config.items():
            if isinstance(v, (np.floating, np.integer)):
                clean_config[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_config[k] = v.tolist()
            else:
                clean_config[k] = v
        data["config"] = clean_config
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_history_csv(filepath, history):
    """
    Сохранение истории симуляции в CSV (для графиков).
    """
    if not history or not history.get("step"):
        return
    
    keys = list(history.keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(history["step"])
        for i in range(n):
            row = []
            for k in keys:
                val = history[k][i] if i < len(history[k]) else ""
                row.append(val)
            writer.writerow(row)


def generate_run_dir(base_dir="results"):
    """Создаёт папку для нового прогона с timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
