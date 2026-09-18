#!/usr/bin/env python3
"""
WaveCAD — Acoustic Monitor Integration Test
Точка входа: импортирует AcousticMonitor и прогоняет полный тест.

Запуск:
    python run_acoustic_test.py
"""

import os
import numpy as np

from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    print("=" * 64)
    print("  WaveCAD — Acoustic Monitor Integration Test")
    print("  v0.4 — 4-mode spherical shell analyzer")
    print("=" * 64)

    # 1. Создаём монитор с конфигом по умолчанию
    monitor = AcousticMonitor()

    print("\n  Опорные частоты мод:")
    for (l, mtype) in monitor.modes:
        name = f"{mtype}_l{l}"
        f = monitor.f0[(l, mtype)]
        print(f"    {name:<13} : {f:.1f} Гц")

    # 2. Прогоняем анализ всех мод
    results = monitor.analyze_all_modes()

    # 3. Вывод результатов
    print("\n  Результаты анализа:")
    print("  " + "-" * 60)
    for key, r in results.items():
        print(f"  {key:<13} | {r['status']:<16} | "
              f"f = {r['f_measured']:.3f} Гц | "
              f"Δf = {r['delta_f']:.3f} Гц | "
              f"σ = {r['sigma_MPa']:.2f} МПа | "
              f"{r['n_peaks']} пик(ов)")
    print("  " + "-" * 60)

    # 4. Тест с напряжением
    print("\n  Тест с внутренним напряжением σ = 0.5 МПа:")
    monitor.set_internal_stress(0.5e6)
    results_stressed = monitor.analyze_all_modes()
    for key, r in results_stressed.items():
        print(f"  {key:<13} | {r['status']:<16} | "
              f"f = {r['f_measured']:.3f} Гц | "
              f"σ = {r['sigma_MPa']:.2f} МПа")

    print("=" * 64)

    # 5. Проверка: при напряжении статусы должны измениться
    free_count = sum(1 for r in results.values() if r['status'] == 'FREE_OR_DAMPED')
    stressed_count = sum(1 for r in results_stressed.values() if r['status'] == 'CRITICAL_STRESS')
    print(f"\n  Свободных мод: {free_count} {'✅' if free_count > 0 else '❌'}")
    print(f"  Напряжённых мод: {stressed_count} {'✅' if stressed_count > 0 else '❌'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
