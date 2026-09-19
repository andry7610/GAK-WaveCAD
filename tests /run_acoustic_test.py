#!/usr/bin/env python3
"""
WaveCAD — Acoustic Monitor Integration Test
Точка входа: импортирует AcousticMonitor и прогоняет полный тест
через интерфейс BaseModule (init → run → get_results).

Запуск:
    python tests/run_acoustic_test.py
"""

import os
import sys

# Добавляем корень проекта в путь импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    print("=" * 64)
    print("  WaveCAD — Acoustic Monitor Integration Test")
    print("  v0.5 — 4-mode spherical shell analyzer (BaseModule)")
    print("=" * 64)

    # 1. Создаём монитор с конфигом по умолчанию
    monitor = AcousticMonitor()

    # 2. Инициализация через интерфейс BaseModule
    if not monitor.init():
        print("  ❌ Инициализация провалилась")
        return
    print("  ✅ Модуль инициализирован")

    # Опорные частоты
    print("\n  Опорные частоты мод:")
    for (l, mtype) in monitor.modes:
        name = f"{mtype}_l{l}"
        f = monitor.f0[(l, mtype)]
        print(f"    {name:<13} : {f:.1f} Гц")

    # 3. Прогон через run()
    if not monitor.run():
        print("  ❌ Выполнение провалилось")
        return
    print("  ✅ Анализ завершён")

    # 4. Результаты через get_results()
    results = monitor.get_results()
    print("\n  Результаты анализа (свободная оболочка):")
    print("  " + "-" * 60)
    for key, r in results.items():
        print(f"  {key:<13} | {r['status']:<16} | "
              f"f = {r['f_measured']:.3f} Гц | "
              f"Δf = {r['delta_f']:.3f} Гц | "
              f"σ = {r['sigma_MPa']:.2f} МПа | "
              f"{r['n_peaks']} пик(ов)")
    print("  " + "-" * 60)

    # 5. Тест с напряжением
    print("\n  Тест с внутренним напряжением σ = 0.5 МПа:")
    monitor.set_internal_stress(0.5e6)
    monitor.run()
    results_stressed = monitor.get_results()
    for key, r in results_stressed.items():
        print(f"  {key:<13} | {r['status']:<16} | "
              f"f = {r['f_measured']:.3f} Гц | "
              f"σ = {r['sigma_MPa']:.2f} МПа")

    # 6. Проверка статусов
    print("\n  " + "=" * 62)
    free_count = sum(1 for r in results.values() if r['status'] == 'FREE_OR_DAMPED')
    stressed_count = sum(1 for r in results_stressed.values() if r['status'] == 'CRITICAL_STRESS')
    print(f"  Свободных мод: {free_count} {'✅' if free_count > 0 else '❌'}")
    print(f"  Напряжённых мод: {stressed_count} {'✅' if stressed_count > 0 else '❌'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
