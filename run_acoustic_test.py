
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
    for mode, f in monitor.modes.items():
        print(f"    {mode:<13} : {f:.1f} Гц")

    # 2. Прогоняем интеграционный тест
    test = monitor.run_test()

    # 3. Вывод результатов
    print("\n  Результаты анализа:")
    print("  " + "-" * 60)
    for r in test["results"]:
        print(f"  {r['mode']:<13} | {r['status']:<16} | "
              f"Δf = {r['delta_f']:.3f} Гц | "
              f"σ = {r['stress_mpa']:.2f} МПа | "
              f"{r['peaks']} пик(ов)")
    print("  " + "-" * 60)

    print(f"\n  Погрешность квадруполя: {test['error_pct']:.2f}%  "
          f"{'✅' if test['error_pct'] < 5 else '❌'}")
    print(f"  Свободная энергия: U = {test['energy']:.2e}")
    print("=" * 64)


if __name__ == "__main__":
    main()
