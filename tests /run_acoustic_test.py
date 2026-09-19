#!/usr/bin/env python3
"""
GAK-WaveCAD — Acoustic Monitor Test
Тест акустических мод сферической оболочки.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    logger = get_logger("acoustic_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Acoustic Monitor Test")
    print("  v0.4 — 4-mode spherical shell analyzer")
    print("=" * 64)

    monitor = AcousticMonitor()
    monitor.init()
    monitor.run()
    results = monitor.get_results()

    print("\n  Опорные частоты мод:")
    for key, r in results.items():
        print(f"    {key:12s} : {r['f_reference']:.1f} Гц")

    print("\n  Измеренные частоты (FFT):")
    for key, r in results.items():
        print(f"    {key:12s} : {r['f_measured']:.1f} Гц "
              f"(Δf = {r['delta_f']:+.1f} Гц, {r['status']})")

    print("\n  Проверки:")
    assert len(results) == 4, "Должно быть 4 моды"
    assert all(r['f_measured'] > 0 for r in results.values()), "Частоты должны быть положительными"
    logger.info("Тест пройден: 4 моды, частоты положительные")
    print("    ✅ Все проверки пройдены")

    print("=" * 64)


if __name__ == "__main__":
    main()
