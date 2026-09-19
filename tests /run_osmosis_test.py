#!/usr/bin/env python3
"""
GAK-WaveCAD — Osmosis Monitor Test
Тест осмотического давления и напряжения в оболочке.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor


def main():
    logger = get_logger("osmosis_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Osmosis Monitor Test")
    print("=" * 64)

    monitor = OsmosisMonitor()
    monitor.init()
    monitor.run()
    results = monitor.get_results()

    monitor.print_report()

    print("\n  Проверки:")
    assert results['pi_Pa'] > 0, "Осмотическое давление должно быть положительным"
    assert results['sigma_Pa'] > 0, "Напряжение должно быть положительным"
    logger.info("Проверки пройдены: pi>0, sigma>0")
    print("    ✅ Все проверки пройдены")

    print("=" * 64)


if __name__ == "__main__":
    main()
