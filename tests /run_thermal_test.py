#!/usr/bin/env python3
"""
GAK-WaveCAD — Thermal Monitor Test
Тест тепловых напряжений в сферической оболочке.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.thermal_monitor import ThermalMonitor


def main():
    logger = get_logger("thermal_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Thermal Monitor Test")
    print("=" * 64)

    monitor = ThermalMonitor()
    monitor.init()
    monitor.run()
    results = monitor.get_results()

    monitor.print_report()

    print("\n  Проверки:")
    assert results['dT'] > 0, "Разность температур должна быть положительной"
    assert results['sigma_thermal_Pa'] > 0, "Тепловое напряжение должно быть положительным"
    assert results['Q'] > 0, "Тепловой поток должен быть положительным"
    logger.info("Проверки пройдены: dT>0, sigma>0, Q>0")
    print("    ✅ Все проверки пройдены")

    print("=" * 64)


if __name__ == "__main__":
    main()
