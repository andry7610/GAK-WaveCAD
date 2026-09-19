#!/usr/bin/env python3
"""
GAK-WaveCAD — Born Collapse Monitor Test
Тест устойчивости сферической оболочки под давлением.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.logger import get_logger
from physical_modules.born_collapse_monitor import BornCollapseMonitor


def main():
    logger = get_logger("born_collapse_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Born Collapse Monitor Test")
    print("=" * 64)

    # Тест 1: устойчивое состояние
    monitor = BornCollapseMonitor()
    monitor.init()
    monitor.run()
    results = monitor.get_results()

    monitor.print_report()

    print("\n  Проверки:")
    assert results['P_cr'] > 0, "Критическое давление должно быть положительным"
    assert results['phase'] == "STABLE", "Без давления оболочка должна быть устойчивой"
    logger.info("Тест 1 пройден: P_cr>0, phase=STABLE")
    print("    ✅ Тест 1: устойчивое состояние — OK")

    # Тест 2: коллапс
    P_cr = results['P_cr']
    monitor.set_external_pressure(P_cr * 1.2)
    monitor.run()
    results2 = monitor.get_results()
    assert results2['phase'] == "COLLAPSE", "При P > P_cr должен быть коллапс"
    logger.info("Тест 2 пройден: phase=COLLAPSE при P > P_cr")
    print("    ✅ Тест 2: коллапс при превышении — OK")

    # Тест 3: надувание
    monitor.set_external_pressure(-1000.0)
    monitor.run()
    results3 = monitor.get_results()
    assert results3['phase'] == "INFLATION", "При отрицательном давлении — надувание"
    logger.info("Тест 3 пройден: phase=INFLATION при P < 0")
    print("    ✅ Тест 3: надувание при отрицательном давлении — OK")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
