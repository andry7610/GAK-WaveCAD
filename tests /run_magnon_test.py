#!/usr/bin/env python3
"""
GAK-WaveCAD — Magnon Monitor Test
Тест спиновых мод в YIG-оболочке.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.magnon_monitor import MagnonMonitor


def main():
    logger = get_logger("magnon_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Magnon Monitor Test")
    print("=" * 64)

    # Тест 1: без напряжения
    monitor = MagnonMonitor()
    monitor.init()
    monitor.run()
    results = monitor.get_results()

    monitor.print_report()

    print("\n  Проверки:")
    assert len(results) == 4, "Должно быть 4 моды"
    assert all(r['f'] > 0 for r in results.values()), "Частоты должны быть положительными"
    assert all(r['df_stress'] == 0 for r in results.values()), "Без напряжения сдвига нет"
    logger.info("Тест 1 пройден: 4 моды, частоты > 0, сдвиг = 0")
    print("    ✅ Тест 1: моды без напряжения — OK")

    # Тест 2: с напряжением
    monitor.set_mechanical_stress(1000.0)
    monitor.run()
    results2 = monitor.get_results()

    print("\n  С напряжением σ=1000 Па:")
    monitor.print_report()

    assert all(r['df_stress'] != 0 for r in results2.values()), "Сдвиг должен быть ненулевым"
    logger.info("Тест 2 пройден: сдвиг частот от напряжения")
    print("    ✅ Тест 2: сдвиг от напряжения — OK")

    # Тест 3: Kittel ниже обменных
    f_kittel = results2['kittel_l0']['f']
    f_exch = results2['exchange_l1']['f']
    assert f_exch > f_kittel, "Обменная мода должна быть выше Kittel"
    logger.info("Тест 3 пройден: f_exchange > f_kittel")
    print("    ✅ Тест 3: обменные моды выше Kittel — OK")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
