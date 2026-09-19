#!/usr/bin/env python3
"""
GAK-WaveCAD — EM Resonance Monitor Test
Тест электромагнитных мод сферического резонатора.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.em_resonance_monitor import EMResonanceMonitor


def main():
    logger = get_logger("em_resonance_test")

    print("=" * 64)
    print("  GAK-WaveCAD — EM Resonance Monitor Test")
    print("=" * 64)

    # Тест 1: без напряжения
    monitor = EMResonanceMonitor()
    monitor.init()
    monitor.run()
    results = monitor.get_results()

    monitor.print_report()

    print("\n  Проверки:")
    assert len(results) == 4, "Должно быть 4 моды"
    assert all(r['f_base'] > 0 for r in results.values()), "Частоты должны быть положительными"
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

    # Тест 3: TM-моды ниже TE-мод (для тех же l, n)
    f_tm = results2['TM_l1_n1']['f_shifted']
    f_te = results2['TE_l1_n1']['f_shifted']
    assert f_tm < f_te, "TM-мода должна быть ниже TE-моды"
    logger.info("Тест 3 пройден: f_TM < f_TE")
    print("    ✅ Тест 3: TM ниже TE — OK")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
