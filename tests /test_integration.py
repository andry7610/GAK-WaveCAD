#!/usr/bin/env python3
"""
GAK-WaveCAD — Integration Test
Связка: осмос → акустика.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    logger = get_logger("integration_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Integration Test (осмос → акустика)")
    print("=" * 64)

    # Ось — осмос
    osmosis = OsmosisMonitor()
    osmosis.init()
    osmosis.run()
    stress = osmosis.get_stress_Pa()

    logger.info(f"Осмос: Pi={osmosis.results['pi_Pa']:.1f} Па, stress={stress:.1f} Па")
    osmosis.print_report()

    # Передача в акустику
    acoustic = AcousticMonitor()
    acoustic.set_internal_stress(stress)
    acoustic.init()
    acoustic.run()
    results = acoustic.get_results()

    print("\n  Акустический анализ:")
    for key, r in results.items():
        print(f"    {key:12s} : f_ref={r['f_reference']:.1f} Гц, "
              f"f_meas={r['f_measured']:.1f} Гц, "
              f"Δf={r['delta_f']:+.1f} Гц, {r['status']}")

    print("\n  Проверки:")
    assert len(results) == 4, "Должно быть 4 моды"
    assert all(r['f_measured'] > 0 for r in results.values()), "Частоты должны быть положительными"
    assert all(abs(r['delta_f']) < 1e6 for r in results.values()), "Сдвиг частоты разумный"
    logger.info("Интеграционный тест пройден: 4 моды, частоты положительные")
    print("    ✅ Все проверки пройдены")

    print("=" * 64)


if __name__ == "__main__":
    main()
