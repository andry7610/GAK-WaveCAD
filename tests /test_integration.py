#!/usr/bin/env python3
"""
GAK-WaveCAD — Integration Test
Связка: осмос → термалка → акустика.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    logger = get_logger("integration_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Integration Test (осмос → термалка → акустика)")
    print("=" * 64)

    # Осмос
    osmosis = OsmosisMonitor()
    osmosis.init()
    osmosis.run()
    stress_osm = osmosis.get_stress_Pa()

    logger.info(f"Осмос: stress={stress_osm:.1f} Па")
    osmosis.print_report()

    # Термалка
    thermal = ThermalMonitor()
    thermal.init()
    thermal.run()
    stress_th = thermal.get_thermal_stress_Pa()

    thermal.print_report()

    # Суммарное напряжение
    total_stress = stress_osm + stress_th
    logger.info(f"Суммарное напряжение: {total_stress:.1f} Па")

    # Акустика
    acoustic = AcousticMonitor()
    acoustic.set_internal_stress(total_stress)
    acoustic.init()
    acoustic.run()
    results = acoustic.get_results()

    print("\n  Акустический анализ:")
    for key, r in results.items():
        print(f"    {key:12s} : f_ref={r['f_reference']:.1f} Гц, "
              f"f_meas={r['f_measured']:.1f} Гц, "
              f"dF={r['delta_f']:+.1f} Гц, {r['status']}")

    print("\n  Проверки:")
    assert len(results) == 4, "Должно быть 4 моды"
    assert all(r['f_measured'] > 0 for r in results.values()), "Частоты должны быть положительными"
    assert stress_th > 0, "Тепловое напряжение должно быть положительным"
    logger.info("Интеграционный тест пройден: осмос → термалка → акустика")
    print("    ✅ Все проверки пройдены")

    print("=" * 64)


if __name__ == "__main__":
    main()
