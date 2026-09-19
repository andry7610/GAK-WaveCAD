#!/usr/bin/env python3
"""
GAK-WaveCAD — Integration Test
Связка: осмос → термалка → акустика → коллапс → магноны → EM-резонанс.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor
from physical_modules.born_collapse_monitor import BornCollapseMonitor
from physical_modules.magnon_monitor import MagnonMonitor
from physical_modules.em_resonance_monitor import EMResonanceMonitor


def main():
    logger = get_logger("integration_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Integration Test")
    print("  осмос → термалка → акустика → коллапс → магноны → EM-резонанс")
    print("=" * 64)

    # Осмос
    osmosis = OsmosisMonitor()
    osmosis.init()
    osmosis.run()
    stress_osm = osmosis.get_stress_Pa()
    osm_pressure = osmosis.results['pi_Pa']
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
    ac_results = acoustic.get_results()

    print("\n  Акустический анализ:")
    for key, r in ac_results.items():
        print(f"    {key:12s} : f_ref={r['f_reference']:.1f} Гц, "
              f"f_meas={r['f_measured']:.1f} Гц, "
              f"dF={r['delta_f']:+.1f} Гц, {r['status']}")

    # Коллапс
    collapse = BornCollapseMonitor()
    collapse.set_external_pressure(osm_pressure)
    collapse.init()
    collapse.run()
    collapse.print_report()

    # Магноны
    magnon = MagnonMonitor()
    magnon.set_mechanical_stress(total_stress)
    magnon.init()
    magnon.run()
    mag_results = magnon.get_results()

    print("\n  Магнонный анализ:")
    for key, r in mag_results.items():
        print(f"    {key:14s} : f={r['f']:.3e} Гц, "
              f"df_stress={r['df_stress']:+.3e} Гц, Q={r['Q']:.0f}")

    # EM-резонанс
    em = EMResonanceMonitor()
    em.set_mechanical_stress(total_stress)
    em.init()
    em.run()
    em_results = em.get_results()

    print("\n  EM-резонанс:")
    for key, r in em_results.items():
        print(f"    {key:12s} : f={r['f_shifted']:.3e} Гц, "
              f"df_stress={r['df_stress']:+.3e} Гц, Q={r['Q']:.0f}")

    print("\n  Проверки:")
    assert len(ac_results) == 4, "Должно быть 4 акустические моды"
    assert len(mag_results) == 4, "Должно быть 4 магнонные моды"
    assert len(em_results) == 4, "Должно быть 4 EM-моды"
    assert all(r['f'] > 0 for r in mag_results.values()), "Магнонные частоты > 0"
    assert all(r['f_shifted'] > 0 for r in em_results.values()), "EM-частоты > 0"
    assert all(r['df_stress'] != 0 for r in em_results.values()), "EM-сдвиг от напряжения ненулевой"
    assert collapse.results['P_cr'] > 0, "P_cr > 0"
    logger.info("Интеграционный тест пройден: 6 модулей, все проверки OK")
    print("    ✅ Все проверки пройдены")

    print("=" * 64)


if __name__ == "__main__":
    main()

