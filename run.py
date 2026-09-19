"""
GAK-WaveCAD — главная точка запуска.

Загружает конфигурацию, инициализирует модули,
запускает цепочку:
  осмос → термалка → акустика → коллапс → магноны → EM → кросс-связи.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.config_loader import ConfigLoader
from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor
from physical_modules.born_collapse_monitor import BornCollapseMonitor
from physical_modules.magnon_monitor import MagnonMonitor
from physical_modules.em_resonance_monitor import EMResonanceMonitor
from physical_modules.coupling_monitor import CouplingMonitor


def main():
    logger = get_logger("run")

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    config = ConfigLoader.load(config_path)
    logger.info("Конфигурация загружена")

    # Модуль 1: осмос
    osm_cfg = config.get("osmosis_monitor", {})
    osmosis = OsmosisMonitor(osm_cfg)
    osmosis.init()
    osmosis.run()
    osmosis.print_report()

    # Модуль 2: термалка
    th_cfg = config.get("thermal_monitor", {})
    thermal = ThermalMonitor(th_cfg)
    thermal.init()
    thermal.run()
    thermal.print_report()

    # Суммарное напряжение
    osm_stress = osmosis.get_stress_Pa()
    th_stress = thermal.get_thermal_stress_Pa()
    total_stress = osm_stress + th_stress
    logger.info(f"Суммарное напряжение: {total_stress:.1f} Па")

    # Модуль 3: акустика
    ac_cfg = config.get("acoustic_monitor", {})
    acoustic = AcousticMonitor(ac_cfg)
    acoustic.set_internal_stress(total_stress)
    acoustic.init()
    acoustic.run()

    ac_results = acoustic.get_results()
    print("\n  Акустический анализ:")
    for key, r in ac_results.items():
        print(f"    {key:12s} : f_ref={r['f_reference']:.1f} Гц, "
              f"f_meas={r['f_measured']:.1f} Гц, "
              f"dF={r['delta_f']:+.1f} Гц, {r['status']}")

    # Модуль 4: коллапс
    bc_cfg = config.get("born_collapse_monitor", {})
    collapse = BornCollapseMonitor(bc_cfg)
    osm_pressure = osmosis.results['pi_Pa']
    collapse.set_external_pressure(osm_pressure)
    collapse.init()
    collapse.run()
    collapse.print_report()

    # Модуль 5: магноны
    mag_cfg = config.get("magnon_monitor", {})
    magnon = MagnonMonitor(mag_cfg)
    magnon.set_mechanical_stress(total_stress)
    magnon.init()
    magnon.run()
    magnon.print_report()

    # Модуль 6: EM-резонанс
    em_cfg = config.get("em_resonance_monitor", {})
    em = EMResonanceMonitor(em_cfg)
    em.set_mechanical_stress(total_stress)
    em.init()
    em.run()
    em.print_report()

    # Модуль 7: кросс-связи
    coupling = CouplingMonitor()
    coupling.init()
    coupling.set_results(
        osmosis=osmosis.get_results(),
        thermal=thermal.get_results(),
        acoustic=acoustic.get_results(),
        collapse=collapse.get_results(),
        magnon=magnon.get_results(),
        em=em.get_results(),
    )
    coupling.run()
    coupling.print_report()

    logger.info("Цепочка выполнена: осмос → термалка → акустика → коллапс → "
                "магноны → EM → кросс-связи")


if __name__ == "__main__":
    main()

