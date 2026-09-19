"""
GAK-WaveCAD — главная точка запуска.

Загружает конфигурацию, инициализирует модули,
запускает цепочку: осмос → термалка → акустика.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.config_loader import ConfigLoader
from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor


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
    logger.info(f"Суммарное напряжение: osm={osm_stress:.1f} Па + th={th_stress:.1f} Па = {total_stress:.1f} Па")

    # Модуль 3: акустика
    ac_cfg = config.get("acoustic_monitor", {})
    acoustic = AcousticMonitor(ac_cfg)
    acoustic.set_internal_stress(total_stress)
    acoustic.init()
    acoustic.run()

    results = acoustic.get_results()
    print("\n  Акустический анализ:")
    for key, r in results.items():
        print(f"    {key:12s} : f_ref={r['f_reference']:.1f} Гц, "
              f"f_meas={r['f_measured']:.1f} Гц, "
              f"dF={r['delta_f']:+.1f} Гц, {r['status']}")

    logger.info("Цепочка выполнена: осмос → термалка → акустика")


if __name__ == "__main__":
    main()
