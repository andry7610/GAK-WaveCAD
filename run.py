"""
GAK-WaveCAD — главная точка запуска.

Загружает конфигурацию, инициализирует модули,
запускает цепочку: осмос → акустика.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.config_loader import ConfigLoader
from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    logger = get_logger("run")

    # Загрузка конфигурации
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    config = ConfigLoader.load(config_path)
    logger.info("Конфигурация загружена")

    # Модуль 1: осмос
    osm_cfg = config.get("osmosis_monitor", {})
    osmosis = OsmosisMonitor(osm_cfg)
    osmosis.init()
    osmosis.run()
    osmosis.print_report()

    # Передача напряжения в акустику
    stress = osmosis.get_stress_Pa()
    logger.info(f"Передача напряжения в акустику: {stress:.1f} Па")

    # Модуль 2: акустика
    ac_cfg = config.get("acoustic_monitor", {})
    acoustic = AcousticMonitor(ac_cfg)
    acoustic.set_internal_stress(stress)
    acoustic.init()
    acoustic.run()

    results = acoustic.get_results()
    print("\n  Акустический анализ:")
    for key, r in results.items():
        print(f"    {key:12s} : f_ref={r['f_reference']:.1f} Гц, "
              f"f_meas={r['f_measured']:.1f} Гц, "
              f"Δf={r['delta_f']:+.1f} Гц, {r['status']}")

    logger.info("Цепочка выполнена: осмос → акустика")


if __name__ == "__main__":
    main()
