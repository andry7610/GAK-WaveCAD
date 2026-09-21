"""
GAK-WaveCAD — главная точка запуска.

Загружает конфигурацию, инициализирует модули,
запускает цепочку:
  осмос → термалка → акустика → коллапс → магноны → EM → плазма → вакуум → био-мост → кросс-связи.

v0.5 — добавлен BioBridge (9-й модуль).
       Нейроны → акустический сдвиг → плазма.
       Плазма → температура → обратная связь на нейроны.
       Вакуум → P_gas → плазма.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from core.config_loader import ConfigLoader
from core.logger import get_logger
from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor
from physical_modules.born_collapse_monitor import BornCollapseMonitor
from physical_modules.magnon_monitor import MagnonMonitor
from physical_modules.em_resonance_monitor import EMResonanceMonitor
from physical_modules.plasma_monitor import PlasmaMonitor
from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.bio_bridge import BioBridge
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

    # --- Параметры симуляции ---
    sim_cfg = config.get("simulation", {})
    N_STEPS = sim_cfg.get("n_steps", 100)
    dt = sim_cfg.get("dt", 0.001)

    # Модуль 7: плазма
    pl_cfg = config.get("plasma_monitor", {})
    plasma = PlasmaMonitor(pl_cfg)
    plasma.init()
    plasma.run()

    # Модуль 8: вакуум (после плазмы)
    vac_cfg = config.get("vacuum_monitor", {})
    vacuum = VacuumMonitor(vac_cfg)
    vacuum.init()
    vacuum.run(external={})

    # Модуль 9: био-мост
    bio_cfg = config.get("bio_bridge", {})
    bio = BioBridge(bio_cfg)
    bio.init()

    # Начальная активность нейронов
    np_rng = np.random.default_rng(42)
    neuron_activity = np_rng.uniform(0, 0.8, size=bio.n_neurons)

    # --- Цикл эволюции: плазма + вакуум + био ---
    for step_i in range(N_STEPS):
        # Плазма делает шаг
        plasma.step(dt)
        plasma.run()

        # Вакуум делает шаг
        pl_results = plasma.get_results()
        T_plasma = pl_results.get("temperature_plasma", 0.0)
        vacuum.step(dt, Q_in=0.0, S_pump=vacuum.pump_speed)
        vacuum.run(external={})

        vac_results = vacuum.get_results()
        P_gas = vac_results.get("pressure_gas", 0.0)

        # Био-мост делает шаг: нейроны → акустика → плазма, плазма → обратная связь
        # Лёгкая стохастическая активность нейронов
        neuron_activity = np_rng.uniform(0, 1.0, size=bio.n_neurons)
        bio.step(dt, neuron_activity, T_plasma, P_gas)

        if step_i == 0 or step_i == N_STEPS - 1 or (step_i + 1) % 10 == 0:
            bio_results = bio.get_results()
            logger.info(f"  Шаг {step_i + 1}/{N_STEPS}: "
                        f"T_plasma={T_plasma:.2e} К, "
                        f"P_gas={P_gas:.2e} Па, "
                        f"spikes={bio_results.get('spike_count', 0)}, "
                        f"shift={bio_results.get('acoustic_freq_shift', 0):.2f} Гц")

    plasma.print_report()
    vacuum.print_report()
    bio.print_report()

    # Модуль 10: кросс-связи (девять модулей)
    coupling = CouplingMonitor()
    coupling.init()
    coupling.set_results(
        osmosis=osmosis.get_results(),
        thermal=thermal.get_results(),
        acoustic=acoustic.get_results(),
        collapse=collapse.get_results(),
        magnon=magnon.get_results(),
        em=em.get_results(),
        plasma=plasma.get_results(),
        vacuum=vacuum.get_results(),
        bio=bio.get_results(),
    )
    coupling.run()
    coupling.print_report()

    logger.info("Цепочка выполнена: осмос → термалка → акустика → коллапс → "
                "магноны → EM → плазма → вакуум → био-мост → кросс-связи")


if __name__ == "__main__":
    main()
