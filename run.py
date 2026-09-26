"""
GAK-WaveCAD — главная точка запуска.

Загружает конфигурацию, инициализирует модули,
запускает цепочку:
  осмос -> термалка -> акустика -> коллапс -> магноны -> EM -> плазма -> вакуум -> квантовый вакуум -> био-мост -> кросс-связи.

v0.9 — фиксированное логирование: P_heat_density, P_cond, ratio (P_heat/P_cond) в одинаковых единицах Вт/м³
v0.8 — динамический E_field: T_plasma -> q_vacuum.step(T_plasma=...)
v0.7 — Швингер: pair_rate из квантового вакуума -> в плазму (T_plasma).
v0.6 — добавлен GAKQuantumVacuum (8b) после технического вакуума.
v0.5 — добавлен BioBridge (9-й модуль) после вакуума.
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
from physical_modules.plasma_monitor import PlasmaMonitor
from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.quantum_vacuum import GAKQuantumVacuum
from physical_modules.bio_bridge import BioBridge
from physical_modules.coupling_monitor import CouplingMonitor

import numpy as np

K_B = 1.380649e-23


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
        if isinstance(r, dict):
            print(f"    {key:12s} : f_ref={r.get('f_reference', 0):.1f} Гц, "
                  f"f_meas={r.get('f_measured', 0):.1f} Гц, "
                  f"dF={r.get('delta_f', 0):+.1f} Гц, {r.get('status', '?')}")

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

    # Модуль 8b: квантовый вакуум
    qvac_cfg = config.get("quantum_vacuum", {})
    q_vacuum = GAKQuantumVacuum(qvac_cfg)
    q_vacuum.run()

    # Модуль 9: био-мост (после вакуума)
    bio_cfg = config.get("bio_bridge", {})
    bio = BioBridge(bio_cfg)
    bio.init()
    bio.run()

    # Объём плазмы для пересчёта P_heat в плотность
    V_plasma = (4.0 / 3.0) * np.pi * plasma.radius ** 3
    pair_energy_gain = plasma.pair_energy_gain

    # --- Цикл эволюции: плазма + вакуум + квантовый вакуум + био ---
    for step_i in range(N_STEPS):
        # Плазма делает шаг
        step_result = plasma.step(dt)

        # Вакуум делает шаг
        pl_results = plasma.get_results()
        T_plasma = pl_results.get("temperature_plasma", 0.0)
        n_gas = pl_results.get("density", 0.0) * 1e-6
        vacuum.step(dt, Q_in=0.0, S_pump=vacuum.pump_speed)
        vacuum.run(external={})

        # Квантовый вакуум: T_plasma -> E_field -> pair_rate (v0.8)
        q_vacuum.step(dt, T_plasma=T_plasma)
        q_vacuum_results = q_vacuum.run()
        pair_rate = q_vacuum_results.get("pair_rate", 0.0)
        E_field_now = q_vacuum_results.get("E_field", 0.0)

        # Плазма получает pair_rate от квантового вакуума
        plasma.run(external_stress={"pair_rate": pair_rate})

        # Био-мост делает шаг: получает T_plasma и P_gas
        vac_results = vacuum.get_results()
        P_gas = vac_results.get("pressure_gas", 0.0)
        bio.step(dt, neuron_activity=None, T_plasma=T_plasma, P_gas=P_gas)
        bio.run()

        # --- Логирование каждые 500 шагов (v0.9) ---
        if step_i == 0 or step_i == N_STEPS - 1 or (step_i + 1) % 500 == 0:
            bio_results = bio.get_results()

            # P_cond из step_result (Вт/м³, плотность мощности)
            P_cond = step_result.get("P_cond", 0.0)

            # P_heat: pair_rate * pair_energy_gain = полная мощность (Вт)
            # Переводим в плотность: P_heat / V_plasma (Вт/м³)
            P_heat = pair_rate * pair_energy_gain
            P_heat_density = P_heat / V_plasma if V_plasma > 0 else 0.0

            # Ratio: P_heat_density / P_cond — должно стремиться к 1.0 при равновесии
            ratio = P_heat_density / P_cond if P_cond > 0 else float('inf')

            logger.info(f"  Шаг {step_i + 1}/{N_STEPS}: "
                        f"T_plasma={T_plasma:.2e} К, "
                        f"P_cond={P_cond:.3e} Вт/м³, "
                        f"P_heat_d={P_heat_density:.3e} Вт/м³, "
                        f"ratio={ratio:.3f}, "
                        f"pair_rate={pair_rate:.2e}, "
                        f"P_gas={P_gas:.2e} Па, "
                        f"bio_activity={bio_results.get('activity', 0):.2f} Гц")

    vacuum.print_report()
    bio.print_report()

    # Модуль 10: кросс-связи
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
        quantum=q_vacuum.get_results(),
    )
    coupling.run()
    coupling.print_report()

    logger.info("Цепочка выполнена: осмос -> термалка -> акустика -> коллапс -> "
                "магноны -> EM -> плазма -> вакуум -> квантовый вакуум -> био-мост -> кросс-связи")


if __name__ == "__main__":
    main()
