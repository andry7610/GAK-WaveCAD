"""
Тесты интеграции вакуума в цепочку GAK-WaveCAD.

Проверки:
  1. VacuumMonitor работает после плазмы в цепочке
  2. Coupling считает k_plasma_vacuum
  3. Coupling без вакуума отдаёт нули
  4. Вакуум влияет на stability_index
  5. Полная цепочка из 8 модулей
  6. Плазма и вакуум эволюционируют в общем цикле step(dt)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.plasma_monitor import PlasmaMonitor
from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.coupling_monitor import CouplingMonitor


VACUUM_CFG = {
    "volume": 1.0,
    "pump_speed": 0.1,
    "gas_temp": 300.0,
    "channel_length": 0.5,
    "channel_diameter": 0.01,
    "gas_viscosity": 1.8e-5,
    "molecular_diameter": 3.7e-10,
}


def make_vacuum():
    v = VacuumMonitor(VACUUM_CFG)
    v.init()
    v.run(external={})
    return v


def make_plasma():
    p = PlasmaMonitor({})
    p.init()
    p.run()
    return p


class TestVacuumInChain:
    """Вакуум работает в цепочке после плазмы."""

    def test_vacuum_in_chain(self):
        """VacuumMonitor работает в цепочке после плазмы."""
        plasma = make_plasma()
        vacuum = make_vacuum()

        # Шаги эволюции
        for _ in range(50):
            plasma.step(0.001)
            vacuum.step(0.001, Q_in=0.0, S_pump=0.1)

        pl_results = plasma.get_results()
        vac_results = vacuum.get_results()

        assert "pressure_gas" in vac_results
        assert "temperature_plasma" in pl_results
        assert vac_results["pressure_gas"] >= 0
        assert not np.isnan(pl_results["temperature_plasma"])


class TestCouplingPlasmaVacuum:
    """Coupling считает кросс-связи плазма ↔ вакуум."""

    def test_coupling_plasma_vacuum(self):
        """Coupling считает k_plasma_vacuum при наличии плазмы и вакуума."""
        plasma = make_plasma()
        vacuum = make_vacuum()

        coupling = CouplingMonitor()
        coupling.init()
        coupling.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"f_reference": 1e6, "f_measured": 1e6,
                                 "delta_f": 0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"mode1": {"f": 1e9, "df_stress": 0},
                    "saturation_field": 1.0},
            em={"mode1": {"f_shifted": 1e9, "df_stress": 0}},
            plasma=plasma.get_results(),
            vacuum=vacuum.get_results(),
        )
        assert coupling.run() is True

        res = coupling.get_results()
        assert "k_plasma_vacuum" in res
        assert "P_gas" in res
        assert "k_vacuum_thermal" in res
        assert res["P_gas"] >= 0

    def test_coupling_no_vacuum(self):
        """Coupling без вакуума отдаёт нули для вакуумных связей."""
        plasma = make_plasma()

        coupling = CouplingMonitor()
        coupling.init()
        coupling.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"f_reference": 1e6, "f_measured": 1e6,
                                 "delta_f": 0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"mode1": {"f": 1e9, "df_stress": 0},
                    "saturation_field": 1.0},
            em={"mode1": {"f_shifted": 1e9, "df_stress": 0}},
            plasma=plasma.get_results(),
        )
        assert coupling.run() is True

        res = coupling.get_results()
        assert res["k_plasma_vacuum"] == 0.0
        assert res["P_gas"] == 0.0
        assert res["k_vacuum_thermal"] == 0.0

    def test_coupling_vacuum_affects_stability(self):
        """Вакуум влияет на stability_index."""
        plasma = make_plasma()

        # Без вакуума
        coupling_no_vac = CouplingMonitor()
        coupling_no_vac.init()
        coupling_no_vac.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"f_reference": 1e6, "f_measured": 1e6,
                                 "delta_f": 0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"mode1": {"f": 1e9, "df_stress": 0},
                    "saturation_field": 1.0},
            em={"mode1": {"f_shifted": 1e9, "df_stress": 0}},
            plasma=plasma.get_results(),
        )
        coupling_no_vac.run()
        s_no_vac = coupling_no_vac.get_results()["stability_index"]

        # С вакуумом
        vacuum = make_vacuum()
        coupling_vac = CouplingMonitor()
        coupling_vac.init()
        coupling_vac.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"f_reference": 1e6, "f_measured": 1e6,
                                 "delta_f": 0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"mode1": {"f": 1e9, "df_stress": 0},
                    "saturation_field": 1.0},
            em={"mode1": {"f_shifted": 1e9, "df_stress": 0}},
            plasma=plasma.get_results(),
            vacuum=vacuum.get_results(),
        )
        coupling_vac.run()
        s_vac = coupling_vac.get_results()["stability_index"]

        # Stability изменился
        assert s_vac != s_no_vac


class TestRunAllEightModules:
    """Полная цепочка из 8 модулей."""

    def test_run_all_eight_modules(self):
        """Coupling принимает результаты от всех 8 модулей."""
        plasma = make_plasma()
        vacuum = make_vacuum()

        coupling = CouplingMonitor()
        coupling.init()

        # Все 8 модулей
        coupling.set_results(
            osmosis={"sigma_Pa": 1e5, "pi_Pa": 1e6},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={
                "mode1": {"f_reference": 1e6, "f_measured": 1.001e6,
                          "delta_f": 1000, "status": "OK"},
                "mode2": {"f_reference": 2e6, "f_measured": 2.002e6,
                          "delta_f": 2000, "status": "OK"},
            },
            collapse={"phase": "STABLE", "ratio": 0.1},
            magnon={
                "mode1": {"f": 1e9, "df_stress": 1e6},
                "saturation_field": 0.5,
            },
            em={"mode1": {"f_shifted": 1.01e9, "df_stress": 5e5}},
            plasma=plasma.get_results(),
            vacuum=vacuum.get_results(),
        )

        assert coupling.run() is True

        res = coupling.get_results()
        assert "stability_index" in res
        assert "system_status" in res
        assert "k_plasma_vacuum" in res
        assert "k_vacuum_thermal" in res
        assert 0.0 <= res["stability_index"] <= 1.0

    def test_step_both_plasma_vacuum_in_loop(self):
        """Плазма и вакуум эволюционируют в общем цикле step(dt)."""
        plasma = make_plasma()
        vacuum = make_vacuum()

        T_initial = plasma.get_results()["temperature_plasma"]
        P_initial = vacuum.get_results()["pressure_gas"]

        for _ in range(100):
            plasma.step(0.001)
            vacuum.step(0.001, Q_in=0.0, S_pump=0.1)

        T_final = plasma.get_results()["temperature_plasma"]
        P_final = vacuum.get_results()["pressure_gas"]

        # Температура изменилась
        assert T_final != T_initial
        # Давление упало (откачка)
        assert P_final < P_initial
        # Нет NaN
        assert not np.isnan(T_final)
        assert not np.isnan(P_final)
