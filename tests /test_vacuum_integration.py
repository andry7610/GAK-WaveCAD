"""
Тесты интеграции вакуума в основную цепочку GAK-WaveCAD.

Шаг 4: вакуум → run.py → coupling.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.coupling_monitor import CouplingMonitor


# --- Хелперы ---

def make_vacuum():
    return VacuumMonitor({
        "volume": 1.0,
        "pump_speed": 0.5,
        "gas_temp": 300.0,
        "channel_length": 0.1,
        "channel_diameter": 0.01,
        "gas_viscosity": 1.8e-5,
    })


def make_plasma():
    from physical_modules.plasma_monitor import PlasmaMonitor
    return PlasmaMonitor({
        "radius": 0.01,
        "temperature": 5e6,
        "density": 1e20,
        "B_field": 2.0,
        "fuel": "custom",
        "fuel_mass": [2.0, 3.0],
        "fuel_charge": [1, 1],
        "fuel_fractions": [0.5, 0.5],
        "rotation_freq": 0.0,
        "viscosity": 0.0,
    })


# --- 1. Вакуум в цепочке ---

class TestVacuumInChain:
    def test_vacuum_in_chain(self):
        """VacuumMonitor инициализируется, прогоняется, отдаёт результаты."""
        v = make_vacuum()
        v.init()
        v.run()
        r = v.get_results()
        assert "pressure_gas" in r
        assert "conductance" in r
        assert "mean_free_path" in r
        assert r["pressure_gas"] >= 0


# --- 2. Coupling плазма ↔ вакуум ---

class TestCouplingPlasmaVacuum:
    def test_coupling_plasma_vacuum(self):
        """Coupling считает k_plasma_vacuum при наличии vacuum_results."""
        c = CouplingMonitor()
        c.init()

        plasma_results = {
            "temperature_plasma": 5e6,
            "density": 1e20,
            "B_field": 2.0,
            "sigma_viscous": 0.0,
            "diffusion_coeff": 0.1,
            "barrier_index": 0.5,
            "phase": "STABLE",
            "lawson_ok": True,
            "lawson_triple": 5e21,
            "tau_E": 0.1,
            "dE_dt": 1e3,
        }
        vacuum_results = {
            "pressure_gas": 0.5,
            "conductance": 0.01,
            "mean_free_path": 0.1,
        }

        c.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"delta_f": 1.0, "f_reference": 100.0,
                                "f_measured": 101.0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.1},
            magnon={"mode1": {"df_stress": 0.5, "f": 1e9},
                    "saturation_field": 1.0},
            em={"mode1": {"df_stress": 0.3, "f_shifted": 1e9}},
            plasma=plasma_results,
            vacuum=vacuum_results,
        )
        c.run()
        r = c.get_results()

        assert "k_plasma_vacuum" in r
        assert r["k_plasma_vacuum"] >= 0
        assert r["k_plasma_vacuum"] <= 1.0
        assert "P_gas" in r
        assert r["P_gas"] == 0.5

    def test_coupling_no_vacuum(self):
        """Coupling работает без vacuum_results (обратная совместимость)."""
        c = CouplingMonitor()
        c.init()

        c.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"saturation_field": 1.0},
            em={},
            plasma=None,
            vacuum=None,
        )
        c.run()
        r = c.get_results()

        assert "k_plasma_vacuum" in r
        assert r["k_plasma_vacuum"] == 0.0
        assert r["P_gas"] == 0.0

    def test_coupling_vacuum_affects_stability(self):
        """Высокое P_gas снижает стабильность."""
        c = CouplingMonitor()
        c.init()

        base = dict(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"saturation_field": 1.0},
            em={},
            plasma={
                "temperature_plasma": 5e6,
                "density": 1e20,
                "B_field": 2.0,
                "sigma_viscous": 0.0,
                "diffusion_coeff": 0.1,
                "barrier_index": 0.5,
                "phase": "STABLE",
                "lawson_ok": True,
                "lawson_triple": 5e21,
                "tau_E": 0.1,
                "dE_dt": 1e3,
            },
        )

        # Без вакуума
        c.set_results(vacuum=None, **base)
        c.run()
        s_no_vac = c.get_results()["stability_index"]

        # С высоким давлением газа
        c2 = CouplingMonitor()
        c2.init()
        c2.set_results(
            vacuum={"pressure_gas": 2.0, "conductance": 0.01,
                    "mean_free_path": 0.01},
            **base
        )
        c2.run()
        s_with_vac = c2.get_results()["stability_index"]

        assert s_with_vac <= s_no_vac


# --- 3. Полная цепочка 8 модулей ---

class TestRunAllEightModules:
    def test_run_all_eight_modules(self):
        """Все 8 модулей в цепочке: осмос → термалка → акустика → коллапс
        → магноны → EM → плазма → вакуум → coupling."""
        v = make_vacuum()
        v.init()
        v.run()
        vacuum_results = v.get_results()

        c = CouplingMonitor()
        c.init()
        c.set_results(
            osmosis={"sigma_Pa": 1e5, "pi_Pa": 1e6},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"delta_f": 1.0, "f_reference": 100.0,
                                "f_measured": 101.0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.1},
            magnon={"mode1": {"df_stress": 0.5, "f": 1e9},
                    "saturation_field": 1.0},
            em={"mode1": {"df_stress": 0.3, "f_shifted": 1e9}},
            plasma={
                "temperature_plasma": 5e6,
                "density": 1e20,
                "B_field": 2.0,
                "sigma_viscous": 0.0,
                "diffusion_coeff": 0.1,
                "barrier_index": 0.5,
                "phase": "STABLE",
                "lawson_ok": True,
                "lawson_triple": 5e21,
                "tau_E": 0.1,
                "dE_dt": 1e3,
            },
            vacuum=vacuum_results,
        )
        c.run()
        r = c.get_results()

        assert "stability_index" in r
        assert "k_plasma_vacuum" in r
        assert "P_gas" in r
        assert "system_status" in r

    def test_step_both_plasma_vacuum_in_loop(self):
        """Плазма и вакуум эволюционируют в общем цикле step(dt).
        Вакуум: давление падает от откачки.
        Плазма: step() не падает, время продвигается."""
        plasma = make_plasma()
        vacuum = make_vacuum()

        plasma.init()
        plasma.run()
        vacuum.init()
        vacuum.run()

        P_initial = vacuum.get_results()["pressure_gas"]

        for _ in range(100):
            plasma.step(0.001)
            vacuum.step(0.001, Q_in=0.0, S_pump=0.1)

        P_final = vacuum.get_results()["pressure_gas"]

        # Вакуум: давление упало от откачки
        assert P_final < P_initial, (
            f"Pressure should drop: {P_final} >= {P_initial}"
        )
        assert P_final >= 0

        # Плазма: step() не упал, нет NaN
        T = plasma.get_results()["temperature_plasma"]
        assert not np.isnan(T)
        assert T > 0
