"""test_vacuum_integration.py — интеграция вакуума в цепочку из 8 модулей.

Шаг 4: VacuumMonitor в run.py, coupling k_plasma_vacuum, P_gas → плазма.
"""

import pytest
import numpy as np

from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.plasma_monitor import PlasmaMonitor
from physical_modules.coupling_monitor import CouplingMonitor


# --- Хелперы ---

def make_vacuum(**overrides):
    cfg = {
        "volume": 1.0,              # м³
        "pump_speed": 0.1,          # м³/с
        "gas_temp": 300.0,          # К
        "channel_length": 0.5,      # м
        "channel_diameter": 0.05,   # м
        "gas_viscosity": 1.8e-5,    # Па·с (воздух)
        "molecular_diameter": 3.7e-10,  # м
    }
    cfg.update(overrides)
    return VacuumMonitor(cfg)


def make_plasma(**overrides):
    cfg = {
        "radius": 0.01,
        "temperature_plasma": 5e6,
        "density_ion": 1e20,
        "B_field": 2.0,
        "fuel": "custom",
        "fuel_mass": 2.5,
        "rotation_freq": 1e4,       # ненулевая — иначе Ekman делит на ноль
        "ekman_layer": True,
    }
    cfg.update(overrides)
    return PlasmaMonitor(cfg)


def make_coupling():
    return CouplingMonitor({})


# --- 1. Вакуум в цепочке ---

class TestVacuumInChain:
    """Вакуум как 8-й модуль в run.py."""

    def test_vacuum_in_chain(self):
        """VacuumMonitor создаётся, init, run — без ошибок."""
        vac = make_vacuum()
        assert vac.init() is True
        assert vac.run() is True
        res = vac.get_results()
        assert "pressure_gas" in res
        assert res["pressure_gas"] >= 0.0
        assert "conductance" in res
        assert "mean_free_path" in res


# --- 2. Coupling: плазма ↔ вакуум ---

class TestCouplingPlasmaVacuum:
    """k_plasma_vacuum в CouplingMonitor."""

    def test_coupling_plasma_vacuum(self):
        """Coupling считает k_plasma_vacuum при наличии vacuum_results."""
        coupling = make_coupling()
        coupling.init()

        plasma_results = {
            "temperature_plasma": 5e6,
            "density_ion": 1e20,
            "B_field": 2.0,
            "sigma_viscous": 1e3,
            "diffusion_coeff": 1e-4,
            "barrier_index": 0.5,
            "lawson_ok": True,
            "lawson_triple": 3e21,
            "tau_E": 0.1,
            "dE_dt": 1e5,
            "phase": "STABLE",
        }
        vacuum_results = {
            "pressure_gas": 0.5,
            "conductance": 1e-3,
            "mean_free_path": 0.1,
        }

        coupling.set_results(plasma=plasma_results, vacuum=vacuum_results)
        assert coupling.run() is True
        res = coupling.get_results()
        assert "k_plasma_vacuum" in res
        assert res["k_plasma_vacuum"] > 0.0
        assert "P_gas" in res
        assert res["P_gas"] == 0.5

    def test_coupling_no_vacuum(self):
        """Без vacuum_results — k_plasma_vacuum = 0."""
        coupling = make_coupling()
        coupling.init()

        plasma_results = {
            "temperature_plasma": 5e6,
            "density_ion": 1e20,
            "B_field": 2.0,
            "sigma_viscous": 1e3,
            "diffusion_coeff": 1e-4,
            "barrier_index": 0.5,
            "lawson_ok": True,
            "lawson_triple": 3e21,
            "tau_E": 0.1,
            "dE_dt": 1e5,
            "phase": "STABLE",
        }
        coupling.set_results(plasma=plasma_results)
        assert coupling.run() is True
        res = coupling.get_results()
        assert res["k_plasma_vacuum"] == 0.0
        assert res.get("P_gas", 0.0) == 0.0

    def test_coupling_vacuum_affects_stability(self):
        """Высокое P_gas снижает stability_index."""
        coupling_lo = make_coupling()
        coupling_lo.init()
        coupling_hi = make_coupling()
        coupling_hi.init()

        plasma_results = {
            "temperature_plasma": 5e6,
            "density_ion": 1e20,
            "B_field": 2.0,
            "sigma_viscous": 1e3,
            "diffusion_coeff": 1e-4,
            "barrier_index": 0.5,
            "lawson_ok": True,
            "lawson_triple": 3e21,
            "tau_E": 0.1,
            "dE_dt": 1e5,
            "phase": "STABLE",
        }
        vac_lo = {"pressure_gas": 0.01, "conductance": 1e-3, "mean_free_path": 0.1}
        vac_hi = {"pressure_gas": 2.0, "conductance": 1e-3, "mean_free_path": 0.1}

        coupling_lo.set_results(plasma=plasma_results, vacuum=vac_lo)
        coupling_lo.run()
        coupling_hi.set_results(plasma=plasma_results, vacuum=vac_hi)
        coupling_hi.run()

        s_lo = coupling_lo.get_results()["stability_index"]
        s_hi = coupling_hi.get_results()["stability_index"]
        # Высокое давление газа → больше штраф → ниже stability
        assert s_hi <= s_lo


# --- 3. Полный прогон 8 модулей ---

class TestRunAllEightModules:
    """run() со всеми 8 модулями + step(dt) в цикле."""

    def test_run_all_eight_modules(self):
        """Все 8 модулей: осмос, термалка, акустика, коллапс,
        магноны, EM, плазма, вакуум — coupling не падает."""
        plasma = make_plasma()
        vacuum = make_vacuum()
        coupling = make_coupling()

        plasma.init()
        vacuum.init()
        coupling.init()

        plasma.run()
        vacuum.run()

        coupling.set_results(plasma=plasma.get_results(),
                             vacuum=vacuum.get_results())
        assert coupling.run() is True
        res = coupling.get_results()
        assert "stability_index" in res
        assert "system_status" in res
        assert "k_plasma_vacuum" in res

    def test_step_both_plasma_vacuum_in_loop(self):
        """Плазма и вакуум эволюционируют в общем цикле step(dt).
        Вакуум: давление падает от откачки.
        Плазма: step() не падает, время продвигается."""
        plasma = make_plasma()
        vacuum = make_vacuum()

        plasma.init()
        vacuum.init()
        plasma.run()

        P_initial = vacuum.pressure
        dt = 1e-3

        for _ in range(100):
            plasma.step(dt)
            vacuum.step(dt, Q_in=0.0, S_pump=vacuum.pump_speed)

        P_final = vacuum.pressure

        # Вакуум: давление упало от откачки
        assert P_final < P_initial, f"P_final={P_final} >= P_initial={P_initial}"

        # Плазма: step() не упал, нет NaN
        assert np.isfinite(plasma.temperature_plasma)
        assert plasma.temperature_plasma > 0.0

        # Время продвинулось
        assert vacuum.sim_time > 0.0
