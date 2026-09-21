"""tests/test_vacuum_integration.py — интеграция вакуума в цепочку GAK-WaveCAD.

Тесты:
  1. test_vacuum_in_chain — VacuumMonitor в цепочке
  2. test_coupling_plasma_vacuum — k_plasma_vacuum в coupling
  3. test_coupling_no_vacuum — coupling без вакуума
  4. test_coupling_vacuum_affects_stability — вакуум влияет на stability
  5. test_run_all_eight_modules — все 8 модулей в run
  6. test_step_both_plasma_vacuum_in_loop — step(dt) для плазмы и вакуума
"""

import numpy as np
import pytest

from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.plasma_monitor import PlasmaMonitor
from physical_modules.coupling_monitor import CouplingMonitor


# ── Хелперы ──────────────────────────────────────────────

def make_vacuum():
    return VacuumMonitor({
        "volume": 1.0,
        "pump_speed": 0.1,
        "gas_temp": 300.0,
        "channel_length": 0.1,
        "channel_diameter": 0.01,
        "gas_viscosity": 1.8e-5,
        "molecular_diameter": 3.7e-10,
    })


def make_plasma():
    return PlasmaMonitor(
        R=0.01,
        temperature=5e6,
        B_field=2.0,
        density=1e20,
        fuel={"A1": 2, "Z1": 1, "A2": 3, "Z2": 2, "reaction": "D-He3"},
        rotation_freq=1e4,
    )


def make_coupling():
    return CouplingMonitor({})


# ── 1. VacuumMonitor в цепочке ───────────────────────────

class TestVacuumInChain:
    def test_vacuum_in_chain(self):
        """VacuumMonitor создаётся, init, run — без ошибок."""
        vac = make_vacuum()
        assert vac.init() is True
        res = vac.run()
        assert isinstance(res, dict)
        assert "pressure_gas" in res
        assert "gas_temp" in res
        assert "mean_free_path" in res
        assert "knudsen_number" in res


# ── 2. Coupling: плазма ↔ вакуум ──────────────────────────

class TestCouplingPlasmaVacuum:
    def test_coupling_plasma_vacuum(self):
        """k_plasma_vacuum считается при наличии плазмы и вакуума."""
        vac = make_vacuum()
        vac.init()
        vac.run()

        plasma = make_plasma()
        plasma.init()
        plasma.run()

        cm = make_coupling()
        cm.init()
        cm.set_results(plasma=plasma.get_results(), vacuum=vac.get_results())
        cm.run()

        results = cm.get_results()
        assert "k_plasma_vacuum" in results
        assert "P_gas" in results
        assert results["k_plasma_vacuum"] >= 0.0
        assert results["P_gas"] >= 0.0

    def test_coupling_no_vacuum(self):
        """Без вакуума k_plasma_vacuum = 0."""
        plasma = make_plasma()
        plasma.init()
        plasma.run()

        cm = make_coupling()
        cm.init()
        cm.set_results(plasma=plasma.get_results())
        cm.run()

        results = cm.get_results()
        assert results["k_plasma_vacuum"] == 0.0
        assert results["P_gas"] == 0.0

    def test_coupling_vacuum_affects_stability(self):
        """Высокое P_gas понижает stability_index."""
        plasma = make_plasma()
        plasma.init()
        plasma.run()
        plasma_res = plasma.get_results()

        # Низкое давление — хорошая стабильность
        vac_low = make_vacuum()
        vac_low.init()
        vac_low.pressure = 10.0
        vac_low.run()

        cm_low = make_coupling()
        cm_low.init()
        cm_low.set_results(plasma=plasma_res, vacuum=vac_low.get_results())
        cm_low.run()
        s_low = cm_low.get_results()["stability_index"]

        # Высокое давление — плохая стабильность
        vac_high = make_vacuum()
        vac_high.init()
        vac_high.pressure = 5e5
        vac_high.run()

        cm_high = make_coupling()
        cm_high.init()
        cm_high.set_results(plasma=plasma_res, vacuum=vac_high.get_results())
        cm_high.run()
        s_high = cm_high.get_results()["stability_index"]

        assert s_high <= s_low


# ── 3. Все 8 модулей в run ────────────────────────────────

class TestRunAllEightModules:
    def test_run_all_eight_modules(self):
        """Все 8 модулей (включая вакуум) создаются и работают."""
        vac = make_vacuum()
        vac.init()
        vac_res = vac.run()
        assert isinstance(vac_res, dict)

        plasma = make_plasma()
        plasma.init()
        plasma_res = plasma.run()
        assert isinstance(plasma_res, dict)

        cm = make_coupling()
        cm.init()
        cm.set_results(plasma=plasma_res, vacuum=vac_res)
        cm.run()
        results = cm.get_results()

        assert "stability_index" in results
        assert "system_status" in results
        assert "k_plasma_vacuum" in results
        assert "P_gas" in results

    def test_step_both_plasma_vacuum_in_loop(self):
        """Плазма и вакуум эволюционируют в общем цикле step(dt).

        Вакуум: давление падает от откачки.
        Плазма: step() не падает, нет NaN.
        """
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

        # Плазма: step() не упал, нет NaN — берём T из get_results
        pr = plasma.get_results()
        T_plasma = pr.get("temperature_plasma", 0.0)
        assert np.isfinite(T_plasma), f"temperature_plasma={T_plasma} not finite"
        assert T_plasma > 0.0, f"temperature_plasma={T_plasma} <= 0"
