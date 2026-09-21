"""
Тесты для шага 4: вакуум в цепочке, coupling плазма-вакуум, все 8 модулей.
"""

import pytest
import numpy as np

from physical_modules.vacuum_monitor import VacuumMonitor
from physical_modules.coupling_monitor import CouplingMonitor
from physical_modules.plasma_monitor import PlasmaMonitor


class TestVacuumInChain:
    """Вакуум как 8-й модуль в цепочке."""

    def test_vacuum_in_chain(self):
        """VacuumMonitor работает в цепочке после плазмы."""
        plasma = PlasmaMonitor({})
        plasma.init()
        plasma.run()

        vacuum = VacuumMonitor({
            "volume": 1.0,
            "pump_speed": 0.1,
            "gas_temp": 300.0,
            "channel_length": 0.5,
            "channel_diameter": 0.01,
            "gas_viscosity": 1.8e-5,
            "molecular_diameter": 3.7e-10,
        })
        vacuum.init()
        vacuum.run(external={})

        # Шаги эволюции
        for _ in range(50):
            plasma.step(0.001)
            vacuum.step(0.001, Q_in=0.0, S_pump=0.1)

        pl_results = plasma.get_results()
        vac_results = vacuum.get_results()

        assert "pressure" in vac_results
        assert vac_results["pressure"] >= 0
        assert "temperature_plasma" in pl_results
        assert not np.isnan(pl_results["temperature_plasma"])


class TestCouplingPlasmaVacuum:
    """Coupling Monitor считает k_plasma_vacuum."""

    def test_coupling_plasma_vacuum(self):
        """Coupling считает k_plasma_vacuum при наличии плазмы и вакуума."""
        plasma = PlasmaMonitor({})
        plasma.init()
        plasma.run()

        vacuum = VacuumMonitor({
            "volume": 1.0,
            "pump_speed": 0.1,
            "gas_temp": 300.0,
            "channel_length": 0.5,
            "channel_diameter": 0.01,
            "gas_viscosity": 1.8e-5,
            "molecular_diameter": 3.7e-10,
        })
        vacuum.init()
        vacuum.run(external={})

        coupling = CouplingMonitor()
        coupling.init()
        coupling.set_results(
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic={"mode1": {"f_reference": 1e6, "f_measured": 1e6, "delta_f": 0, "status": "OK"}},
            collapse={"phase": "STABLE", "ratio": 0.0},
            magnon={"mode1": {"f": 1e9, "df_stress": 0}, "saturation_field": 1.0},
            em={"mode1": {"f_shifted": 1e9, "df_stress": 0}},
            plasma=plasma.get_results(),
            vacuum=vacuum.get_results(),
        )
        coupling.run()
        results = coupling.get_results()

        assert "k_plasma_vacuum" in results
        assert "P_gas" in results
        assert results["k_plasma_vacuum"] >= 0.0
        assert results["P_gas"] >= 0.0
        assert results["stability_index"] > 0.0

    def test_coupling_no_vacuum(self):
        """Coupling без вакуума: k_plasma_vacuum = 0."""
        plasma = PlasmaMonitor({})
        plasma.init()
        plasma.run()

        coupling = CouplingMonitor()
        coupling.init()
        coupling.set_results(
            plasma=plasma.get_results(),
        )
        coupling.run()
        results = coupling.get_results()

        assert results["k_plasma_vacuum"] == 0.0
        assert results["P_gas"] == 0.0

    def test_coupling_vacuum_affects_stability(self):
        """Высокое давление газа снижает stability."""
        plasma = PlasmaMonitor({})
        plasma.init()
        plasma.run()

        # Низкое давление
        vacuum_low = VacuumMonitor({
            "volume": 10.0,
            "pump_speed": 1.0,
            "gas_temp": 300.0,
            "channel_length": 0.5,
            "channel_diameter": 0.01,
            "gas_viscosity": 1.8e-5,
            "molecular_diameter": 3.7e-10,
        })
        vacuum_low.init()
        # Установим низкое давление
        vacuum_low.pressure = 0.01
        vacuum_low.run(external={})

        # Высокое давление
        vacuum_high = VacuumMonitor({
            "volume": 0.001,
            "pump_speed": 0.001,
            "gas_temp": 300.0,
            "channel_length": 0.5,
            "channel_diameter": 0.01,
            "gas_viscosity": 1.8e-5,
            "molecular_diameter": 3.7e-10,
        })
        vacuum_high.init()
        vacuum_high.pressure = 10.0
        vacuum_high.run(external={})

        coupling_low = CouplingMonitor()
        coupling_low.init()
        coupling_low.set_results(
            plasma=plasma.get_results(),
            vacuum=vacuum_low.get_results(),
        )
        coupling_low.run()

        coupling_high = CouplingMonitor()
        coupling_high.init()
        coupling_high.set_results(
            plasma=plasma.get_results(),
            vacuum=vacuum_high.get_results(),
        )
        coupling_high.run()

        s_low = coupling_low.get_results()["stability_index"]
        s_high = coupling_high.get_results()["stability_index"]

        # Высокое давление → больше штраф → ниже stability
        assert s_high <= s_low


class TestRunAllEightModules:
    """Полный прогон: все 8 модулей через coupling."""

    def test_run_all_eight_modules(self):
        """Coupling принимает результаты от всех 8 модулей."""
        plasma = PlasmaMonitor({})
        plasma.init()
        plasma.run()

        vacuum = VacuumMonitor({
            "volume": 1.0,
            "pump_speed": 0.1,
            "gas_temp": 300.0,
            "channel_length": 0.5,
            "channel_diameter": 0.01,
            "gas_viscosity": 1.8e-5,
            "molecular_diameter": 3.7e-10,
        })
        vacuum.init()
        vacuum.run(external={})

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
        results = coupling.get_results()

        # Все ключи на месте
        expected_keys = [
            'k_magnetoelastic', 'k_piezoelectric', 'k_magnetoelectric',
            'k_thermal_osmosis', 'geom_factor', 'collapse_phase',
            'k_plasma_acoustic', 'k_plasma_magnon', 'k_plasma_em',
            'k_plasma_thermal', 'k_plasma_collapse',
            'k_plasma_vacuum', 'P_gas',
            'stability_index', 'system_status',
            'lawson_ok', 'lawson_triple', 'tau_E', 'dE_dt',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

        # Нет NaN
        assert not np.isnan(results['stability_index'])
        assert not np.isnan(results['k_plasma_vacuum'])
        assert results['system_status'] in ('HEALTHY', 'DEGRADED', 'CRITICAL')

    def test_step_both_plasma_vacuum_in_loop(self):
        """Плазма и вакуум эволюционируют в общем цикле step(dt)."""
        plasma = PlasmaMonitor({})
        plasma.init()
        plasma.run()

        vacuum = VacuumMonitor({
            "volume": 1.0,
            "pump_speed": 0.1,
            "gas_temp": 300.0,
            "channel_length": 0.5,
            "channel_diameter": 0.01,
            "gas_viscosity": 1.8e-5,
            "molecular_diameter": 3.7e-10,
        })
        vacuum.init()
        vacuum.run(external={})

        T_initial = plasma.get_results()["temperature_plasma"]
        P_initial = vacuum.get_results()["pressure"]

        for _ in range(100):
            plasma.step(0.001)
            vacuum.step(0.001, Q_in=0.0, S_pump=0.1)

        T_final = plasma.get_results()["temperature_plasma"]
        P_final = vacuum.get_results()["pressure"]

        # Что-то изменилось (или осталось, но не NaN)
        assert not np.isnan(T_final)
        assert not np.isnan(P_final)
        assert P_final >= 0
