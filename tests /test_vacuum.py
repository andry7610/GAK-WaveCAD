"""test_vacuum.py — Тесты технического вакуума (v0.1).

9 тестов:
  - давление из плотности
  - проводимость вязкая (Пуазейль)
  - проводимость молекулярная (Кнудсен)
  - поток пропорционален Delta P (молекулярный режим)
  - откачка камеры
  - нет отрицательного давления
  - шаг продвигает время
  - нет NaN на длинном прогоне
  - связь с плазмой
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from physical_modules.vacuum_monitor import VacuumMonitor

K_B = 1.380649e-23


class TestPressure:
    """Давление из плотности — P = n * k_B * T."""

    def test_pressure_from_density(self):
        vm = VacuumMonitor({"gas_temp": 300.0})
        n = 1e20  # м⁻³
        P = vm.compute_pressure(n)
        expected = n * K_B * 300.0
        assert np.isclose(P, expected)
        assert P > 0


class TestConductance:
    """Проводимость канала."""

    def test_conductance_viscous(self):
        """Вязкий режим: Kn << 1, C пропорциональна P."""
        vm = VacuumMonitor({
            "channel_diameter": 0.01,
            "channel_length": 1.0,
            "gas_viscosity": 1.8e-5,
            "initial_pressure": 1e5,
        })
        Kn = vm.compute_knudsen(1e5)
        assert Kn < 0.01, f"Kn={Kn}, ожидался вязкий режим"

        C = vm.compute_conductance(1e5)
        d = 0.01
        L = 1.0
        mu = 1.8e-5
        expected = (np.pi * d**4 / (128 * mu * L)) * 1e5
        assert np.isclose(C, expected)
        assert C > 0

    def test_conductance_molecular(self):
        """Молекулярный режим: Kn >> 1, C не зависит от P."""
        vm = VacuumMonitor({
            "channel_diameter": 0.01,
            "channel_length": 1.0,
            "gas_temp": 300.0,
            "molecule_mass": 4.65e-26,
            "initial_pressure": 1e-3,
        })
        Kn = vm.compute_knudsen(1e-3)
        assert Kn > 1.0, f"Kn={Kn}, ожидался молекулярный режим"

        C = vm.compute_conductance(1e-3)
        d = 0.01
        L = 1.0
        v_bar = np.sqrt(8 * K_B * 300.0 / (np.pi * 4.65e-26))
        expected = (np.pi * d**3 / (12 * L)) * v_bar
        assert np.isclose(C, expected)
        assert C > 0

        # C не зависит от P в молекулярном режиме
        C2 = vm.compute_conductance(1e-5)
        assert np.isclose(C, C2, rtol=1e-6)


class TestFlow:
    """Поток газа через канал."""

    def test_flow_proportional_to_delta_P(self):
        """Q = C * Delta P — в молекулярном режиме C константа,
        поэтому Q прямо пропорциональна Delta P."""
        vm = VacuumMonitor({
            "channel_diameter": 0.01,
            "channel_length": 1.0,
            "initial_pressure": 1e-3,  # молекулярный режим
        })
        # Фиксируем P2, меняем P1 → Delta P удваивается
        Q1 = vm.compute_flow(2e-3, 1e-3)   # Delta P = 1e-3
        Q2 = vm.compute_flow(3e-3, 1e-3)   # Delta P = 2e-3
        ratio = Q2 / Q1 if Q1 != 0 else 0
        assert np.isclose(ratio, 2.0, rtol=0.01)


class TestPumpDown:
    """Откачка камеры."""

    def test_pump_down(self):
        """Давление падает при откачке."""
        vm = VacuumMonitor({
            "volume": 1.0,
            "pump_speed": 0.5,
            "initial_pressure": 1e5,
        })
        P0 = vm.pressure
        for _ in range(100):
            vm.step(dt=0.1)
        assert vm.pressure < P0
        assert vm.pressure > 0

    def test_no_negative_pressure(self):
        """Давление не уходит в минус при долгой откачке."""
        vm = VacuumMonitor({
            "volume": 0.1,
            "pump_speed": 1.0,
            "initial_pressure": 1e3,
        })
        for _ in range(10000):
            vm.step(dt=0.01)
        assert vm.pressure >= 0.0
        assert not np.isnan(vm.pressure)


class TestStep:
    """Шаг симуляции."""

    def test_step_advances(self):
        """step(dt) продвигает время."""
        vm = VacuumMonitor({"initial_pressure": 1e3})
        t0 = vm.time
        vm.step(dt=0.5)
        assert np.isclose(vm.time, t0 + 0.5)

    def test_no_nan_long_run(self):
        """10000 шагов — без NaN и inf."""
        vm = VacuumMonitor({
            "volume": 1.0,
            "pump_speed": 0.1,
            "initial_pressure": 1e5,
        })
        for _ in range(10000):
            vm.step(dt=0.01, Q_in=1.0)
        assert np.isfinite(vm.pressure)
        assert vm.pressure >= 0


class TestCouplingWithPlasma:
    """Связь вакуума с плазмой."""

    def test_coupling_with_plasma(self):
        """run() принимает plasma results и сохраняет их."""
        vm = VacuumMonitor({"initial_pressure": 1e3})
        vm.init()
        external = {
            "temperature_plasma": 1e6,
            "pressure_plasma": 5e4,
        }
        results = vm.run(external=external)
        assert results is not None
        assert results["plasma_temp"] == 1e6
        assert results["plasma_pressure"] == 5e4
        assert results["pressure_gas"] > 0
        assert "mean_free_path" in results
        assert "knudsen_number" in results
        assert "conductance" in results
