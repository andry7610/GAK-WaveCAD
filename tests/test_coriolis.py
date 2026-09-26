"""
Тесты Кориолиса (метрика Гёделя) — v0.4

Проверяют:
  1. Нулевая угловая скорость → нулевая сила
  2. Линейность по Omega
  3. Линейность по v
  4. Направление (перпендикулярно Omega и v — модуль всегда положителен)
  5. Интеграция в results
"""

import numpy as np
import pytest
from physical_modules.plasma_monitor import PlasmaMonitor


class TestCoriolisZero:
    """При omega = 0 сила Кориолиса = 0."""

    def test_coriolis_zero_omega(self):
        p = PlasmaMonitor({"rotation_freq": 0.0, "flow_velocity": 1.0e4})
        cor = p.compute_coriolis_force()
        assert cor["force_magnitude"] == 0.0
        assert cor["coriolis_stress"] == 0.0

    def test_coriolis_zero_velocity(self):
        p = PlasmaMonitor({"rotation_freq": 1.0e3, "flow_velocity": 0.0})
        cor = p.compute_coriolis_force()
        assert cor["force_magnitude"] == 0.0
        assert cor["coriolis_stress"] == 0.0

    def test_coriolis_disabled(self):
        p = PlasmaMonitor({
            "rotation_freq": 1.0e3,
            "flow_velocity": 1.0e4,
            "coriolis_enabled": False,
        })
        cor = p.compute_coriolis_force()
        assert cor["force_magnitude"] == 0.0
        assert cor["coriolis_stress"] == 0.0


class TestCoriolisLinearOmega:
    """Линейность по Omega: F ~ Omega."""

    def test_coriolis_linear_in_omega(self):
        omega1 = 1.0e3
        omega2 = 5.0e3
        v = 1.0e4
        rho = 1.0e-3

        p1 = PlasmaMonitor({
            "rotation_freq": omega1,
            "flow_velocity": v,
            "mass_density": rho,
        })
        p2 = PlasmaMonitor({
            "rotation_freq": omega2,
            "flow_velocity": v,
            "mass_density": rho,
        })

        F1 = p1.compute_coriolis_force()["force_magnitude"]
        F2 = p2.compute_coriolis_force()["force_magnitude"]

        assert F2 / F1 == pytest.approx(omega2 / omega1, rel=1e-10)

    def test_coriolis_stress_linear_in_omega(self):
        omega1 = 1.0e3
        omega2 = 3.0e3
        v = 1.0e4
        rho = 1.0e-3

        p1 = PlasmaMonitor({
            "rotation_freq": omega1,
            "flow_velocity": v,
            "mass_density": rho,
        })
        p2 = PlasmaMonitor({
            "rotation_freq": omega2,
            "flow_velocity": v,
            "mass_density": rho,
        })

        S1 = p1.compute_coriolis_force()["coriolis_stress"]
        S2 = p2.compute_coriolis_force()["coriolis_stress"]

        assert S2 / S1 == pytest.approx(omega2 / omega1, rel=1e-10)


class TestCoriolisLinearVelocity:
    """Линейность по v: F ~ v."""

    def test_coriolis_linear_in_velocity(self):
        omega = 2.0e3
        v1 = 1.0e4
        v2 = 4.0e4
        rho = 1.0e-3

        p1 = PlasmaMonitor({
            "rotation_freq": omega,
            "flow_velocity": v1,
            "mass_density": rho,
        })
        p2 = PlasmaMonitor({
            "rotation_freq": omega,
            "flow_velocity": v2,
            "mass_density": rho,
        })

        F1 = p1.compute_coriolis_force()["force_magnitude"]
        F2 = p2.compute_coriolis_force()["force_magnitude"]

        assert F2 / F1 == pytest.approx(v2 / v1, rel=1e-10)


class TestCoriolisDirection:
    """Модуль силы всегда положителен (перпендикулярно — модуль)."""

    def test_coriolis_magnitude_positive(self):
        p = PlasmaMonitor({
            "rotation_freq": 1.0e3,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })
        cor = p.compute_coriolis_force()
        assert cor["force_magnitude"] > 0
        assert cor["coriolis_stress"] > 0

    def test_coriolis_negative_omega_positive_force(self):
        """Отрицательная Omega (вращение в другую сторону) — модуль тот же."""
        p_pos = PlasmaMonitor({
            "rotation_freq": 1.0e3,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })
        p_neg = PlasmaMonitor({
            "rotation_freq": -1.0e3,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })

        F_pos = p_pos.compute_coriolis_force()["force_magnitude"]
        F_neg = p_neg.compute_coriolis_force()["force_magnitude"]

        assert F_pos == pytest.approx(F_neg, rel=1e-10)


class TestCoriolisFormula:
    """Точность формулы: F = 2 * Omega * v, stress = 2 * rho * Omega * v."""

    def test_force_formula(self):
        omega = 1.0e3
        v = 1.0e4
        p = PlasmaMonitor({
            "rotation_freq": omega,
            "flow_velocity": v,
            "mass_density": 1.0e-3,
        })
        cor = p.compute_coriolis_force()
        expected_F = 2.0 * omega * v
        assert cor["force_magnitude"] == pytest.approx(expected_F, rel=1e-10)

    def test_stress_formula(self):
        omega = 1.0e3
        v = 1.0e4
        rho = 1.0e-3
        p = PlasmaMonitor({
            "rotation_freq": omega,
            "flow_velocity": v,
            "mass_density": rho,
        })
        cor = p.compute_coriolis_force()
        expected_stress = 2.0 * rho * omega * v
        assert cor["coriolis_stress"] == pytest.approx(expected_stress, rel=1e-10)


class TestCoriolisInResults:
    """Кориолис в results после run()."""

    def test_run_has_coriolis_stress(self):
        p = PlasmaMonitor({
            "rotation_freq": 1.0e3,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })
        p.init()
        results = p.run()
        assert "coriolis_stress" in results
        assert results["coriolis_stress"] > 0

    def test_run_has_coriolis_force(self):
        p = PlasmaMonitor({
            "rotation_freq": 1.0e3,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })
        p.init()
        results = p.run()
        assert "coriolis_force" in results
        assert results["coriolis_force"] > 0

    def test_run_zero_omega_zero_coriolis(self):
        p = PlasmaMonitor({
            "rotation_freq": 0.0,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })
        p.init()
        results = p.run()
        assert results["coriolis_stress"] == 0.0
        assert results["coriolis_force"] == 0.0

    def test_plasma_stress_includes_coriolis(self):
        """plasma_stress = sigma_viscous + rho*v^2 + coriolis_stress."""
        p = PlasmaMonitor({
            "rotation_freq": 1.0e3,
            "flow_velocity": 1.0e4,
            "mass_density": 1.0e-3,
        })
        p.init()
        results = p.run()

        sigma_visc = results["sigma_viscous"]
        rho_v2 = p.mass_density * p.flow_velocity**2
        cor_stress = results["coriolis_stress"]
        expected = sigma_visc + rho_v2 + cor_stress

        assert results["plasma_stress"] == pytest.approx(expected, rel=1e-6)
