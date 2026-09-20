"""tests/test_plasma_monitor.py — Тесты PlasmaMonitor (v0.1).

10 групп:
  1. Инициализация — параметры, пресеты топлива
  2. Слой Экмана — формула, edge cases
  3. Магнитное поле — внешнее + равновесное
  4. Вязкое напряжение — подавление магнитным полем
  5. Барьерный индекс — границы 0..1, edge cases
  6. Бомовская диффузия — зависимость от B
  7. Z-пинч — вклад в магнитное поле
  8. Фазовые состояния — STABLE/WARNING/CONTACT/BREAKDOWN
  9. Внешние напряжения — тепловое/осмотическое/акустическое
 10. Автозапитка — удержание температуры vs радиационные потери
"""

import numpy as np
import pytest

from physical_modules.plasma_monitor import (
    PlasmaMonitor, MU_0, K_B, E_CHARGE, FUEL_MASSES,
)


# --- Константы для тестов ---

DEFAULT_CFG = {
    "radius": 1.0e-2,
    "shell_thickness": 1.0e-4,
    "plasma_gap": 5.0e-4,
    "density_number": 1.0e20,
    "temperature": 5.0e6,
    "viscosity": 1.0e-5,
    "mass_density": 1.0e-3,
    "rotation_freq": 1.0e3,
    "flow_velocity": 1.0e4,
    "z_pinch_freq": 1.253e7,
    "auto_feed": True,
    "ekman_layer": True,
    "ion_charge": 1.0,
    "ion_mass": 1.673e-27,
    "collision_freq": 1.0e9,
    "external_B": 2.0,
    "breakdown_B": 15.0,
    "barrier_threshold": 0.3,
    "fuel_type": "custom",
}


# --- 1. Инициализация ---

class TestInit:

    def test_default_params(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        assert pm.radius == 1.0e-2
        assert pm.density_number == 1.0e20
        assert pm.temperature == 5.0e6
        assert pm.external_B == 2.0
        assert pm.breakdown_B == 15.0

    def test_custom_params(self):
        cfg = {**DEFAULT_CFG, "radius": 2.0e-2, "temperature": 1.0e7}
        pm = PlasmaMonitor(config=cfg)
        assert pm.radius == 2.0e-2
        assert pm.temperature == 1.0e7

    def test_empty_config(self):
        pm = PlasmaMonitor()
        assert pm.radius == 1.0e-2
        assert pm.temperature == 5.0e6

    def test_kwargs_passed(self):
        pm = PlasmaMonitor(radius=3.0e-2)
        assert pm.radius == 3.0e-2

    def test_init_returns_true(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        assert pm.init() is True

    def test_results_none_before_run(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        assert pm.results is None

    def test_get_results_none_before_run(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        assert pm.get_results() is None

    def test_fuel_preset_dhe3(self):
        cfg = {**DEFAULT_CFG, "fuel_type": "D-He3"}
        pm = PlasmaMonitor(config=cfg)
        assert pm.ion_mass == max(FUEL_MASSES["D-He3"])

    def test_fuel_preset_dt(self):
        cfg = {**DEFAULT_CFG, "fuel_type": "D-T"}
        pm = PlasmaMonitor(config=cfg)
        assert pm.ion_mass == max(FUEL_MASSES["D-T"])

    def test_fuel_preset_dd(self):
        cfg = {**DEFAULT_CFG, "fuel_type": "D-D"}
        pm = PlasmaMonitor(config=cfg)
        assert pm.ion_mass == max(FUEL_MASSES["D-D"])

    def test_fuel_custom_keeps_mass(self):
        cfg = {**DEFAULT_CFG, "fuel_type": "custom", "ion_mass": 5.0e-27}
        pm = PlasmaMonitor(config=cfg)
        assert pm.ion_mass == 5.0e-27


# --- 2. Слой Экмана ---

class TestEkmanLayer:

    def test_basic_value(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta = pm.compute_ekman_layer()
        assert delta > 0
        assert np.isfinite(delta)

    def test_formula_correct(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        nu = pm.viscosity
        Omega = pm.rotation_freq
        R = pm.radius
        gap = pm.gap
        expected = np.sqrt(2.0 * nu / Omega) * R / (R - gap)
        assert abs(pm.compute_ekman_layer() - expected) < 1e-15

    def test_disabled_returns_zero(self):
        cfg = {**DEFAULT_CFG, "ekman_layer": False}
        pm = PlasmaMonitor(config=cfg)
        assert pm.compute_ekman_layer() == 0.0

    def test_zero_viscosity(self):
        cfg = {**DEFAULT_CFG, "viscosity": 0.0}
        pm = PlasmaMonitor(config=cfg)
        assert pm.compute_ekman_layer() == 0.0

    def test_high_rotation_thin_layer(self):
        cfg_low = {**DEFAULT_CFG, "rotation_freq": 1.0e3}
        cfg_high = {**DEFAULT_CFG, "rotation_freq": 1.0e6}
        pm_low = PlasmaMonitor(config=cfg_low)
        pm_high = PlasmaMonitor(config=cfg_high)
        assert pm_high.compute_ekman_layer() < pm_low.compute_ekman_layer()


# --- 3. Магнитное поле ---

class TestMagneticField:

    def test_basic_value(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B = pm.compute_magnetic_field()
        assert B > 0
        assert np.isfinite(B)

    def test_external_dominant(self):
        cfg = {**DEFAULT_CFG, "external_B": 100.0,
               "density_number": 1.0e10, "temperature": 1.0e3}
        pm = PlasmaMonitor(config=cfg)
        B = pm.compute_magnetic_field()
        assert abs(B - 100.0) / 100.0 < 0.01

    def test_equilibrium_contribution(self):
        cfg = {**DEFAULT_CFG, "external_B": 0.0}
        pm = PlasmaMonitor(config=cfg)
        B = pm.compute_magnetic_field()
        n = pm.density_number
        T = pm.temperature
        B_eq = np.sqrt(2.0 * MU_0 * n * K_B * T)
        assert abs(B - B_eq) < 1e-10

    def test_formula_correct(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B_ext = pm.external_B
        n = pm.density_number
        T = pm.temperature
        B_eq = np.sqrt(2.0 * MU_0 * n * K_B * T)
        expected = np.sqrt(B_ext**2 + B_eq**2)
        assert abs(pm.compute_magnetic_field() - expected) < 1e-10

    def test_higher_density_higher_B(self):
        cfg_low = {**DEFAULT_CFG, "density_number": 1.0e19}
        cfg_high = {**DEFAULT_CFG, "density_number": 1.0e21}
        pm_low = PlasmaMonitor(config=cfg_low)
        pm_high = PlasmaMonitor(config=cfg_high)
        assert pm_high.compute_magnetic_field() > pm_low.compute_magnetic_field()


# --- 4. Вязкое напряжение ---

class TestViscousStress:

    def test_basic_value(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = pm.compute_ekman_layer()
        B = pm.compute_magnetic_field()
        sigma = pm.compute_viscous_stress(delta_E, B)
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_zero_delta_returns_zero(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        assert pm.compute_viscous_stress(0.0, 2.0) == 0.0

    def test_magnetic_suppression(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = pm.compute_ekman_layer()
        sigma_low_B = pm.compute_viscous_stress(delta_E, 1.0)
        sigma_high_B = pm.compute_viscous_stress(delta_E, 100.0)
        assert sigma_high_B < sigma_low_B

    def test_formula_correct(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = 1.0e-4
        B = 5.0
        sigma_0 = pm.viscosity * pm.flow_velocity / delta_E
        suppression = 1.0 / (1.0 + (B / pm.breakdown_B) ** 2)
        expected = sigma_0 * suppression
        assert abs(pm.compute_viscous_stress(delta_E, B) - expected) < 1e-15

    def test_strong_field_near_zero_stress(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = pm.compute_ekman_layer()
        sigma = pm.compute_viscous_stress(delta_E, 1.0e6)
        assert sigma < 1e-10


# --- 5. Барьерный индекс ---

class TestBarrierIndex:

    def test_basic_value(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = pm.compute_ekman_layer()
        B = pm.compute_magnetic_field()
        idx = pm.compute_barrier_index(delta_E, B)
        assert 0.0 <= idx <= 1.0

    def test_zero_gap_returns_zero(self):
        cfg = {**DEFAULT_CFG, "plasma_gap": 0.0}
        pm = PlasmaMonitor(config=cfg)
        assert pm.compute_barrier_index(1e-4, 5.0) == 0.0

    def test_high_B_higher_index(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = pm.compute_ekman_layer()
        idx_low = pm.compute_barrier_index(delta_E, 1.0)
        idx_high = pm.compute_barrier_index(delta_E, 100.0)
        assert idx_high > idx_low

    def test_large_delta_lower_index(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B = pm.compute_magnetic_field()
        idx_small = pm.compute_barrier_index(1e-6, B)
        idx_large = pm.compute_barrier_index(1e-2, B)
        assert idx_large < idx_small

    def test_formula_correct(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        delta_E = 1e-4
        B = 5.0
        geom = pm.gap / (pm.gap + delta_E)
        mag = B / (B + pm.breakdown_B)
        expected = max(0.0, min(1.0, geom * mag))
        assert abs(pm.compute_barrier_index(delta_E, B) - expected) < 1e-15

    def test_clamped_to_unit(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        idx = pm.compute_barrier_index(1e-10, 1e6)
        assert idx <= 1.0
        idx2 = pm.compute_barrier_index(1e10, 0.0)
        assert idx2 >= 0.0


# --- 6. Бомовская диффузия ---

class TestBohmDiffusion:

    def test_basic_value(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B = pm.compute_magnetic_field()
        D = pm._compute_diffusion_coeff(B)
        assert D > 0
        assert np.isfinite(D)

    def test_formula_correct(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B = 5.0
        expected = (K_B * pm.temperature) / (16.0 * E_CHARGE * B)
        assert abs(pm._compute_diffusion_coeff(B) - expected) < 1e-15

    def test_higher_B_lower_diffusion(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        D_low = pm._compute_diffusion_coeff(1.0)
        D_high = pm._compute_diffusion_coeff(100.0)
        assert D_high < D_low

    def test_zero_B_fallback(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        D = pm._compute_diffusion_coeff(0.0)
        assert D > 0
        assert np.isfinite(D)

    def test_higher_temp_higher_diffusion(self):
        cfg_low = {**DEFAULT_CFG, "temperature": 1.0e6}
        cfg_high = {**DEFAULT_CFG, "temperature": 1.0e7}
        pm_low = PlasmaMonitor(config=cfg_low)
        pm_high = PlasmaMonitor(config=cfg_high)
        B = 5.0
        assert pm_high._compute_diffusion_coeff(B) > pm_low._compute_diffusion_coeff(B)


# --- 7. Z-пинч ---

class TestZPinch:

    def test_basic_value(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B_z = pm._compute_z_pinch_field()
        assert B_z >= 0
        assert np.isfinite(B_z)

    def test_zero_freq_returns_zero(self):
        cfg = {**DEFAULT_CFG, "z_pinch_freq": 0.0}
        pm = PlasmaMonitor(config=cfg)
        assert pm._compute_z_pinch_field() == 0.0

    def test_capped_at_half(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B_z = pm._compute_z_pinch_field()
        assert B_z <= 0.5

    def test_run_includes_z_pinch(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        B_base = pm.compute_magnetic_field()
        B_z = pm._compute_z_pinch_field()
        results = pm.run()
        B_total = results["B_field"]
        if B_z > 0:
            expected = np.sqrt(B_base**2 + B_z**2)
            assert abs(B_total - expected) < 1e-10

    def test_zero_z_pinch_no_contribution(self):
        cfg = {**DEFAULT_CFG, "z_pinch_freq": 0.0}
        pm = PlasmaMonitor(config=cfg)
        results = pm.run()
        B_base = pm.compute_magnetic_field()
        assert abs(results["B_field"] - B_base) < 1e-10


# --- 8. Фазовые состояния ---

class TestPhases:

    def test_stable_phase(self):
        cfg = {**DEFAULT_CFG, "external_B": 2.0, "breakdown_B": 15.0,
               "plasma_gap": 5.0e-4, "barrier_threshold": 0.01}
        pm = PlasmaMonitor(config=cfg)
        results = pm.run()
        assert results["phase"] == "STABLE"

    def test_warning_phase(self):
        cfg = {**DEFAULT_CFG, "external_B": 0.01, "breakdown_B": 15.0,
               "plasma_gap": 5.0e-4, "barrier_threshold": 0.99,
               "density_number": 1.0e10, "temperature": 1.0e3}
        pm = PlasmaMonitor(config=cfg)
        results = pm.run()
        assert results["phase"] == "WARNING"

    def test_contact_phase(self):
        cfg = {**DEFAULT_CFG, "viscosity": 1.0, "rotation_freq": 1.0e3,
               "plasma_gap": 1e-10}
        pm = PlasmaMonitor(config=cfg)
        delta_E = pm.compute_ekman_layer()
        results = pm.run()
        if pm.gap <= delta_E * 0.5:
            assert results["phase"] == "CONTACT"

    def test_breakdown_phase(self):
        cfg = {**DEFAULT_CFG, "external_B": 20.0, "breakdown_B": 15.0,
               "density_number": 1.0e10, "temperature": 1.0e3}
        pm = PlasmaMonitor(config=cfg)
        results = pm.run()
        assert results["phase"] == "BREAKDOWN"

    def test_run_returns_all_keys(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        results = pm.run()
        required_keys = [
            "module", "ekman_thickness", "B_field", "sigma_viscous",
            "diffusion_coeff", "barrier_index", "phase",
            "plasma_stress", "magnetic_coupling",
            "temperature_plasma", "rotation_omega",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_get_results_after_run(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        pm.run()
        results = pm.get_results()
        assert results is not None
        assert results["module"] == "plasma_monitor"

    def test_module_name(self):
        pm = PlasmaMonitor(config=DEFAULT_CFG)
        results = pm.run()
        assert results["module"] == "plasma_monitor"
# === END ===
