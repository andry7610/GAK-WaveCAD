"""
Тест модуля плазмы (Plasma Monitor).

Проверяет:
  1. Расчёт зоны Экмана
  2. Магнитное поле
  3. Вязкое напряжение
  4. Барьерный индекс
  5. Фазовое состояние
  6. Передачу напряжений от других модулей
  7. Топливные пресеты (массы ионов)
  8. Z-пинч (бегущее поле)
  9. Автозапитка
  10. Флаг зоны Экмана
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_modules.plasma_monitor import PlasmaMonitor


# --- Базовый конфиг с обновлёнными параметрами (1 см, 5 МК, 2 Тл) ---
BASE_CONFIG = {
    "radius": 1.0e-2,
    "shell_thickness": 1.0e-4,
    "plasma_gap": 5.0e-4,
    "density_number": 1.0e20,
    "temperature": 5.0e6,
    "viscosity": 1.0e-5,
    "mass_density": 1.0e-3,
    "rotation_freq": 1.0e3,
    "flow_velocity": 1.0e4,
    "ion_charge": 1.0,
    "ion_mass": 1.673e-27,
    "collision_freq": 1.0e9,
    "external_B": 2.0,
    "breakdown_B": 15.0,
    "barrier_threshold": 0.3,
    "z_pinch_freq": 1.253e7,
    "auto_feed": True,
    "ekman_layer": True,
    "fuel_type": "D-He3",
}


def test_ekman_layer():
    """Толщина зоны Экмана должна быть положительной и физически осмысленной."""
    config = BASE_CONFIG.copy()
    pm = PlasmaMonitor(config)
    delta_E = pm.compute_ekman_layer()

    assert delta_E > 0, "Толщина зоны Экмана должна быть положительной"
    assert delta_E < 0.01, "Толщина зоны Экмана нереалистично велика"

    # Базовая формула: delta = sqrt(2 * nu / Omega) * geom
    nu = config["viscosity"]
    Omega = config["rotation_freq"]
    R = config["radius"]
    gap = config["plasma_gap"]
    delta_E_base = np.sqrt(2 * nu / Omega)
    geom = R / (R - gap)
    assert np.isclose(delta_E, delta_E_base * geom, rtol=1e-6), \
        f"delta_E={delta_E}, expected={delta_E_base * geom}"
    print(f"  [OK] Ekman thickness: {delta_E:.4e} m")
    return pm


def test_magnetic_field():
    """Магнитное поле должно включать внешний и равновесный вклады."""
    config = BASE_CONFIG.copy()
    config["external_B"] = 2.0
    config["density_number"] = 1.0e20
    config["temperature"] = 5.0e6
    pm = PlasmaMonitor(config)
    B = pm.compute_magnetic_field()

    assert B > 2.0, "B должно быть больше внешнего поля"
    mu_0 = 4 * np.pi * 1e-7
    k_B = 1.380649e-23
    B_eq = np.sqrt(2 * mu_0 * 1.0e20 * k_B * 5.0e6)
    B_expected = np.sqrt(2.0**2 + B_eq**2)
    assert np.isclose(B, B_expected, rtol=1e-6), f"B={B}, expected={B_expected}"
    print(f"  [OK] Magnetic field: {B:.4e} T (B_eq={B_eq:.4e} T)")
    return pm


def test_viscous_stress():
    """Вязкое напряжение должно быть положительным и подавляться магнитным полем."""
    config = BASE_CONFIG.copy()
    pm = PlasmaMonitor(config)
    delta_E = 5.0e-4
    B = 2.0

    sigma = pm.compute_viscous_stress(delta_E, B)
    assert sigma > 0, "Вязкое напряжение должно быть положительным"

    # Проверка подавления магнитным полем
    sigma_no_B = pm.compute_viscous_stress(delta_E, 0.0)
    assert sigma < sigma_no_B, "Магнитное поле должно подавлять напряжение"
    print(f"  [OK] Viscous stress: {sigma:.4e} Pa (no B: {sigma_no_B:.4e} Pa)")
    return pm


def test_barrier_index():
    """Барьерный индекс должен быть в диапазоне 0..1."""
    config = BASE_CONFIG.copy()
    pm = PlasmaMonitor(config)

    # Большой зазор, сильное поле — высокий барьер
    idx = pm.compute_barrier_index(5.0e-4, 5.0)
    assert 0 < idx <= 1, f"Барьер {idx} вне диапазона"

    # Нулевой зазор — нулевой барьер
    pm.gap = 0
    idx_zero = pm.compute_barrier_index(5.0e-4, 5.0)
    assert idx_zero == 0, "При нулевом зазоре барьер должен быть 0"
    print(f"  [OK] Barrier index: {idx:.3f}")
    return pm


def test_full_run():
    """Полный прогон модуля с внешними напряжениями."""
    config = BASE_CONFIG.copy()
    pm = PlasmaMonitor(config)

    # Без внешних напряжений
    result = pm.run()
    assert result["phase"] in ["STABLE", "WARNING", "CONTACT", "BREAKDOWN"]
    assert result["ekman_thickness"] > 0
    assert result["B_field"] > 0
    assert result["barrier_index"] >= 0
    print(f"  [OK] Full run (no external): phase={result['phase']}, "
          f"delta_E={result['ekman_thickness']:.4e}, "
          f"B={result['B_field']:.4e}, "
          f"barrier={result['barrier_index']:.3f}")

    # С внешними напряжениями (от осмоса и термалки)
    external = {
        "sigma_thermal": 1e6,
        "sigma_osmotic": 1e5,
        "acoustic_freq_shift": 100.0,
    }
    result_ext = pm.run(external_stress=external)
    assert result_ext["temperature_plasma"] > 5.0e6, \
        "Тепловое напряжение должно разогревать плазму"
    assert result_ext["rotation_omega"] > 1.0e3, \
        "Акустический сдвиг должен менять Omega"
    print(f"  [OK] Full run (external): T={result_ext['temperature_plasma']:.4e} K, "
          f"Omega={result_ext['rotation_omega']:.4e} rad/s")
    return pm


def test_coupling_output():
    """Проверка формата выходных данных для Coupling Monitor."""
    config = BASE_CONFIG.copy()
    pm = PlasmaMonitor(config)
    result = pm.run()

    required_keys = [
        "module", "ekman_thickness", "B_field", "sigma_viscous",
        "diffusion_coeff", "barrier_index", "phase",
        "plasma_stress", "magnetic_coupling",
        "temperature_plasma", "rotation_omega",
    ]
    for key in required_keys:
        assert key in result, f"Отсутствует ключ '{key}' в выходных данных"
    print("  [OK] Coupling output format verified")


def test_fuel_mass():
    """Топливные пресеты должны пересчитывать массы ионов."""
    # D-He3
    config = BASE_CONFIG.copy()
    config["fuel_type"] = "D-He3"
    pm = PlasmaMonitor(config)
    m_D = 3.34e-27
    m_He3 = 5.01e-27
    assert np.isclose(pm.ion_mass, m_D, rtol=1e-2) or \
           np.isclose(pm.ion_mass, m_He3, rtol=1e-2), \
        f"ion_mass={pm.ion_mass}, expected D or He3"
    print(f"  [OK] D-He3: ion_mass={pm.ion_mass:.3e} kg")

    # D-T
    config["fuel_type"] = "D-T"
    pm = PlasmaMonitor(config)
    m_T = 5.01e-27
    assert np.isclose(pm.ion_mass, m_T, rtol=1e-2), \
        f"ion_mass={pm.ion_mass}, expected tritium {m_T}"
    print(f"  [OK] D-T: ion_mass={pm.ion_mass:.3e} kg")

    # Fallback — без fuel_type
    config["fuel_type"] = "custom"
    config["ion_mass"] = 1.673e-27
    pm = PlasmaMonitor(config)
    assert np.isclose(pm.ion_mass, 1.673e-27, rtol=1e-6), \
        f"ion_mass={pm.ion_mass}, expected fallback 1.673e-27"
    print(f"  [OK] Custom fallback: ion_mass={pm.ion_mass:.3e} kg")
    return pm


def test_z_pinch():
    """Z-пинч (бегущее поле) должен влиять на динамику плазмы."""
    config_with = BASE_CONFIG.copy()
    config_with["z_pinch_freq"] = 1.253e7
    pm_with = PlasmaMonitor(config_with)
    result_with = pm_with.run()

    config_without = BASE_CONFIG.copy()
    config_without["z_pinch_freq"] = 0.0
    pm_without = PlasmaMonitor(config_without)
    result_without = pm_without.run()

    # При активном Z-пинче эффективное поле должно быть сильнее
    assert result_with["B_field"] >= result_without["B_field"], \
        "Z-пинч должен усиливать магнитное поле"
    print(f"  [OK] Z-pinch: B_with={result_with['B_field']:.4e} T, "
          f"B_without={result_without['B_field']:.4e} T")
    return pm_with


def test_auto_feed():
    """Автозапитка должна поддерживать температуру плазмы."""
    config_on = BASE_CONFIG.copy()
    config_on["auto_feed"] = True
    pm_on = PlasmaMonitor(config_on)
    result_on = pm_on.run()

    config_off = BASE_CONFIG.copy()
    config_off["auto_feed"] = False
    pm_off = PlasmaMonitor(config_off)
    result_off = pm_off.run()

    # С автозапиткой температура должна быть не ниже исходной
    assert result_on["temperature_plasma"] >= config_on["temperature"], \
        "Автозапитка должна поддерживать или повышать температуру"
    # Без автозапитки температура может падать
    print(f"  [OK] Auto-feed ON:  T={result_on['temperature_plasma']:.4e} K")
    print(f"  [OK] Auto-feed OFF: T={result_off['temperature_plasma']:.4e} K")
    return pm_on


def test_ekman_flag():
    """Флаг ekman_layer должен включать/выключать расчёт зоны Экмана."""
    config_on = BASE_CONFIG.copy()
    config_on["ekman_layer"] = True
    pm_on = PlasmaMonitor(config_on)
    result_on = pm_on.run()
    assert result_on["ekman_thickness"] > 0, \
        "При ekman_layer=True толщина должна быть положительной"

    config_off = BASE_CONFIG.copy()
    config_off["ekman_layer"] = False
    pm_off = PlasmaMonitor(config_off)
    result_off = pm_off.run()
    assert result_off["ekman_thickness"] == 0, \
        "При ekman_layer=False толщина должна быть 0"
    print(f"  [OK] Ekman ON:  delta={result_on['ekman_thickness']:.4e} m")
    print(f"  [OK] Ekman OFF: delta={result_off['ekman_thickness']:.4e} m")
    return pm_on


if __name__ == "__main__":
    print("\n=== Plasma Monitor Tests ===\n")

    print("[1] Ekman layer:")
    test_ekman_layer()

    print("[2] Magnetic field:")
    test_magnetic_field()

    print("[3] Viscous stress:")
    test_viscous_stress()

    print("[4] Barrier index:")
    test_barrier_index()

    print("[5] Full run:")
    test_full_run()

    print("[6] Coupling output:")
    test_coupling_output()

    print("[7] Fuel mass presets:")
    test_fuel_mass()

    print("[8] Z-pinch (travelling field):")
    test_z_pinch()

    print("[9] Auto-feed:")
    test_auto_feed()

    print("[10] Ekman layer flag:")
    test_ekman_flag()

    print("\n=== All plasma tests passed ===\n")
