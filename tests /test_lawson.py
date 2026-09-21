"""
tests/test_lawson.py — Критерий Лоусона для плазмы.

Проверяет: зажигается ли термоядерная реакция при заданных параметрах.

Критерий Лоусона:
    n * T_keV * tau_E >= n_tau_T_threshold

Пороги (тройное произведение):
    D-T:   3e21  кэВ·с/м³
    D-D:   1e24  кэВ·с/м³
    D-He3: 1e22  кэВ·с/м³
"""

import numpy as np
import pytest

from physical_modules.plasma_monitor import PlasmaMonitor


# --- Константы ---
K_B = 1.380649e-23       # Дж/К
E_CHARGE = 1.602176634e-19  # Кл

# Пороги Лоусона (тройное произведение, кэВ·с/м³)
LAWSON_THRESHOLDS = {
    "D-T":   3.0e21,
    "D-D":   1.0e24,
    "D-He3": 1.0e22,
}


def _kelvin_to_kev(T_K):
    """Перевод Кельвин → кэВ."""
    return T_K * K_B / (E_CHARGE * 1e3)


def _make_plasma(**overrides):
    """Создать PlasmaMonitor с дефолтными параметрами + override."""
    cfg = {
        "radius": 1.0e-2,         # 1 см
        "density_number": 1.0e20, # 1e20 м⁻³
        "temperature": 1.16e8,    # ~10 кэВ
        "external_B": 2.0,        # 2 Тл
        "fuel_type": "D-T",
    }
    cfg.update(overrides)
    return PlasmaMonitor(config=cfg)


# ============================================================
# Тест 1: D-T выше порога — зажигается
# ============================================================

def test_lawson_dt_above_threshold():
    """D-T: n=1e20, T=10 кэВ, tau_E=1 с → n*T*tau_E = 1e22 > 3e21 → зажигается."""
    p = _make_plasma(
        density_number=1.0e20,
        temperature=1.16e8,   # ~10 кэВ
        external_B=2.0,
        fuel_type="D-T",
    )

    # Вычислить tau_E (метод будет добавлен)
    tau_E = p.compute_tau_E()  # секунд

    # Тройное произведение
    T_keV = _kelvin_to_kev(p.temperature)
    n_tau_T = p.density_number * T_keV * tau_E

    threshold = LAWSON_THRESHOLDS["D-T"]

    # Проверить метод check_lawson
    ignited = p.check_lawson()

    assert n_tau_T > threshold, (
        f"n*T*tau_E = {n_tau_T:.2e} <= {threshold:.2e}"
    )
    assert ignited is True, (
        f"Должно зажигаться: n*T*tau_E = {n_tau_T:.2e} > {threshold:.2e}"
    )


# ============================================================
# Тест 2: D-T ниже порога — не зажигается
# ============================================================

def test_lawson_dt_below_threshold():
    """D-T: n=1e19, T=1 кэВ → n*T*tau_E << 3e21 → не зажигается."""
    p = _make_plasma(
        density_number=1.0e19,   # в 10 раз меньше
        temperature=1.16e7,      # ~1 кэВ (в 10 раз меньше)
        external_B=0.5,          # слабое поле → малое tau_E
        fuel_type="D-T",
    )

    tau_E = p.compute_tau_E()

    T_keV = _kelvin_to_kev(p.temperature)
    n_tau_T = p.density_number * T_keV * tau_E

    threshold = LAWSON_THRESHOLDS["D-T"]

    ignited = p.check_lawson()

    assert n_tau_T < threshold, (
        f"n*T*tau_E = {n_tau_T:.2e} >= {threshold:.2e}"
    )
    assert ignited is False, (
        f"Не должно зажигаться: n*T*tau_E = {n_tau_T:.2e} < {threshold:.2e}"
    )


# ============================================================
# Тест 3: Правильный порог для разных топлив
# ============================================================

@pytest.mark.parametrize("fuel,threshold", [
    ("D-T", 3.0e21),
    ("D-D", 1.0e24),
    ("D-He3", 1.0e22),
])
def test_lawson_uses_correct_threshold(fuel, threshold):
    """Порог Лоусона зависит от типа топлива."""
    p = _make_plasma(
        density_number=1.0e20,
        temperature=1.16e8,  # 10 кэВ
        external_B=2.0,
        fuel_type=fuel,
    )

    tau_E = p.compute_tau_E()

    T_keV = _kelvin_to_kev(p.temperature)
    n_tau_T = p.density_number * T_keV * tau_E

    # Порог должен быть правильным
    expected_threshold = LAWSON_THRESHOLDS[fuel]
    assert expected_threshold == threshold

    # При D-T — зажигается; при D-D — нет (порог в 300 раз выше)
    if fuel == "D-T":
        assert p.check_lawson() is True
    elif fuel == "D-D":
        assert p.check_lawson() is False
