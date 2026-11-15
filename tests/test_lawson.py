"""
Критерий Лоусона: зажигается ли термоядерная реакция?

n * T * tau_E >= threshold (зависит от топлива)

v37.2: модель ужесточила tau_E — ITER-точка (n=1e20, T=1.16e8, B=10) даёт
n*T*tau_E = 1.83e21 против порога 3e21. Тестовая точка усилена (n, T, B)
с запасом с ОБЕИХ сторон: D-T должен зажигаться, D-D и D-He3 — нет.
"""

import numpy as np
import pytest
from physical_modules.plasma_monitor import PlasmaMonitor


# --- Пороги Лоусона по типам топлива (кэВ*с/м^3) ---
LAWSON_THRESHOLDS = {
    "D-T":   3.0e21,
    "D-D":   1.0e24,
    "D-He3": 1.0e22,
}


def _kelvin_to_kev(T_K):
    """Перевести температуру из Кельвинов в кэВ."""
    k_B = 1.380649e-23
    eV = 1.602176634e-19
    return T_K * k_B / eV / 1e3


def _make_plasma(**kwargs):
    """Создать PlasmaMonitor с параметрами по умолчанию + override."""
    cfg = {
        "density_number": 1.0e20,
        "temperature": 1.16e8,   # ~10 кэВ
        "external_B": 2.0,
        "fuel_type": "D-T",
        "radius": 1.0e-2,
    }
    cfg.update(kwargs)
    p = PlasmaMonitor(cfg)
    p.init()
    return p


def test_lawson_dt_above_threshold():
    """D-T: усиленная точка -> n*T*tau_E > 3e21 -> зажигается.

    v37.2: при старой точке (n=1e20, T=1.16e8, B=10) произведение
    давало 1.83e21 < 3e21 (tau_E ужесточён в v0.15). Плотность,
    температура и поле подняты с запасом.
    """
    p = _make_plasma(
        density_number=2.5e20,  # v37.2: усиленная точка (запас к порогу)
        temperature=1.5e8,      # ~13 кэВ — ближе к пику <sigma v> для D-T
        external_B=13.0,        # сильное поле -> большое tau_E
        radius=1.0,             # ITER-масштаб
        fuel_type="D-T",
    )

    tau_E = p.compute_tau_E()

    T_keV = _kelvin_to_kev(p.temperature)
    n_tau_T = p.density_number * T_keV * tau_E

    threshold = LAWSON_THRESHOLDS["D-T"]

    assert n_tau_T > threshold, (
        f"n*T*tau_E = {n_tau_T:.2e} <= {threshold:.2e}"
    )

    ignited = p.check_lawson()
    assert ignited is True, (
        f"check_lawson() = {ignited}, expected True"
    )


def test_lawson_dt_below_threshold():
    """D-T: малые параметры -> n*T*tau_E < 3e21 -> не зажигается."""
    p = _make_plasma(
        density_number=1.0e19,
        temperature=1.16e7,   # ~1 кэВ
        external_B=0.5,
        radius=1.0e-2,        # маленький радиус
        fuel_type="D-T",
    )

    tau_E = p.compute_tau_E()

    T_keV = _kelvin_to_kev(p.temperature)
    n_tau_T = p.density_number * T_keV * tau_E

    threshold = LAWSON_THRESHOLDS["D-T"]

    assert n_tau_T < threshold, (
        f"n*T*tau_E = {n_tau_T:.2e} >= {threshold:.2e}"
    )

    ignited = p.check_lawson()
    assert ignited is False, (
        f"check_lawson() = {ignited}, expected False"
    )


@pytest.mark.parametrize("fuel,threshold", [
    ("D-T", 3.0e21),
    ("D-D", 1.0e24),
    ("D-He3", 1.0e22),
])
def test_lawson_uses_correct_threshold(fuel, threshold):
    """Порог Лоусона зависит от типа топлива.

    v37.2: точка усилена так же, как в test_lawson_dt_above_threshold —
    запас подобран, чтобы D-T зажигался, а D-He3 (порог 1e22) ещё нет.
    """
    p = _make_plasma(
        density_number=2.5e20,  # v37.2: усиленная точка (запас к порогу)
        temperature=1.5e8,      # ~13 кэВ
        external_B=13.0,        # сильное поле
        radius=1.0,             # ITER-масштаб
        fuel_type=fuel,
    )

    tau_E = p.compute_tau_E()

    T_keV = _kelvin_to_kev(p.temperature)
    n_tau_T = p.density_number * T_keV * tau_E

    expected_threshold = LAWSON_THRESHOLDS[fuel]
    assert expected_threshold == threshold

    # D-T зажигается; D-D и D-He3 — нет (порог выше)
    if fuel == "D-T":
        assert p.check_lawson() is True
    else:
        assert p.check_lawson() is False

