"""
Энергобаланс плазмы: P_fusion vs P_rad + P_cond.

dE/dt = P_fusion - P_radiation - P_conduction
"""

import numpy as np
import pytest
from physical_modules.plasma_monitor import PlasmaMonitor


def _kelvin_to_kev(T_K):
    k_B = 1.380649e-23
    eV = 1.602176634e-19
    return T_K * k_B / eV / 1e3


def _make_plasma(**kwargs):
    cfg = {
        "density_number": 1.0e20,
        "temperature": 1.16e8,
        "external_B": 2.0,
        "fuel_type": "D-T",
        "radius": 1.0e-2,
    }
    cfg.update(kwargs)
    p = PlasmaMonitor(cfg)
    p.init()
    return p


def test_positive_balance_heats_plasma():
    """ITER-масштаб: P_fus > P_rad + P_cond -> dE/dt > 0."""
    p = _make_plasma(
        density_number=2.0e20,
        temperature=1.5e8,      # ~13 кэВ (близко к пику <σv> для D-T)
        external_B=10.0,       # сильное поле -> большое tau_E -> малые потери
        radius=1.0,             # ITER-масштаб
        fuel_type="D-T",
    )

    balance = p.compute_energy_balance()

    assert balance["dE_dt"] > 0, (
        f"Ожидался положительный баланс, получили dE/dt = {balance['dE_dt']:.3e} Вт/м³\n"
        f"  P_fusion = {balance['P_fusion']:.3e}\n"
        f"  P_rad    = {balance['P_radiation']:.3e}\n"
        f"  P_cond   = {balance['P_conduction']:.3e}"
    )


def test_negative_balance_cools_plasma():
    """Низкие n, T, B: P_fus < P_rad + P_cond -> dE/dt < 0."""
    p = _make_plasma(
        density_number=1.0e18,
        temperature=4.6e6,      # ~0.4 кэВ
        external_B=0.1,
        radius=1.0e-2,
        fuel_type="D-T",
    )

    balance = p.compute_energy_balance()

    assert balance["dE_dt"] < 0, (
        f"Ожидался отрицательный баланс, получили dE/dt = {balance['dE_dt']:.3e} Вт/м³"
    )


def test_steady_state_balance():
    """Аддитивность: все компоненты > 0, dE/dt = P_fus - P_rad - P_cond."""
    p = _make_plasma(
        density_number=5.0e19,
        temperature=1.0e8,      # ~8.6 кэВ
        external_B=3.0,
        radius=0.5,
        fuel_type="D-T",
    )

    balance = p.compute_energy_balance()

    # Все компоненты должны быть положительными числами
    assert balance["P_fusion"] >= 0, "P_fusion должно быть неотрицательным"
    assert balance["P_rad"] >= 0, "P_rad должно быть неотрицательным"
    assert balance["P_cond"] >= 0, "P_cond должно быть неотрицательным"

    # Проверка аддитивности
    expected_dE = balance["P_fusion"] - balance["P_rad"] - balance["P_cond"]
    assert np.isclose(balance["dE_dt"], expected_dE, rtol=1e-10), (
        f"dE_dt = {balance['dE_dt']:.3e}, "
        f"P_fus - P_rad - P_cond = {expected_dE:.3e}"
    )

    # dE/dt — конечное число
    assert np.isfinite(balance["dE_dt"]), "dE_dt должно быть конечным"
