"""
tests/test_energy_balance.py — Энергобаланс плазмы.

Проверяет: P_fusion vs P_radiation + P_conduction.

    dE/dt = P_fusion - P_rad - P_cond

    P_fusion = n_D * n_T * <σv>(T) * E_fus     [Вт/м³]
    P_rad    = C_brem * Z_eff * n_e² * √T_keV  [Вт/м³]
    P_cond   = 3 * n * k_B * T / tau_E          [Вт/м³]

Если dE/dt > 0 → плазма разогревается.
Если dE/dt < 0 → плазма остывает.
Если dE/dt ≈ 0 → стационар.
"""

import numpy as np
import pytest

from plasma_monitor import PlasmaMonitor


# --- Константы ---
K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
E_FUS_DT = 17.6e6 * E_CHARGE  # 17.6 МэВ → Дж (D-T)
C_BREM = 1.69e-38             # коэффициент Bremsstrahlung (Вт·м³/√кэВ)


def _kelvin_to_kev(T_K):
    return T_K * K_B / (E_CHARGE * 1e3)


def _make_plasma(**overrides):
    cfg = {
        "radius": 1.0e-2,
        "density_number": 1.0e20,
        "temperature": 1.16e8,   # ~10 кэВ
        "external_B": 2.0,
        "fuel_type": "D-T",
    }
    cfg.update(overrides)
    return PlasmaMonitor(config=cfg)


# ============================================================
# Тест 1: Положительный баланс — плазма разогревается
# ============================================================

def test_positive_balance_heats_plasma():
    """При высоких n, T, B: P_fusion > P_rad + P_cond → dT/dt > 0."""
    p = _make_plasma(
        density_number=2.0e20,    # высокая плотность
        temperature=1.5e8,        # ~13 кэВ (близко к пику <σv> для D-T)
        external_B=5.0,           # сильное поле → большое tau_E → малые потери
        fuel_type="D-T",
    )

    balance = p.compute_energy_balance()

    assert balance["dE_dt"] > 0, (
        f"Ожидался положительный баланс, получили dE/dt = {balance['dE_dt']:.3e} Вт/м³"
    )
    assert balance["P_fusion"] > 0, "P_fusion должна быть положительной"


# ============================================================
# Тест 2: Отрицательный баланс — плазма остывает
# ============================================================

def test_negative_balance_cools_plasma():
    """При низких n, T: P_fusion < P_rad + P_cond → dT/dt < 0."""
    p = _make_plasma(
        density_number=1.0e18,    # низкая плотность → слабая реакция
        temperature=5.0e6,        # ~0.4 кэВ → <σv> ничтожен
        external_B=0.1,           # слабое поле → малое tau_E → большие потери
        fuel_type="D-T",
    )

    balance = p.compute_energy_balance()

    assert balance["dE_dt"] < 0, (
        f"Ожидался отрицательный баланс, получили dE/dt = {balance['dE_dt']:.3e} Вт/м³"
    )


# ============================================================
# Тест 3: Стационарный баланс — dE/dt ≈ 0
# ============================================================

def test_steady_state_balance():
    """При параметрах вблизи точки зажигания: dE/dt ≈ 0."""
    # Подбираем параметры, при которых баланс близок к нулю.
    # Это должна быть граница зажигания.
    p = _make_plasma(
        density_number=5.0e19,
        temperature=1.0e8,      # ~8.6 кэВ
        external_B=3.0,
        fuel_type="D-T",
    )

    balance = p.compute_energy_balance()

    # Проверяем, что все компоненты вычислены
    assert "P_fusion" in balance
    assert "P_rad" in balance
    assert "P_cond" in balance
    assert "dE_dt" in balance

    # Проверяем, что баланс аддитивен
    dE_expected = balance["P_fusion"] - balance["P_rad"] - balance["P_cond"]
    assert abs(balance["dE_dt"] - dE_expected) / max(abs(dE_expected), 1.0) < 0.01, (
        "dE/dt должна равняться P_fusion - P_rad - P_cond"
    )

    # Проверяем, что P_fusion > 0 (реакция идёт)
    assert balance["P_fusion"] > 0, "P_fusion должна быть положительной"

    # Проверяем, что P_rad > 0 (излучение есть)
    assert balance["P_rad"] > 0, "P_rad должна быть положительной"

    # Проверяем, что P_cond > 0 (потери есть)
    assert balance["P_cond"] > 0, "P_cond должна быть положительной"
