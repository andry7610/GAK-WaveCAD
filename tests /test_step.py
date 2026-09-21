"""
tests/test_step.py — Временная динамика плазмы (step()).

Проверяет: step(dt) корректно продвигает состояние во времени.
"""

import numpy as np
import pytest

from plasma import PlasmaMonitor


def _make_plasma(**overrides):
    cfg = {
        "radius": 1.0e-2,
        "density_number": 1.0e20,
        "temperature": 1.16e8,
        "external_B": 2.0,
        "fuel_type": "D-T",
    }
    cfg.update(overrides)
    return PlasmaMonitor(config=cfg)


# ============================================================
# Тест 1: step() продвигает время и меняет температуру
# ============================================================

def test_step_advances_time():
    """step(dt) меняет T и возвращает новое состояние."""
    p = _make_plasma(
        density_number=2.0e20,
        temperature=1.5e8,
        external_B=5.0,
    )

    T0 = p.temperature
    dt = 1e-6  # 1 микросекунда

    result = p.step(dt)

    # Температура должна измениться
    assert p.temperature != T0, "T должна измениться после step()"

    # Результат должен содержать T и dE_dt
    assert "temperature" in result or "T" in result
    assert "dE_dt" in result

    # Если положительный баланс — T должна вырасти
    if result["dE_dt"] > 0:
        assert p.temperature > T0, "T должна расти при положительном балансе"
    else:
        assert p.temperature < T0, "T должна падать при отрицательном балансе"


# ============================================================
# Тест 2: 1000 шагов без NaN
# ============================================================

def test_step_no_nan():
    """За 1000 шагов нет NaN, T остаётся физичной."""
    p = _make_plasma(
        density_number=1.0e20,
        temperature=1.16e8,
        external_B=2.0,
    )

    dt = 1e-7  # 0.1 микросекунды

    for i in range(1000):
        p.step(dt)
        assert not np.isnan(p.temperature), f"NaN на шаге {i}"
        assert p.temperature > 0, f"T <= 0 на шаге {i}"
        # T не должна улетать в бесконечность
        assert p.temperature < 1e12, f"T взлетела до {p.temperature:.2e} на шаге {i}"
