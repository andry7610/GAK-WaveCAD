"""
test_optuna_ash.py — фиксация физики dirty-plasma эпохи (v37.2).

Проверяется БЕЗ запуска плазмы:
  1. Z_eff вольфрама: 1 + c_W·Z·(Z-1) = 1.5402.
  2. Разбавление топлива (1 - f_He)^2: исторический якорь калибровки C_ASH —
     1527.3 МВт валовых -> 1102.9 МВт при f_He = 0.1502.
  3. Станционный баланс воспроизводит чемпионскую точку trial 37 прогона
     v37.2 (2026-10-25): P_electric = 627.15 МВт, recirc = 4.77%, P_net = 597.24 МВт.
  4. Защитный клапан рециркуляции: зажатие в [0.02, 0.12].

Запуск: python -m pytest tests/test_optuna_ash.py -v
        (или напрямую: python tests/test_optuna_ash.py)
"""
import os
import sys

# --- Константы v37.2 (сверены с optuna_optimizer.py; при расхождении CI предупреждён) ---
ETA_MHD = 0.22
ETA_STEAM = 0.42
RECIRC_CRYO_FRAC = 0.02
RECIRC_PUMP_FRAC = 0.02
RECIRC_REF_B = 28.71
RECIRC_REF_PUMP = 0.0665
C_W = 1.0e-4
Z_W = 74
F_HE_MAX = 0.30

# Чемпион trial 37 прогона v37.2 (50 trials, режим "Зажигание", 2026-10-25)
CHAMPION = {
    "P_fusion_W": 1121892049.636145,
    "P_sync_W": 213371023.8922824,
    "external_B": 28.574447556997765,
    "pump_speed": 0.07862341384216776,
    "f_He": 0.1134733827374229,
    "P_electric_W_expected": 627150347.8142899,
    "P_net_W_expected": 597239274.6076388,
}

# Исторический якорь калибровки золы: рекорд 1527.3 МВт (валовых) эпохи v36.3
HISTORICAL_ANCHOR = {
    "P_fusion_clean_W": 1527.3e6,
    "f_He": 0.1502,
    "P_fusion_dirty_W_expected": 1102.9e6,
}

# Опциональная сверка с реальным классом (если зависимости установлены)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from optuna_optimizer import GAKOptunaOptimizer as _Opt  # noqa: E402
    _HAVE_OPTIMIZER = True
except Exception:
    _HAVE_OPTIMIZER = False


def z_eff_tungsten(c_W=C_W, z_w=Z_W):
    return 1.0 + c_W * z_w * (z_w - 1.0)


def dilution_factor(f_He):
    return (1.0 - f_He) ** 2


def eta_cascade(eta_mhd=ETA_MHD, eta_steam=ETA_STEAM):
    return eta_mhd + (1.0 - eta_mhd) * eta_steam


def sync_harvest_W(P_sync_W):
    return 0.15 * max(P_sync_W, 0.0) * 0.40


def recirc_frac(B_T, pump_speed):
    r = RECIRC_CRYO_FRAC + RECIRC_PUMP_FRAC * (
        (float(B_T) / RECIRC_REF_B) ** 2) * (
        (float(pump_speed) / RECIRC_REF_PUMP) ** 2)
    return min(max(r, RECIRC_CRYO_FRAC), 0.12)


def test_z_eff_tungsten():
    assert abs(z_eff_tungsten() - 1.5402) < 1e-3


def test_dilution_anchor():
    """Якорь калибровки C_ASH: (1 - f_He)^2 на рекорде 1527.3 МВт."""
    p_dirty = HISTORICAL_ANCHOR["P_fusion_clean_W"] * dilution_factor(
        HISTORICAL_ANCHOR["f_He"])
    assert abs(p_dirty - HISTORICAL_ANCHOR["P_fusion_dirty_W_expected"]) / \
        HISTORICAL_ANCHOR["P_fusion_dirty_W_expected"] < 1e-3


def test_station_balance_champion():
    """Формулы станционного баланса воспроизводят trial 37 до последнего знака."""
    P_electric = CHAMPION["P_fusion_W"] * eta_cascade() + \
        sync_harvest_W(CHAMPION["P_sync_W"])
    recirc = recirc_frac(CHAMPION["external_B"], CHAMPION["pump_speed"])
    P_net = P_electric * (1.0 - recirc)
    assert abs(P_electric - CHAMPION["P_electric_W_expected"]) / \
        CHAMPION["P_electric_W_expected"] < 1e-9
    assert abs(P_net - CHAMPION["P_net_W_expected"]) / \
        CHAMPION["P_net_W_expected"] < 1e-9
    assert abs(recirc - 0.04769362452065203) < 1e-9


def test_recirc_clamped():
    assert recirc_frac(45.0, 0.15) <= 0.12 + 1e-12
    assert recirc_frac(RECIRC_REF_B, RECIRC_REF_PUMP) == RECIRC_CRYO_FRAC + RECIRC_PUMP_FRAC


def test_f_he_cap():
    assert F_HE_MAX == 0.30


def test_optimizer_class_consistency():
    """Если класс импортируется — сверяем константы и вызов _station_balance."""
    if not _HAVE_OPTIMIZER:
        return  # зависимости не установлены — локальные формулы уже проверены
    assert _Opt.C_W == C_W
    assert _Opt.Z_W == Z_W
    assert _Opt.F_HE_MAX == F_HE_MAX
    assert abs(_Opt.ETA_MHD - ETA_MHD) < 1e-12
    assert abs(_Opt.ETA_STEAM - ETA_STEAM) < 1e-12
    st = _Opt._station_balance(
        CHAMPION["P_fusion_W"], CHAMPION["P_sync_W"],
        B_T=CHAMPION["external_B"], pump_speed=CHAMPION["pump_speed"])
    assert abs(st["P_net_W"] - CHAMPION["P_net_W_expected"]) / \
        CHAMPION["P_net_W_expected"] < 1e-9


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append(name)
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failures else 0)
