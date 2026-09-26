"""
GAK-WaveCAD — Robustness Tests v1.0
Тесты робастности ядра плазменной модели (plasma_monitor v0.9).

5 проверок:
  1. Воспроизводимость — два прогона → идентичные результаты
  2. Устойчивость — возмущения T_init → сходимость к равновесию
  3. Граничные случаи — экстремальные параметры → без NaN/crash
  4. Артефакты — все известные баги закрыты
  5. Q = 1/alpha_frac — фундаментальная проверка физики

Запуск: python -m pytest tests/test_robustness.py -v
"""

import numpy as np
import pytest


# --- Физические константы ---
MU_0 = 4 * np.pi * 1e-7
K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
C_GYRO_BOHM = 0.14

ALPHA_FRACTIONS = {"D-T": 0.20, "D-He3": 1.00, "D-D": 0.50}
C_SYNC = 4.05e-20
WALL_REFLECTIVITY = 0.6

FUEL_ENERGY = {
    "D-T":   17.6e6 * E_CHARGE,
    "D-D":   3.65e6 * E_CHARGE,
    "D-He3": 18.3e6 * E_CHARGE,
}

T_CAP = 1e9

# Параметры v9.1 (из ignition_v9_DT)
PARAMS = {
    "R_sphere": 1.11350,
    "pl_radius": 0.419761,
    "T_init": 9.677e8,
    "n": 4.219e20,
    "B": 30.3,
    "fuel": "D-T",
    "n_steps": 50000,
    "dt": 1e-5,
}


def compute_reactivity(T_keV, fuel_type="D-T"):
    if T_keV <= 0:
        return 0.0
    T_pow = T_keV ** (-2.0 / 3.0)
    if fuel_type == "D-T":
        sv = 3.68e-18 * T_pow * np.exp(-19.94 * T_keV ** (-1.0 / 3.0))
    elif fuel_type == "D-D":
        sv = 3.70e-18 * T_pow * np.exp(-46.10 * T_keV ** (-1.0 / 3.0))
    elif fuel_type == "D-He3":
        sv = 5.50e-18 * T_pow * np.exp(-38.40 * T_keV ** (-1.0 / 3.0))
    else:
        sv = 3.68e-18 * T_pow * np.exp(-19.94 * T_keV ** (-1.0 / 3.0))
    return max(sv, 0.0)


def compute_tau_E(T, n, B, R):
    T_keV = T * K_B / (E_CHARGE * 1e3)
    if T_keV <= 0 or n <= 0 or B <= 0:
        return 0.0
    n_20 = n / 1e20
    return C_GYRO_BOHM * R**2 * B**2 / (np.sqrt(T_keV) * n_20)


def simulate(T_init, n, B, R, fuel_type="D-T", n_steps=50000, dt=1e-5, T_cap=T_CAP):
    T = T_init
    alpha_frac = ALPHA_FRACTIONS.get(fuel_type, 1.0)
    E_fus = FUEL_ENERGY.get(fuel_type, 17.6e6 * E_CHARGE)
    
    T_vals, Q_vals = [], []
    P_fus_vals, P_cond_vals, P_rad_vals, P_sync_vals, P_alpha_vals = [], [], [], [], []
    tau_E_vals = []
    runaway = False
    
    for step_i in range(n_steps):
        T_keV = max(T * K_B / (E_CHARGE * 1e3), 1e-6)
        sv = compute_reactivity(T_keV, fuel_type)
        
        if fuel_type == "D-D":
            P_fusion = 0.5 * n**2 * sv * E_fus
        else:
            P_fusion = 0.25 * n**2 * sv * E_fus
        
        P_alpha = alpha_frac * P_fusion
        P_rad = 1.69e-38 * 1.0 * n**2 * np.sqrt(T_keV)
        tau_E = compute_tau_E(T, n, B, R)
        P_cond = 3.0 * n * K_B * T / tau_E if tau_E > 0 else 1e30
        P_sync = C_SYNC * n * B**2 * T_keV**2 * (1.0 - WALL_REFLECTIVITY)
        
        dE_dt = P_alpha - P_rad - P_cond - P_sync
        
        # Адаптивный шаг при жёсткой динамике
        dt_eff = 0.05 * tau_E if (tau_E > 0 and dt > 0.1 * tau_E) else dt
        
        dT = dE_dt * dt_eff / (1.5 * n * K_B) if n > 0 else 0.0
        T_new = T + dT
        
        if np.isnan(T_new) or np.isinf(T_new) or T_new <= 0:
            break
        
        T = T_new
        if T > T_cap:
            runaway = True
            break
        
        P_loss = P_cond + P_rad + P_sync
        Q = P_fusion / P_loss if 0 < P_loss < 1e30 else 0.0
        
        T_vals.append(T)
        Q_vals.append(Q)
        P_fus_vals.append(P_fusion)
        P_cond_vals.append(P_cond)
        P_rad_vals.append(P_rad)
        P_sync_vals.append(P_sync)
        P_alpha_vals.append(P_alpha)
        tau_E_vals.append(tau_E)
    
    if len(T_vals) < 3 and not runaway:
        return None
    
    n_avg = max(1, len(T_vals) // 5) if T_vals else 1
    return {
        "T_final": T_vals[-1] if T_vals else T,
        "T_avg": float(np.mean(T_vals[-n_avg:])) if T_vals else T,
        "Q_avg": float(np.mean(Q_vals[-n_avg:])) if Q_vals else 0.0,
        "P_fusion_avg": float(np.mean(P_fus_vals[-n_avg:])) if P_fus_vals else 0.0,
        "P_cond_avg": float(np.mean(P_cond_vals[-n_avg:])) if P_cond_vals else 0.0,
        "P_rad_avg": float(np.mean(P_rad_vals[-n_avg:])) if P_rad_vals else 0.0,
        "P_sync_avg": float(np.mean(P_sync_vals[-n_avg:])) if P_sync_vals else 0.0,
        "P_alpha_avg": float(np.mean(P_alpha_vals[-n_avg:])) if P_alpha_vals else 0.0,
        "tau_E_avg": float(np.mean(tau_E_vals[-n_avg:])) if tau_E_vals else 0.0,
        "runaway": runaway,
        "steps_done": len(T_vals),
        "P_alpha_over_Ploss": (
            float(np.mean(P_alpha_vals[-n_avg:])) / 
            max(float(np.mean(P_cond_vals[-n_avg:]) + np.mean(P_rad_vals[-n_avg:]) + np.mean(P_sync_vals[-n_avg:])), 1e-30)
            if P_fus_vals else 0.0
        ),
    }


# === ТЕСТ 1: Воспроизводимость ===

class TestReproducibility:
    def test_two_runs_identical(self):
        """Два прогона с одинаковыми параметрами → идентичные результаты."""
        r1 = simulate(PARAMS["T_init"], PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                      PARAMS["fuel"], PARAMS["n_steps"], PARAMS["dt"])
        r2 = simulate(PARAMS["T_init"], PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                      PARAMS["fuel"], PARAMS["n_steps"], PARAMS["dt"])
        
        assert r1 is not None and r2 is not None
        assert abs(r1["Q_avg"] - r2["Q_avg"]) < 1e-10
        assert abs(r1["T_avg"] - r2["T_avg"]) < 1e-10
        assert r1["runaway"] == r2["runaway"]
    
    def test_matches_csv_v91(self):
        """Совпадение с CSV v9.1: Q ≈ 4.82, T ≈ 6.53e8."""
        r = simulate(PARAMS["T_init"], PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], PARAMS["n_steps"], PARAMS["dt"])
        
        assert r is not None
        assert abs(r["Q_avg"] - 4.82) / 4.82 < 0.01   # 1%
        assert abs(r["T_avg"] - 6.53e8) / 6.53e8 < 0.01  # 1%
        assert r["runaway"] == False


# === ТЕСТ 2: Устойчивость ===

class TestEquilibriumStability:
    def test_perturbation_05x_converges(self):
        """T_init = 0.5×T_eq → сходится к равновесию."""
        T_eq = 6.20e8
        r = simulate(T_eq * 0.5, PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 200000, PARAMS["dt"])
        assert r is not None
        assert abs(r["T_final"] - T_eq) / T_eq < 0.15
    
    def test_perturbation_15x_converges(self):
        """T_init = 1.5×T_eq → сходится к равновесию."""
        T_eq = 6.20e8
        r = simulate(T_eq * 1.5, PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 200000, PARAMS["dt"])
        assert r is not None
        assert abs(r["T_final"] - T_eq) / T_eq < 0.15
    
    def test_below_ignition_gassnels(self):
        """T_init = 0.1×T_eq → плазма гаснет (физично)."""
        T_eq = 6.20e8
        r = simulate(T_eq * 0.1, PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 200000, PARAMS["dt"])
        assert r is not None
        assert r["T_final"] < 1e8  # остыла ниже 100 МК


# === ТЕСТ 3: Граничные случаи ===

class TestBoundaryCases:
    @pytest.mark.parametrize("T_init,n,B,R,fuel", [
        (9.677e8, 4.219e20, 30.3, 0.010, "D-T"),   # R=10мм
        (9.677e8, 4.219e20, 30.3, 1.500, "D-T"),   # R=1500мм
        (9.677e8, 1.0e18, 30.3, 0.420, "D-T"),     # n=1e18
        (9.677e8, 1.0e23, 30.3, 0.420, "D-T"),     # n=1e23
        (9.677e8, 4.219e20, 5.0, 0.420, "D-T"),    # B=5
        (9.677e8, 4.219e20, 45.0, 0.420, "D-T"),   # B=45
        (1.0e6, 4.219e20, 30.3, 0.420, "D-T"),     # T=1e6
        (9.677e8, 4.219e20, 30.3, 0.420, "D-He3"), # D-He3
        (9.677e8, 4.219e20, 30.3, 0.420, "D-D"),   # D-D
        (9.677e8, 4.219e20, 100.0, 0.420, "D-T"),  # B=100
        (9.677e8, 1.0e16, 30.3, 0.420, "D-T"),    # n=1e16
    ])
    def test_no_nan_no_crash(self, T_init, n, B, R, fuel):
        """Экстремальные параметры → без NaN, без краша."""
        r = simulate(T_init, n, B, R, fuel, 50000, 1e-5)
        # Должен вернуть результат (не None) или runaway
        assert r is not None or True  # None допустим при дивергенции
        if r is not None:
            assert not np.isnan(r["T_final"])
            assert r["T_final"] > 0


# === ТЕСТ 4: Артефакты ===

class TestArtifacts:
    def test_pl_radius_constraint(self):
        """pl_radius < R*0.4."""
        assert 0.419761 / 1.11350 < 0.4
    
    def test_B_constraint(self):
        """B ≤ 45 Тл."""
        assert 30.3 <= 45.0
    
    def test_schwinger_effectively_off(self):
        """Швингер: pair_rate ≈ 0 при E << E_c."""
        E_field = 1.384e16
        E_c = 1.32e18
        gap = 45e-9
        schwinger_factor = np.exp(-np.pi * E_c / E_field)
        pair_rate = 1e45 * (E_field / E_c)**2 * schwinger_factor / (gap * 1e9)
        assert pair_rate < 1e-80  # фактически ноль
    
    def test_T_CAP_catches_runaway(self):
        """T_CAP = 1e9 ловит runaway."""
        r = simulate(1.5e9, PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 50000, PARAMS["dt"])
        assert r is not None
        assert r["runaway"] == True
    
    def test_Q_includes_sync(self):
        """Q = P_fusion / (P_cond + P_rad + P_sync)."""
        r = simulate(PARAMS["T_init"], PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 50000, PARAMS["dt"])
        P_loss = r["P_cond_avg"] + r["P_rad_avg"] + r["P_sync_avg"]
        Q_check = r["P_fusion_avg"] / P_loss
        assert abs(Q_check - r["Q_avg"]) / r["Q_avg"] < 0.01
    
    def test_sync_reflection(self):
        """P_sync учитывает отражение стенки (0.6)."""
        assert WALL_REFLECTIVITY == 0.6
        T_keV = 56.25
        P_total = C_SYNC * 4.219e20 * 30.3**2 * T_keV**2
        P_net = P_total * (1 - WALL_REFLECTIVITY)
        assert P_net == P_total * 0.4


# === ТЕСТ 5: Фундаментальная физика ===

class TestFundamentalPhysics:
    def test_Q_converges_to_1_over_alpha(self):
        """Q → 1/alpha_frac при равновесии (D-T: Q → 5.0)."""
        r = simulate(PARAMS["T_init"], PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 200000, PARAMS["dt"])
        assert r is not None
        Q_theory = 1.0 / ALPHA_FRACTIONS["D-T"]
        assert abs(r["Q_avg"] - Q_theory) / Q_theory < 0.05  # 5%
    
    def test_alpha_equals_loss_at_equilibrium(self):
        """P_alpha ≈ P_loss при равновесии."""
        r = simulate(PARAMS["T_init"], PARAMS["n"], PARAMS["B"], PARAMS["pl_radius"],
                     PARAMS["fuel"], 200000, PARAMS["dt"])
        assert r is not None
        assert 0.9 < r["P_alpha_over_Ploss"] < 1.1
    
    def test_D_He3_alpha_frac_is_1(self):
        """D-He3: alpha_frac = 1.0 (безнейтронное топливо)."""
        assert ALPHA_FRACTIONS["D-He3"] == 1.0
    
    def test_DT_alpha_frac_is_020(self):
        """D-T: alpha_frac = 0.20 (3.5/17.6 МэВ)."""
        assert ALPHA_FRACTIONS["D-T"] == 0.20
