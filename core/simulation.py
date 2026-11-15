"""
core/simulation.py — Единая функция симуляции для GAK-WaveCAD.

Объединяет логику из:
- optuna_optimizer.py::_run_quick (полный прогон с метриками)
- ai_validation.py::_run_val_sim (быстрый прогон для валидации)

Два режима:
- full: полный прогон с Q-фактором, Lawson, всеми метриками
- quick: быстрый прогон для валидации/DOE (T, P_gas, pair_rate, ratio)
"""

import numpy as np


def run_simulation(cfg, mode="full", T_CAP=1e9, return_history=False, history_stride=2):
    """
    Единая функция симуляции.
    
    Параметры:
        cfg — полный конфиг (из config_builder.build_config)
        mode — "full" (Optuna) или "quick" (валидация/DOE)
        T_CAP — потолок температуры, K
        return_history — возвращать ли историю (для графиков)
        history_stride — каждые N шагов сохранять точку истории
    
    Возвращает:
        full: dict с метриками (Q, Lawson, P_fusion, tau_E, ...) или None при runaway
        quick: dict с history (T_plasma, P_gas, pair_rate, ratio)
    """
    from physical_modules.plasma_monitor import PlasmaMonitor
    
    sim_n = cfg["simulation"]["n_steps"]
    sim_dt = cfg["simulation"]["dt"]
    
    plasma = PlasmaMonitor(cfg.get("plasma_monitor", {}))
    plasma.init()
    
    V_plasma = (4.0 / 3.0) * np.pi * (plasma.radius ** 3)
    external_B = cfg["plasma_monitor"]["external_B"]
    fuel_type = cfg["plasma_monitor"]["fuel_type"]
    n_density = cfg["plasma_monitor"]["density_number"]
    
    # ── T_CAP pre-check ──
    T_init_check = cfg["plasma_monitor"]["temperature"]
    if T_init_check > T_CAP:
        return None
    
    # ── Вычислить pair_rate один раз ──
    qv_cfg = cfg.get("quantum_vacuum", {})
    E_field_qv = qv_cfg.get("E_field", 0.0)
    gap_qv = qv_cfg.get("gap", 1e-9)
    E_c = 1.32e18
    pair_rate = 0.0
    if E_field_qv > 0.001 * E_c:
        schwinger_factor = np.exp(-np.pi * E_c / max(E_field_qv, 1e-30))
        pair_rate = 1e45 * (E_field_qv / E_c) ** 2 * schwinger_factor / (gap_qv * 1e9)
        pair_rate = min(pair_rate, 1e10)
    
    # ── История (опционально) ──
    history = None
    if return_history or mode == "quick":
        history = {
            "step": [], "T_plasma": [], "P_fusion": [], "P_cond": [],
            "P_rad": [], "P_sync": [], "P_alpha": [], "tau_E": [],
            "Q": [], "ratio": [], "P_gas": [], "pair_rate": [],
            "sim_time": [],
        }
    
    # ── Накопители для full-режима ──
    ratios = []
    pair_rates = []
    T_vals = []
    P_fusion_vals = []
    P_cond_vals = []
    P_rad_vals = []
    P_sync_vals = []
    P_alpha_vals = []
    tau_E_vals = []
    Q_vals = []
    dt_actual_vals = []
    sim_time = 0.0
    
    for step_i in range(sim_n):
        # Adaptive dt
        tau_E_current = plasma.compute_tau_E()
        if tau_E_current > 0:
            dt_actual = min(sim_dt, 0.05 * tau_E_current)
        else:
            dt_actual = sim_dt
        
        step_result = plasma.step(dt_actual, pair_rate=pair_rate)
        sim_time += dt_actual
        
        T_plasma = step_result["temperature"]
        P_fusion = step_result["P_fusion"]
        P_cond = step_result["P_cond"]
        P_rad = step_result["P_rad"]
        tau_E = step_result["tau_E"]
        P_sync = step_result.get("P_sync", 0.0)
        P_alpha = step_result.get("P_alpha", 0.2 * P_fusion)
        alpha_frac = step_result.get("alpha_frac", 0.2)
        
        # Runaway check
        if T_plasma > T_CAP:
            return None
        
        # ── Швингер нагрев ──
        P_schwinger_heat = 0.0
        if pair_rate > 0 and T_plasma > 0:
            E_pair = 2 * 511e3 * 1.602e-19  # 2 × me × c²
            P_schwinger_heat = pair_rate * E_pair
        
        P_heat = P_fusion + P_schwinger_heat
        P_loss = P_cond + P_rad + P_sync
        ratio = P_heat / max(P_loss, 1e-30)
        
        # ── Q-фактор ──
        P_external = max(P_cond + P_rad + P_sync - P_alpha, 1e-30)
        Q = P_fusion / P_external
        
        # ── Газовое давление ──
        P_gas = max(0.0, n_density * 1.38e-23 * max(T_plasma, 1.0))
        
        # ── Накопление ──
        if mode == "full":
            ratios.append(ratio)
            pair_rates.append(pair_rate)
            T_vals.append(T_plasma)
            P_fusion_vals.append(P_fusion)
            P_cond_vals.append(P_cond)
            P_rad_vals.append(P_rad)
            P_sync_vals.append(P_sync)
            P_alpha_vals.append(P_alpha)
            tau_E_vals.append(tau_E)
            Q_vals.append(Q)
            dt_actual_vals.append(dt_actual)
        
        # ── История ──
        if (return_history or mode == "quick") and step_i % history_stride == 0:
            history["step"].append(step_i)
            history["T_plasma"].append(T_plasma)
            history["P_fusion"].append(P_fusion)
            history["P_cond"].append(P_cond)
            history["P_rad"].append(P_rad)
            history["P_sync"].append(P_sync)
            history["P_alpha"].append(P_alpha)
            history["tau_E"].append(tau_E)
            history["Q"].append(Q)
            history["ratio"].append(ratio)
            history["P_gas"].append(P_gas)
            history["pair_rate"].append(max(pair_rate, 0.0))
            history["sim_time"].append(sim_time)
    
    # ── Возврат ──
    if mode == "quick":
        return history
    
    # full: метрики из последних 20% шагов
    tail = max(1, len(Q_vals) // 5)
    
    # Lawson criterion
    T_keV = np.mean(T_vals[-tail:]) / 11600 / 1000  # K → keV
    n_tau = n_density * np.mean(tau_E_vals[-tail:])
    lawson = n_tau * T_keV * 1e-20  # нормировка
    
    # Пороги Lawson
    lawson_thresholds = {
        "D-He3": 1.0e22,
        "D-T": 3.0e21,
        "D-D": 1.0e24,
    }
    lawson_threshold = lawson_thresholds.get(fuel_type, 3e21)
    lawson_ratio = lawson / lawson_threshold if lawson_threshold > 0 else 0
    
    result = {
        "Q": float(np.mean(Q_vals[-tail:])),
        "Q_std": float(np.std(Q_vals[-tail:])),
        "Lawson": float(lawson),
        "Lawson_threshold": float(lawson_threshold),
        "Lawson_ratio": float(lawson_ratio),
        "T_plasma_eq": float(np.mean(T_vals[-tail:])),
        "T_keV": float(T_keV),
        "P_fusion": float(np.mean(P_fusion_vals[-tail:])),
        "P_alpha": float(np.mean(P_alpha_vals[-tail:])),
        "P_cond": float(np.mean(P_cond_vals[-tail:])),
        "P_rad": float(np.mean(P_rad_vals[-tail:])),
        "P_sync": float(np.mean(P_sync_vals[-tail:])),
        "tau_E": float(np.mean(tau_E_vals[-tail:])),
        "ratio": float(np.mean(ratios[-tail:])),
        "pair_rate": float(np.mean(pair_rates[-tail:])),
        "P_gas": float(np.mean([max(0.0, n_density * 1.38e-23 * max(t, 1.0)) for t in T_vals[-tail:]])),
        "runaway": False,
        "n_steps": sim_n,
        "sim_time": sim_time,
        "dt_actual_avg": float(np.mean(dt_actual_vals)),
    }
    
    if return_history:
        result["history"] = history
    
    return result
