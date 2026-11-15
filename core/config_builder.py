"""
core/config_builder.py — Единая сборка конфигурации для GAK-WaveCAD.

Объединяет логику из:
- optuna_optimizer.py::_build_trial_config
- ai_validation.py::_build_val_config

Используется везде: симуляция, оптимизация, валидация, DOE.
"""

import numpy as np


def build_config(base_config, *, R_mm=1000, h_mm=2.0,
                 T_inner=310.0, T_outer=293.0, k_thermal=15.0,
                 pl_radius_mm=50.0, T_plasma=3e8, n_density=1e20,
                 fuel_type="D-T", P_initial=1e3, pump_speed=1e-2,
                 E_field=3e17, gap_nm=3.5, external_B=30.0,
                 n_steps=2500000, dt=1e-5,
                 auto_feed=False, quantum_feedback=True):
    """
    Сборка полного конфига из параметров.
    
    Параметры:
        base_config — базовый конфиг из config.yaml
        R_mm — радиус сферы, мм
        h_mm — толщина стенки, мм
        T_inner / T_outer — температуры стенки, K
        k_thermal — теплопроводность
        pl_radius_mm — радиус плазмы, мм
        T_plasma — начальная температура плазмы, K
        n_density — плотность, м^-3
        fuel_type — "D-T", "D-D", "D-He3"
        P_initial — стартовое давление, Па
        pump_speed — скорость откачки, м^3/с
        E_field — E-поле, В/м
        gap_nm — зазор, нм
        external_B — внешнее магнитное поле, Тл
        n_steps — шагов симуляции
        dt — шаг по времени, с
    
    Возвращает:
        cfg — полный конфиг для PlasmaMonitor и модулей
    """
    cfg = dict(base_config)
    
    R_m = R_mm / 1000
    h_m = h_mm / 1000
    pl_radius_m = pl_radius_mm / 1000
    
    # Тепловые и структурные модули — общий R и h
    for mod_key in ("thermal_monitor", "acoustic_monitor",
                    "born_collapse_monitor", "magnon_monitor",
                    "em_resonance_monitor"):
        mod = dict(base_config.get(mod_key, {}))
        mod["R"] = R_m
        mod["h"] = h_m
        cfg[mod_key] = mod
    
    # Тепловые параметры
    cfg["thermal_monitor"]["T_inner"] = T_inner
    cfg["thermal_monitor"]["T_outer"] = T_outer
    cfg["thermal_monitor"]["k_thermal"] = k_thermal
    
    # Плазма
    cfg["plasma_monitor"] = {
        **base_config.get("plasma_monitor", {}),
        "radius": pl_radius_m,
        "temperature": T_plasma,
        "density_number": n_density,
        "fuel_type": fuel_type,
        "auto_feed": auto_feed,
        "quantum_feedback": quantum_feedback,
        "external_B": external_B,
    }
    
    # Вакуум
    cfg["vacuum_monitor"] = {
        **base_config.get("vacuum_monitor", {}),
        "P_initial": P_initial,
        "pump_speed": pump_speed,
    }
    
    # Квантовый вакуум (Швингер)
    cfg["quantum_vacuum"] = {
        **base_config.get("quantum_vacuum", {}),
        "E_field": E_field,
        "gap": gap_nm * 1e-9,
    }
    
    # Симуляция
    cfg["simulation"] = {"n_steps": n_steps, "dt": dt}
    
    return cfg


def build_config_from_optuna_trial(base_config, trial, bounds, fixed_fuel_idx=1, T_CAP=1e9):
    """
    Сборка конфига из Optuna trial.
    
    Параметры:
        base_config — базовый конфиг
        trial — optuna.Trial
        bounds — dict с диапазонами параметров
        fixed_fuel_idx — индекс топлива (0=D-He3, 1=D-T, 2=D-D)
        T_CAP — потолок температуры
    
    Возвращает:
        cfg — полный конфиг
    """
    fuel_types = ["D-He3", "D-T", "D-D"]
    
    R_mm = trial.suggest_float("R_mm", *bounds["R_mm"])
    h_mm = trial.suggest_float("h_mm", *bounds["h_mm"])
    T_inner = trial.suggest_float("T_inner", *bounds["T_inner"])
    T_outer = trial.suggest_float("T_outer", *bounds["T_outer"])
    k_thermal = trial.suggest_float("k_thermal", *bounds["k_thermal"])
    
    # Constraint: плазма внутри сферы
    pl_radius_max = R_mm * 0.5
    pl_radius_mm = trial.suggest_float("pl_radius_mm",
                                        max(R_mm * 0.25, 10),
                                        pl_radius_max)
    
    T_plasma = trial.suggest_float("T_plasma_init", *bounds["T_plasma_init"], log=True)
    n_density = trial.suggest_float("n_density", *bounds["n_density"], log=True)
    fuel_type = fuel_types[fixed_fuel_idx]
    
    P_initial = trial.suggest_float("P_initial", *bounds["P_initial"], log=True)
    pump_speed = trial.suggest_float("pump_speed", *bounds["pump_speed"], log=True)
    E_field = trial.suggest_float("E_field", *bounds["E_field"], log=True)
    gap_nm = trial.suggest_int("gap_nm", *bounds["gap_nm"])
    external_B = trial.suggest_float("external_B", *bounds["external_B"])
    
    return build_config(
        base_config,
        R_mm=R_mm, h_mm=h_mm,
        T_inner=T_inner, T_outer=T_outer, k_thermal=k_thermal,
        pl_radius_mm=pl_radius_mm, T_plasma=T_plasma,
        n_density=n_density, fuel_type=fuel_type,
        P_initial=P_initial, pump_speed=pump_speed,
        E_field=E_field, gap_nm=gap_nm, external_B=external_B,
        n_steps=2500000, dt=1e-5,
    )
