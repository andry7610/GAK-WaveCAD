"""
core/session_state.py — Единый session_state для Streamlit multipage.

Все общие объекты хранятся здесь, чтобы страницы не пересоздавали их.
"""

import streamlit as st


def init_session_state():
    """Инициализация session_state при первом запуске."""
    defaults = {
        # ── Конфигурация ──
        "base_config": None,
        "script_dir": None,
        
        # ── Параметры симуляции (из сайдбара) ──
        "R_mm": 1000.0,
        "h_mm": 2.0,
        "T_inner": 310.0,
        "T_outer": 293.0,
        "k_thermal": 15.0,
        "pl_radius_mm": 50.0,
        "T_plasma_init": 3e8,
        "n_density": 1e20,
        "fuel_type": "D-T",
        "P_initial": 1e3,
        "pump_speed": 1e-2,
        "E_field": 3e17,
        "gap_nm": 3.5,
        "external_B": 30.0,
        "n_steps": 2500000,
        "sim_dt": 1e-5,
        
        # ── Результаты ──
        "last_result": None,
        "last_history": None,
        "last_params": None,
        
        # ── AI ──
        "ai_predictor": None,
        "ai_loaded": False,
        "ai_model_path": None,
        
        # ── Optuna ──
        "optimizer": None,
        "optuna_study": None,
        "optuna_results": None,
        "bounds": None,
        
        # ── UI ──
        "current_page": "main",
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_param(key, default=None):
    """Получить параметр из session_state."""
    return st.session_state.get(key, default)


def set_param(key, value):
    """Установить параметр в session_state."""
    st.session_state[key] = value


def get_all_params():
    """Получить все параметры симуляции как dict."""
    param_keys = [
        "R_mm", "h_mm", "T_inner", "T_outer", "k_thermal",
        "pl_radius_mm", "T_plasma_init", "n_density", "fuel_type",
        "P_initial", "pump_speed", "E_field", "gap_nm", "external_B",
        "n_steps", "sim_dt",
    ]
    return {k: st.session_state.get(k) for k in param_keys}


def set_ai_predictor(predictor, model_path=None):
    """Сохранить AI-предиктор в session_state."""
    st.session_state["ai_predictor"] = predictor
    st.session_state["ai_loaded"] = predictor is not None and predictor.is_trained
    st.session_state["ai_model_path"] = model_path


def get_ai_predictor():
    """Получить AI-предиктор из session_state."""
    return st.session_state.get("ai_predictor")
