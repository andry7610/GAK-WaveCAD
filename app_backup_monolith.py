"""
GAK-WaveCAD — Главный пульт управления.
Чистая страница: только параметры, кнопки, статус.
Графики — на отдельных страницах (pages/).
"""

import streamlit as st
import os
import sys

# ── Core модули ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.session_state import init_session_state, get_all_params, get_ai_predictor, set_ai_predictor
from core.config_builder import build_config
from core.simulation import run_simulation
from core.save_results import save_csv_results, generate_run_dir
from core.config_loader import ConfigLoader


def main():
    st.set_page_config(
        page_title="GAK-WaveCAD",
        page_icon="⚛️",
        layout="wide",
    )
    
    # ── Инициализация ──
    init_session_state()
    
    st.title("⚛️ GAK-WaveCAD — Пульт управления")
    st.markdown("Управление симуляцией плазмы. Графики — в сайдбаре слева.")
    
    # ── Загрузка конфига ──
    if st.session_state.base_config is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "configs", "config.yaml")
        try:
            st.session_state.base_config = ConfigLoader.load(config_path)
            st.session_state.script_dir = script_dir
        except Exception as e:
            st.error(f"Не удалось загрузить config.yaml: {e}")
            return
    
    # ── Параметры в сайдбаре ──
    with st.sidebar:
        st.markdown("### ⚙️ Параметры")
        
        st.session_state.R_mm = st.slider("Радиус сферы, мм", 100, 2000, int(st.session_state.R_mm), 10)
        st.session_state.h_mm = st.slider("Толщина стенки, мм", 0.5, 20.0, st.session_state.h_mm, 0.1)
        st.session_state.pl_radius_mm = st.slider("Радиус плазмы, мм", 5, 600, int(st.session_state.pl_radius_mm), 5)
        st.session_state.T_plasma_init = st.number_input("T плазмы (init), K", value=st.session_state.T_plasma_init, format="%.2e")
        st.session_state.n_density = st.number_input("Плотность, м^-3", value=st.session_state.n_density, format="%.2e")
        st.session_state.external_B = st.slider("B внешнее, Тл", 5.0, 45.0, st.session_state.external_B, 0.1)
        st.session_state.E_field = st.number_input("E-поле, В/м", value=st.session_state.E_field, format="%.2e")
        st.session_state.gap_nm = st.slider("Зазор, нм", 1, 50, int(st.session_state.gap_nm), 1)
        st.session_state.n_steps = st.slider("Шагов симуляции", 100000, 5000000, st.session_state.n_steps, 100000)
        
        fuel_options = ["D-T", "D-D", "D-He3"]
        st.session_state.fuel_type = st.selectbox("Топливо", fuel_options, 
                                                   index=fuel_options.index(st.session_state.fuel_type) if st.session_state.fuel_type in fuel_options else 0)
        
        # ── Пресеты bounds (если есть optimizer) ──
        st.markdown("---")
        st.markdown("### 🎯 Пресеты Optuna")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("Золотая полоса", key="preset_golden"):
                st.session_state.external_B = 30.0
                st.session_state.R_mm = 1000
                st.session_state.n_density = 1e20
                st.rerun()
        with col_p2:
            if st.button("Широкий поиск", key="preset_wide"):
                st.session_state.external_B = 33.0
                st.session_state.R_mm = 1500
                st.session_state.n_density = 1e21
                st.rerun()
        with col_p3:
            if st.button("Тонкая настройка", key="preset_fine"):
                st.session_state.external_B = 31.0
                st.session_state.R_mm = 1100
                st.session_state.n_density = 1.1e20
                st.rerun()
    
    # ── Главные кнопки ──
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Запустить симуляцию", type="primary", use_container_width=True):
            params = get_all_params()
            cfg = build_config(
                st.session_state.base_config,
                R_mm=params["R_mm"], h_mm=params["h_mm"],
                T_inner=params["T_inner"], T_outer=params["T_outer"],
                k_thermal=params["k_thermal"],
                pl_radius_mm=params["pl_radius_mm"],
                T_plasma=params["T_plasma_init"],
                n_density=params["n_density"],
                fuel_type=params["fuel_type"],
                P_initial=params["P_initial"],
                pump_speed=params["pump_speed"],
                E_field=params["E_field"],
                gap_nm=params["gap_nm"],
                external_B=params["external_B"],
                n_steps=params["n_steps"],
                dt=params["sim_dt"],
            )
            
            with st.spinner("Симуляция..."):
                result = run_simulation(cfg, mode="full", return_history=True)
            
            if result is None:
                st.error("Runaway — T > T_CAP")
            else:
                st.session_state.last_result = result
                st.session_state.last_history = result.pop("history", None)
                st.session_state.last_params = params
                st.success(f"✅ Q = {result['Q']:.4f} | Lawson = {result['Lawson_ratio']:.2f}×")
    
    with col2:
        if st.button("📊 Optuna", type="secondary", use_container_width=True):
            st.switch_page("pages/2_Оптимизация.py")
    
    with col3:
        if st.button("🤖 ИИ", type="secondary", use_container_width=True):
            st.switch_page("pages/3_ИИ.py")
    
    # ── Табло последнего результата ──
    if st.session_state.last_result:
        st.markdown("---")
        st.markdown("### 📋 Последний результат")
        
        r = st.session_state.last_result
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Q-фактор", f"{r['Q']:.4f}")
        with col_m2:
            st.metric("Lawson", f"{r['Lawson_ratio']:.2f}×")
        with col_m3:
            st.metric("T_plasma", f"{r['T_keV']:.1f} кэВ")
        with col_m4:
            st.metric("tau_E", f"{r['tau_E']:.2f} с")
        
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        with col_m5:
            st.metric("P_fusion", f"{r['P_fusion']/1e6:.2f} МВт")
        with col_m6:
            st.metric("P_alpha", f"{r['P_alpha']/1e6:.2f} МВт")
        with col_m7:
            st.metric("P_loss", f"{(r['P_cond']+r['P_rad']+r['P_sync'])/1e6:.2f} МВт")
        with col_m8:
            st.metric("ratio", f"{r['ratio']:.2f}")
        
        # Сохранение
        if st.button("💾 Сохранить результат"):
            run_dir = generate_run_dir()
            save_csv_results(os.path.join(run_dir, "result.csv"),
                           st.session_state.last_params, r)
            st.success(f"Сохранено в {run_dir}")
    else:
        st.info("Запустите симуляцию, чтобы увидеть результат")


if __name__ == "__main__":
    main()
