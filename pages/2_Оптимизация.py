import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import traceback
from datetime import datetime

# Add parent dir to path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from optuna_optimizer import GAKOptunaOptimizer, _get_ai_predictor


def get_base_config():
    """Базовая конфигурация для оптимизатора."""
    return {
        "thermal_monitor": {"R": 1.0, "h": 0.01, "T_inner": 350, "T_outer": 280, "k_thermal": 50},
        "acoustic_monitor": {"R": 1.0, "h": 0.01},
        "born_collapse_monitor": {"R": 1.0, "h": 0.01},
        "magnon_monitor": {"R": 1.0, "h": 0.01},
        "em_resonance_monitor": {"R": 1.0, "h": 0.01},
        "plasma_monitor": {
            "radius": 0.3, "temperature": 2e8, "density_number": 3e20,
            "fuel_type": "D-T", "auto_feed": False, "quantum_feedback": True,
            "external_B": 30.0,
        },
        "vacuum_monitor": {"P_initial": 1e4, "pump_speed": 1e-2},
        "quantum_vacuum": {"E_field": 5e17, "gap": 5e-9},
        "simulation": {"n_steps": 2500000, "dt": 1e-5},
    }


def safe_str(s):
    """Sanitize surrogate characters from string."""
    if s is None:
        return ""
    if isinstance(s, str):
        return s.encode('utf-8', errors='replace').decode('utf-8')
    return str(s)


st.set_page_config(page_title="GAK-WaveCAD | Оптимизация", layout="wide")
st.title("GAK-WaveCAD | Оптимизация")
st.markdown("---")

# --- Sidebar: settings ---
st.sidebar.header("Настройки оптимизации")

mode_options = {"Экономика (LCOE min)": "lcoe",
                "Зажигание (Q max + Lawson)": "ignition",
                "Усиление (Q max)": "gain",
                "Парето (pair_rate + Q)": "pareto"}
mode_label = st.sidebar.selectbox("Режим", list(mode_options.keys()))
mode = mode_options[mode_label]
st.session_state["mode"] = mode  # v45R.1: экспорт смотрит на выбранный режим

use_ai = st.sidebar.checkbox("AI-фильтр (суррогат)", value=True)
n_trials = st.sidebar.slider("Trials (без AI)", 10, 200, 50)
surrogate_trials = st.sidebar.slider("AI trials", 100, 2000, 500, step=100)
refine_top = st.sidebar.slider("Уточнение топ-N", 3, 30, 10)
n_steps = st.sidebar.slider("Шагов симуляции", 500000, 5000000, 2500000, step=500000)

# --- Bounds UI ---
# v36.1 UI FIX: слайдеры синхронизированы с BOUNDS оптимизатора.
# Границы и дефолты берутся прямо из GAKOptunaOptimizer.BOUNDS —
# при изменении BOUNDS в коде интерфейс подхватывает их автоматически.
# Вручную можно только СУЗИТЬ золотую зону, расширить — нельзя.
st.sidebar.markdown("### Диапазоны (золотая зона)")
BOUNDS_LABELS = {
    "R_mm":          "R сферы, мм",
    "external_B":    "B внешнее, Тл",
    "T_plasma_init": "T плазмы init, K",
    "n_density":     "Плотность, м^-3",
    "E_field":       "E-поле, В/м",
}
bounds = {}
for key, label in BOUNDS_LABELS.items():
    lo, hi = GAKOptunaOptimizer.BOUNDS[key]
    lo, hi = float(lo), float(hi)
    fmt = "%.2e" if hi >= 1e6 else ("%.1f" if hi < 100 else "%.0f")
    bounds[key] = st.sidebar.slider(label, lo, hi, (lo, hi), format=fmt)

# --- AI predictor status ---
predictor = None
if use_ai:
    try:
        predictor = _get_ai_predictor(script_dir)
        if predictor and predictor.is_trained:
            try:
                err_log = getattr(predictor, 'last_error_pct', 0)
                err_lin = getattr(predictor, 'last_error_linear_pct', 0)
                n_exp = getattr(predictor, 'n_experiences', 0)
                st.sidebar.success(
                    f"AI: {n_exp} exp | log {err_log:.1f}% | lin {err_lin:.1f}%")
            except Exception:
                st.sidebar.success("AI: loaded")
        else:
            st.sidebar.warning("AI: model not loaded")
            predictor = None
    except Exception as e:
        st.sidebar.warning(f"AI: {safe_str(e)}")
        predictor = None

# --- Run button ---
st.markdown("## Запуск оптимизации")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Запустить", type="primary", use_container_width=True):
        base_config = get_base_config()
        base_config["simulation"]["n_steps"] = n_steps

        optimizer = GAKOptunaOptimizer(base_config)
        # v36.1 UI FIX: bounds теперь по умолчанию совпадают с BOUNDS.
        # update() оставлен: позволяет только сузить зону через слайдер,
        # выйти за золотые границы интерфейсом больше нельзя.
        optimizer.BOUNDS.update(bounds)

        if predictor and predictor.is_trained:
            optimizer.set_predictor(predictor)

        try:
            with st.status("Оптимизация...", expanded=True) as status:
                optimizer.optimize(
                    mode=mode,
                    n_trials=n_trials,
                    use_surrogate=(predictor is not None and predictor.is_trained),
                    surrogate_trials=surrogate_trials,
                    refine_top=refine_top,
                )
                status.update(label="Оптимизация завершена!", state="complete")

            st.session_state["optimizer"] = optimizer
            st.session_state["study_name"] = optimizer.study.study_name  # v45R.1
            st.session_state["opt_done"] = True
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка оптимизации: {safe_str(e)}")
            try:
                tb = traceback.format_exc()
                tb_safe = tb.encode('utf-8', errors='replace').decode('utf-8')
                with st.expander("Traceback"):
                    st.code(tb_safe)
            except Exception:
                pass

# --- Results ---
if st.session_state.get("opt_done") and "optimizer" in st.session_state:
    opt = st.session_state["optimizer"]

    st.markdown("---")
    st.markdown("## Результаты")

    # Pre-screen stats
    stats = opt._prescreen_stats
    if stats["screened"] > 0:
        st.info(
            f"Pre-screen: {stats['screened']} проверено, "
            f"{stats['rejected']} отбраковано, "
            f"{stats['passed']} пущено в физику"
        )

    # MLP vs physics table
    refine_log = opt.get_refine_log()
    if refine_log:
        st.markdown("### MLP vs физика (топ-N)")
        try:
            df_refine = pd.DataFrame(refine_log)
            st.dataframe(df_refine, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось показать таблицу: {safe_str(e)}")

    # Results summary
    try:
        summary = opt.get_results_summary()
        if summary is not None:
            st.markdown("### Параметры и метрики")
            st.dataframe(summary, use_container_width=True)
    except Exception as e:
        st.warning(f"Не удалось показать сводку: {safe_str(e)}")

    # Convergence plot
    try:
        fig = opt.plot_convergence()
        if fig is not None:
            st.markdown("### График сходимости")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Не удалось показать график: {safe_str(e)}")

    # CSV export
    try:
        csv_df = opt.export_best_csv()
        if csv_df is not None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
            csv_data = csv_df.to_csv(index=False)
            st.download_button(
                "Экспорт CSV",
                data=csv_data,
                file_name=f"{timestamp}_export.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.warning(f"Не удалось экспортировать CSV: {safe_str(e)}")

st.divider()
if st.button("📂 Выгрузить все триалы в CSV"):
    import optuna
    _study_name_export = st.session_state.get("study_name") or \
        f"gak_opt_{st.session_state.get('mode', 'lcoe')}_v45R"
    try:
        study = optuna.load_study(
            study_name=_study_name_export,
            storage="sqlite:///models/optuna_study.db",
        )
    except KeyError:
        st.warning("Study не найден в базе — запусти оптимизацию, полка создастся сама.")
        st.stop()
    df = study.trials_dataframe()
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Скачать trials_export.csv",
        csv_bytes,
        "trials_export.csv",
        "text/csv",
        key="dl_trials",
    )
    st.dataframe(df.head(10))
