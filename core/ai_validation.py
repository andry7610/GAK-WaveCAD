"""
ai_validation.py — ИИ-валидация трансформера GAK-WaveCAD.

Сравнивает прогноз ИИ-модели с реальной симуляцией на данных,
которых модель не видела при обучении.

Важно: валидация использует ТЕ ЖЕ 5 параметров, что и DOE-обучение:
  - T_plasma (temperature)
  - n_density (density_number)
  - pl_radius (radius)
  - E_field
  - external_B

Остальные параметры берутся из base_config (как при обучении).

v0.2 — обработка runaway (T > T_CAP), пропуск нестабильных точек
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import json

# Порог runaway — если T_plasma превышает это значение, точка считается нестабильной
T_CAP = 1e10  # K

# Диапазоны DOE-обучения (из app.py)
_DOE_RANGES = {
    "D-T":   {"T": (1e6, 1e8),   "n": (1e18, 1e22), "E": (1e16, 1e20), "R": (1, 50), "B": (5, 45), "seed": 99},
    "D-D":   {"T": (1e7, 5e8),   "n": (1e19, 1e22), "E": (1e16, 1e20), "R": (1, 50), "B": (5, 45), "seed": 99},
    "D-He3": {"T": (5e7, 1e9),   "n": (1e20, 1e22), "E": (1e16, 1e20), "R": (1, 50), "B": (5, 45), "seed": 99},
}

_FUEL_SUFFIX = {"D-T": "DT", "D-D": "DD", "D-He3": "DHe3"}


def _get_model_path(script_dir, fuel):
    suffix = _FUEL_SUFFIX.get(fuel, "DT")
    return os.path.join(script_dir, "models", f"ai_predictor_{suffix}.json")


def _build_val_config(base_config, fuel_type, T_p, n_d, R_p_mm, E_f, B_f,
                      R_mm=50, h_mm=2.0, T_inner=310.0, T_outer=293.0, k_thermal=15.0,
                      gap_nm=3.5):
    """Сборка конфига для валидации — та же структура, что в DOE."""
    cfg = dict(base_config)
    cfg["plasma_monitor"] = {
        **base_config.get("plasma_monitor", {}),
        "radius": R_p_mm / 1000,
        "temperature": T_p,
        "density_number": n_d,
        "fuel_type": fuel_type,
        "external_B": B_f,
        "auto_feed": False,
        "quantum_feedback": True,
    }
    cfg["thermal_monitor"] = {
        **base_config.get("thermal_monitor", {}),
        "R": R_mm / 1000,
        "h": h_mm / 1000,
        "T_inner": T_inner,
        "T_outer": T_outer,
        "k_thermal": k_thermal,
    }
    cfg["quantum_vacuum"] = {
        **base_config.get("quantum_vacuum", {}),
        "E_field": E_f,
        "gap": gap_nm * 1e-9,
    }
    cfg["simulation"] = {"n_steps": 1000, "dt": 1.0e-5}
    return cfg


def _run_val_sim(cfg):
    """Быстрая симуляция — плазма + Швингер (как в DOE).
    
    Возвращает history (dict) или None при runaway (T > T_CAP).
    """
    from physical_modules.plasma_monitor import PlasmaMonitor

    plasma = PlasmaMonitor(cfg.get("plasma_monitor", {}))
    plasma.init()

    sim_n = cfg["simulation"]["n_steps"]
    sim_dt = cfg["simulation"]["dt"]

    V_plasma = (4.0 / 3.0) * np.pi * (plasma.radius ** 3)
    pair_energy_gain = plasma.pair_energy_gain
    external_B = cfg["plasma_monitor"]["external_B"]
    n_density = cfg["plasma_monitor"]["density_number"]

    history = {"step": [], "T_plasma": [], "P_gas": [], "pair_rate": [], "ratio": []}

    for step_i in range(sim_n):
        tau_E = plasma.compute_tau_E()
        dt_actual = min(sim_dt, 0.05 * tau_E) if tau_E > 0 else sim_dt
        step_result = plasma.step(dt_actual)

        T_plasma = step_result["temperature"]
        P_fusion = step_result["P_fusion"]
        P_cond = step_result["P_cond"]
        P_rad = step_result["P_rad"]
        tau_E = step_result["tau_E"]

        # ── Runaway detection ──────────────────────────────────────────
        if T_plasma > T_CAP:
            return None  # runaway — точка нестабильна

        # Швингер
        pair_rate = 0.0
        qv_cfg = cfg.get("quantum_vacuum", {})
        E_field = qv_cfg.get("E_field", 0.0)
        gap = qv_cfg.get("gap", 3.5e-9)

        if E_field > 0 and gap > 0 and T_plasma > 0:
            E_schwinger = 1.32e18  # В/м
            if E_field < E_schwinger:
                schwinger_rate = np.exp(-np.pi * E_schwinger / max(E_field, 1e10))
                vol_gap = np.pi * gap**3 / 6.0
                n_pairs = schwinger_rate * vol_gap * 1e45
                pair_rate = n_pairs
                E_pair = 2 * 511e3 * 1.602e-19  # 2 × me × c²
                P_schwinger = pair_rate * E_pair
                P_heat = P_fusion + P_schwinger
            else:
                P_heat = P_fusion + P_fusion * 0.1
        else:
            P_heat = P_fusion

        P_heat_total = P_heat
        ratio = P_heat_total / max(P_cond + P_rad, 1e-30)

        P_gas = max(0.0, n_density * 1.38e-23 * max(T_plasma, 1.0))

        if step_i % 2 == 0:  # stride
            history["step"].append(step_i)
            history["T_plasma"].append(T_plasma)
            history["P_gas"].append(P_gas)
            history["pair_rate"].append(max(pair_rate, 0.0))
            history["ratio"].append(ratio)

    return history


def render_ai_validation_tab():
    """Главная функция рендера вкладки ИИ-валидации."""
    st.markdown("### 🤖 ИИ-валидация трансформера")
    st.markdown("Сравнение прогноза ИИ с реальной симуляцией на данных, **не использованных при обучении**.")
    st.markdown(f"⚠️ Точки с T > {T_CAP:.0e} K (runaway) пропускаются.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir.endswith("core"):
        script_dir = os.path.dirname(script_dir)

    # Загружаем base_config
    try:
        from core.config_loader import ConfigLoader
        config_path = os.path.join(script_dir, "configs", "config.yaml")
        base_config = ConfigLoader.load(config_path)
    except Exception:
        st.error("Не удалось загрузить config.yaml")
        return

    # Определяем тип топлива из конфига
    fuel_type = base_config.get("plasma_monitor", {}).get("fuel_type", "D-T")

    # Проверяем наличие модуля ИИ
    try:
        from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
    except ImportError:
        st.error("Модуль ai_transformer_predictor не найден")
        return

    # Настройки валидации
    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        val_fuel = st.selectbox("Топливо", ["D-T", "D-D", "D-He3"], index=["D-T", "D-D", "D-He3"].index(fuel_type),
                                  key="val_fuel")
    with col_v2:
        n_val_points = st.slider("Точек валидации", 10, 100, 30, step=10, key="n_val_points")
    with col_v3:
        val_seed = st.number_input("Seed", value=99, key="val_seed")

    # Загружаем модель
    model_path = _get_model_path(script_dir, val_fuel)

    if not os.path.exists(model_path):
        st.warning(f"Файл модели не найден: {model_path}")
        st.info(f"Обучите модель для топлива {val_fuel} на вкладке ИИ (DOE-автообучение).")
        return

    predictor = GAKTransformerPredictor()
    loaded = predictor.load(model_path)

    if not loaded or not predictor.is_trained:
        st.error("Модель не загружена или не обучена")
        return

    st.success(f"✅ Модель загружена | Опытов: {predictor.n_experiences} | "
               f"Ошибка (log): {predictor.last_error_pct:.1f}% | (лин): {predictor.last_error_linear_pct:.1f}%")

    # Чекбоксы для выбора таргетов
    st.markdown("#### Параметры валидации")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        val_T = st.checkbox("T_plasma", value=True, key="val_T")
    with col_t2:
        val_P = st.checkbox("P_gas", value=False, key="val_P")
    with col_t3:
        val_pair = st.checkbox("pair_rate", value=False, key="val_pair")
    with col_t4:
        val_ratio = st.checkbox("ratio", value=True, key="val_ratio")

    target_flags = {"T_plasma": val_T, "P_gas": val_P, "pair_rate": val_pair, "ratio": val_ratio}
    active_targets = [k for k, v in target_flags.items() if v]

    if not active_targets:
        st.warning("Выберите хотя бы один таргет для валидации")
        return

    # Диапазоны DOE для выбранного топлива
    doe = _DOE_RANGES.get(val_fuel, _DOE_RANGES["D-T"])

    if st.button("🚀 Запустить валидацию", type="primary", key="val_start"):
        progress = st.progress(0, text="Валидация...")
        status = st.empty()

        rng = np.random.default_rng(int(val_seed))

        # Генерация случайных конфигов в диапазонах DOE
        configs = []
        for _ in range(n_val_points):
            T_p = rng.uniform(*doe["T"])
            n_d = 10 ** rng.uniform(np.log10(doe["n"][0]), np.log10(doe["n"][1]))
            R_p = rng.uniform(*doe["R"])
            E_f = 10 ** rng.uniform(np.log10(doe["E"][0]), np.log10(doe["E"][1]))
            B_f = rng.uniform(*doe["B"])
            configs.append((T_p, n_d, R_p, E_f, B_f))

        results = []
        runaway_count = 0

        for i, (T_p, n_d, R_p, E_f, B_f) in enumerate(configs):
            # Сборка конфига
            cfg = _build_val_config(base_config, val_fuel, T_p, n_d, R_p, E_f, B_f)

            # Прогноз ИИ
            forecast = predictor.predict(cfg)

            # Реальная симуляция
            history = _run_val_sim(cfg)

            # ── Пропуск runaway-точек ───────────────────────────────────
            if history is None:
                runaway_count += 1
                progress.progress((i + 1) / n_val_points, text=f"Валидация: {i+1}/{n_val_points}")
                status.text(f"Точка {i+1}/{n_val_points}: RUNAWAY (T > {T_CAP:.0e}) — пропущена")
                continue

            # Финальные значения (среднее последних 20% точек)
            n_hist = len(history["T_plasma"])
            if n_hist == 0:
                continue
            tail = max(1, n_hist // 5)

            for target in active_targets:
                real_vals = history.get(target, [])
                if not real_vals:
                    continue
                real_mean = float(np.mean(real_vals[-tail:]))
                pred_val = forecast.get(target, 0.0) if forecast else 0.0

                if abs(real_mean) > 1e-30:
                    err_pct = abs(pred_val - real_mean) / abs(real_mean) * 100.0
                else:
                    err_pct = 0.0

                results.append({
                    "idx": i,
                    "target": target,
                    "real": real_mean,
                    "pred": pred_val,
                    "err_pct": err_pct,
                })

            progress.progress((i + 1) / n_val_points, text=f"Валидация: {i+1}/{n_val_points}")
            status.text(f"Точка {i+1}/{n_val_points}: T={T_p:.1e}, n={n_d:.1e}, R={R_p:.1f}мм, E={E_f:.1e}, B={B_f:.1f}")

        progress.empty()
        status.empty()

        # Информация о runaway
        n_valid = len(configs) - runaway_count
        if runaway_count > 0:
            st.info(f"Runaway: {runaway_count}/{n_val_points} точек пропущено (T > {T_CAP:.0e} K). "
                    f"Валидация на {n_valid} стабильных точках.")

        if not results:
            st.error("Нет результатов — все точки ушли в runaway или симуляция не вернула данных")
            return

        df = pd.DataFrame(results)

        # Сводка по таргетам
        st.markdown("#### 📊 Сводка ошибок")
        summary = df.groupby("target").agg(
            mean_err=("err_pct", "mean"),
            median_err=("err_pct", "median"),
            max_err=("err_pct", "max"),
            min_err=("err_pct", "min"),
            count=("err_pct", "count"),
        ).reset_index()

        st.dataframe(summary.style.format({
            "mean_err": "{:.1f}%",
            "median_err": "{:.1f}%",
            "max_err": "{:.1f}%",
            "min_err": "{:.1f}%",
        }), use_container_width=True)

        # График прогноз vs реальность для каждого таргета
        n_targets = len(active_targets)
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
        if n_targets == 1:
            axes = [axes]

        for ax, target in zip(axes, active_targets):
            sub = df[df["target"] == target]
            ax.scatter(sub["real"], sub["pred"], alpha=0.6, s=40, c="crimson")
            # Линия идеального прогноза
            all_vals = np.concatenate([sub["real"].values, sub["pred"].values])
            lo, hi = np.min(all_vals), np.max(all_vals)
            if lo > 0 and hi > 0:
                ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="идеал")
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlabel("Реальность")
            ax.set_ylabel("Прогноз")
            ax.set_title(target)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Таблица с экспортом
        st.markdown("#### 📋 Детальные результаты")
        export_df = df.pivot(index="idx", columns="target", values=["real", "pred", "err_pct"])
        st.dataframe(export_df, use_container_width=True)

        # Экспорт в CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Скачать CSV",
            data=csv,
            file_name=f"ai_validation_{val_fuel}_{int(val_seed)}.csv",
            mime="text/csv",
        )
