"""
ИИ: DOE-обучение, прогноз, валидация MLP-предиктора GAK-WaveCAD.
"""
import streamlit as st
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.session_state import init_session_state, get_ai_predictor, set_ai_predictor

st.set_page_config(page_title="ИИ", page_icon="🤖", layout="wide")
init_session_state()

# ── Константы ──
T_CAP = 1e10  # K — порог runaway

_DOE_RANGES = {
    "D-T":   {"T": (1e6, 1e8),   "n": (1e18, 1e22), "E": (1e16, 1e20), "R": (1, 50), "B": (5, 45), "seed": 99},
    "D-D":   {"T": (1e7, 5e8),   "n": (1e19, 1e22), "E": (1e16, 1e20), "R": (1, 50), "B": (5, 45), "seed": 99},
    "D-He3": {"T": (5e7, 1e9),   "n": (1e20, 1e22), "E": (1e16, 1e20), "R": (1, 50), "B": (5, 45), "seed": 99},
}

_FUEL_SUFFIX = {"D-T": "DT", "D-D": "DD", "D-He3": "DHe3"}


# ── Сборка конфига для DOE-точки ──
def _build_doe_config(base_config, fuel_type, T_p, n_d, R_p_mm, E_f, B_f):
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
    }
    cfg["quantum_vacuum"] = {
        **base_config.get("quantum_vacuum", {}),
        "E_field": E_f,
        "gap": 3.5e-9,
    }
    cfg["simulation"] = {"n_steps": 1000, "dt": 1.0e-5}
    return cfg


# ── Быстрая симуляция одной DOE-точки ──
def _run_doe_sim(cfg):
    """Симуляция плазма + Швингер. Возвращает dict таргетов или None (runaway)."""
    from physical_modules.plasma_monitor import PlasmaMonitor

    plasma = PlasmaMonitor(cfg.get("plasma_monitor", {}))
    plasma.init()

    sim_n = cfg["simulation"]["n_steps"]
    sim_dt = cfg["simulation"]["dt"]
    n_density = cfg["plasma_monitor"]["density_number"]

    T_plasma = 0.0
    pair_rate = 0.0
    ratio = 0.0
    P_fusion = 0.0
    P_cond = 0.0
    P_rad = 0.0

    for step_i in range(sim_n):
        tau_E = plasma.compute_tau_E()
        dt_actual = min(sim_dt, 0.05 * tau_E) if tau_E > 0 else sim_dt
        step_result = plasma.step(dt_actual)

        T_plasma = step_result["temperature"]

        # ── Runaway check ──
        if T_plasma > T_CAP:
            return None  # runaway — пропускаем точку

        P_fusion = step_result["P_fusion"]
        P_cond = step_result["P_cond"]
        P_rad = step_result["P_rad"]

        # Швингер
        pair_rate = 0.0
        qv_cfg = cfg.get("quantum_vacuum", {})
        E_field = qv_cfg.get("E_field", 0.0)
        gap = qv_cfg.get("gap", 3.5e-9)

        if E_field > 0 and gap > 0 and T_plasma > 0:
            E_schwinger = 1.32e18  # В/м
            if E_field < E_schwinger:
                schwinger_rate = np.exp(-np.pi * E_schwinger / max(E_field, 1e10))
                vol_gap = np.pi * gap ** 3 / 6.0
                n_pairs = schwinger_rate * vol_gap * 1e45
                pair_rate = n_pairs
                E_pair = 2 * 511e3 * 1.602e-19
                P_schwinger = pair_rate * E_pair
                P_heat = P_fusion + P_schwinger
            else:
                P_heat = P_fusion + P_fusion * 0.1
        else:
            P_heat = P_fusion

        ratio = P_heat / max(P_cond + P_rad, 1e-30)

    return {
        "T_plasma": T_plasma,
        "P_gas": max(0.0, n_density * 1.38e-23 * max(T_plasma, 1.0)),
        "pair_rate": max(pair_rate, 0.0),
        "ratio": ratio,
    }


# ── Генерация DOE-точек (Latin Hypercube) ──
def _gen_lhs_points(ranges, n_points, seed=99):
    """Latin Hypercube Sampling в логарифмическом пространстве."""
    rng = np.random.default_rng(seed)
    T_lo, T_hi = ranges["T"]
    n_lo, n_hi = ranges["n"]
    E_lo, E_hi = ranges["E"]
    R_lo, R_hi = ranges["R"]
    B_lo, B_hi = ranges["B"]

    # LHS: делим каждый диапазон на n_points равных отрезков, берём по одной точке
    def lhs_1d(lo, hi, log=False):
        perms = rng.permutation(n_points)
        u = (perms + rng.uniform(0, 1, n_points)) / n_points
        if log:
            return 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
        else:
            return lo + u * (hi - lo)

    Ts = lhs_1d(T_lo, T_hi, log=True)
    ns = lhs_1d(n_lo, n_hi, log=True)
    Es = lhs_1d(E_lo, E_hi, log=True)
    Rs = lhs_1d(R_lo, R_hi, log=False)
    Bs = lhs_1d(B_lo, B_hi, log=False)

    return list(zip(Ts, ns, Rs, Es, Bs))


# ════════════════════════════════════════════════════════════════
#  СТРАНИЦА
# ════════════════════════════════════════════════════════════════

st.title("🤖 ИИ-предиктор (MLP)")

# ── Пути ──
script_dir = st.session_state.get("script_dir", os.path.dirname(os.path.abspath(__file__)))
if script_dir.endswith("pages"):
    script_dir = os.path.dirname(script_dir)

# ── Загрузка base_config ──
try:
    from core.config_loader import ConfigLoader
    config_path = os.path.join(script_dir, "configs", "config.yaml")
    base_config = ConfigLoader.load(config_path)
except Exception as e:
    st.error(f"Не удалось загрузить config.yaml: {e}")
    st.stop()

fuel_type = base_config.get("plasma_monitor", {}).get("fuel_type", "D-T")

# ── Импорт предиктора ──
try:
    from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
except ImportError:
    st.error("Модуль ai_transformer_predictor не найден")
    st.stop()

# ── Выбор топлива ──
fuel = st.selectbox("Топливо", ["D-T", "D-D", "D-He3"],
                     index=["D-T", "D-D", "D-He3"].index(fuel_type),
                     key="ai_fuel")
suffix = _FUEL_SUFFIX[fuel]
model_path = os.path.join(script_dir, "models", f"ai_predictor_{suffix}.json")

# ── Текущий предиктор ──
predictor = get_ai_predictor()

# ════════════════════════════════════════════════════════════════
#  Секция 1: Загрузка / статус модели
# ════════════════════════════════════════════════════════════════
st.markdown("### 📦 Статус модели")

col_s1, col_s2 = st.columns([3, 1])

with col_s1:
    if predictor is not None and predictor.is_trained:
        st.success(
            f"✅ Модель загружена | Опытов: {predictor.n_experiences} | "
            f"Ошибка (log): {predictor.last_error_pct:.1f}% | "
            f"(лин): {predictor.last_error_linear_pct:.1f}%"
        )
    elif os.path.exists(model_path):
        st.info(f"Файл модели найден: {model_path}")
        if st.button("Загрузить модель"):
            p = GAKTransformerPredictor()
            if p.load(model_path) and p.is_trained:
                set_ai_predictor(p, model_path)
                st.success(f"✅ Загружено | Опытов: {p.n_experiences}")
                st.rerun()
            else:
                st.error("Не удалось загрузить — возможно, старый формат (трансформер). "
                         "Переобучите модель.")
    else:
        st.warning(f"Модель не найдена: {model_path}")
        st.info("Запустите DOE-обучение ниже 👇")

with col_s2:
    if predictor is not None:
        if st.button("🗑 Сбросить"):
            set_ai_predictor(None, None)
            st.rerun()

# ════════════════════════════════════════════════════════════════
#  Секция 2: DOE-обучение
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔬 DOE-обучение")

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    n_doe = st.number_input("Точек DOE", 30, 2000, 100, step=10, key="n_doe")
with col_d2:
    doe_seed = st.number_input("Seed", value=99, key="doe_seed")
with col_d3:
    doe_retrain = st.checkbox("Дополнить существующую модель", value=False, key="doe_retrain")
with col_d4:
    st.write("")  # отступ

if st.button("🚀 Запустить DOE-обучение", type="primary", use_container_width=True):
    ranges = _DOE_RANGES[fuel]

    # Генерация точек
    points = _gen_lhs_points(ranges, n_doe, seed=doe_seed)
    st.info(f"Сгенерировано {n_doe} точек LHS для топлива {fuel}")

    # Создаём или используем существующий предиктор
    if doe_retrain and predictor is not None and predictor.is_trained:
        p = predictor
        st.info(f"Дополнение модели: {p.n_experiences} существующих опытов")
    else:
        p = GAKTransformerPredictor()
        st.info("Начинаем обучение с нуля")

    # Прогресс
    progress = st.progress(0.0, text="Подготовка...")
    log_area = st.empty()
    log_lines = []

    stable_count = 0
    runaway_count = 0
    error_list = []

    for i, (T_p, n_d, R_p, E_f, B_f) in enumerate(points):
        # Сборка конфига
        cfg = _build_doe_config(base_config, fuel, T_p, n_d, R_p, E_f, B_f)

        # Симуляция
        try:
            result = _run_doe_sim(cfg)
        except Exception as ex:
            log_lines.append(f"[{i+1}/{n_doe}] ОШИБКА: {ex}")
            continue

        if result is None:
            runaway_count += 1
            log_lines.append(f"[{i+1}/{n_doe}] ⚠ Runaway (T > {T_CAP:.0e} K) — пропуск")
        else:
            stable_count += 1
            # Таргеты
            targets = {
                "T_plasma": result["T_plasma"],
                "P_gas": result["P_gas"],
                "pair_rate": result["pair_rate"],
                "ratio": result["ratio"],
            }

            # Обучение
            try:
                p.self_learn(cfg, targets)
                err = p.last_error_pct
                error_list.append(err)
                log_lines.append(
                    f"[{i+1}/{n_doe}] ✅ T={T_p:.2e} n={n_d:.2e} R={R_p:.1f} "
                    f"E={E_f:.2e} B={B_f:.1f} → T_out={result['T_plasma']:.2e} "
                    f"ratio={result['ratio']:.3f} err={err:.1f}%"
                )
            except Exception as ex:
                log_lines.append(f"[{i+1}/{n_doe}] ОШИБКА ОБУЧЕНИЯ: {ex}")

        # Обновление прогресса
        progress.progress((i + 1) / n_doe, text=f"Обработано {i+1}/{n_doe} | Стабильно: {stable_count} | Runaway: {runaway_count}")

        # Лог (последние 8 строк)
        log_area.text("\n".join(log_lines[-8:]))

    # Сохранение модели
    os.makedirs(os.path.join(script_dir, "models"), exist_ok=True)
    p.save(model_path)
    set_ai_predictor(p, model_path)

    progress.empty()
    log_area.empty()

    # Итог
    st.success(
        f"✅ DOE завершён | Стабильно: {stable_count} | Runaway: {runaway_count} | "
        f"Всего опытов: {p.n_experiences} | Финальная ошибка: {p.last_error_pct:.1f}%"
    )

    # График ошибки
    if error_list:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(1, len(error_list) + 1), error_list, "b-", linewidth=0.8)
        ax.set_xlabel("Опыт")
        ax.set_ylabel("Ошибка, %")
        ax.set_title("Кривая обучения")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.info(f"Модель сохранена: {model_path}")

# ════════════════════════════════════════════════════════════════
#  Секция 3: Прогноз
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📈 Прогноз")

if predictor is not None and predictor.is_trained:
    ranges = _DOE_RANGES[fuel]

    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        test_T = st.number_input("T, K", value=5e7, format="%.2e", key="test_T")
    with col_p2:
        test_n = st.number_input("n, м⁻³", value=1e20, format="%.2e", key="test_n")
    with col_p3:
        test_R = st.number_input("R плазмы, мм", value=10.0, key="test_R")
    with col_p4:
        test_E = st.number_input("E, В/м", value=1e18, format="%.2e", key="test_E")
    with col_p5:
        test_B = st.number_input("B, Тл", value=30.0, key="test_B")

    if st.button("Предсказать"):
        cfg = _build_doe_config(base_config, fuel, test_T, test_n, test_R, test_E, test_B)
        pred = predictor.predict(cfg)

        if pred is not None:
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("T_plasma", f"{pred['T_plasma']:.2e} K")
            with col_r2:
                st.metric("P_gas", f"{pred['P_gas']:.2e} Па")
            with col_r3:
                st.metric("pair_rate", f"{pred['pair_rate']:.2e}")
            with col_r4:
                st.metric("ratio", f"{pred['ratio']:.3f}")

            # Сравнение с реальной симуляцией
            if st.checkbox("Сравнить с симуляцией"):
                with st.spinner("Симуляция..."):
                    real = _run_doe_sim(cfg)
                if real is None:
                    st.warning("Runaway в симуляции — нельзя сравнить")
                else:
                    st.markdown("**Сравнение предсказания и реальности:**")
                    comp_data = {
                        "Параметр": ["T_plasma", "P_gas", "pair_rate", "ratio"],
                        "Предсказание": [pred['T_plasma'], pred['P_gas'], pred['pair_rate'], pred['ratio']],
                        "Реальность": [real['T_plasma'], real['P_gas'], real['pair_rate'], real['ratio']],
                    }
                    st.dataframe(comp_data, use_container_width=True)
else:
    st.info("Обучите модель (DOE), чтобы получить прогноз")

# ════════════════════════════════════════════════════════════════
#  Секция 4: Валидация
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔍 Валидация")

if predictor is not None and predictor.is_trained:
    if st.button("Запустить валидацию", type="secondary"):
        try:
            from ai_validation import render_ai_validation_tab
            render_ai_validation_tab()
        except ImportError:
            st.info("Модуль ai_validation не найден — валидация доступна на отдельной вкладке")
else:
    st.info("Обучите модель (DOE), чтобы запустить валидацию")
