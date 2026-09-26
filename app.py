"""
GAK-WaveCAD VERSION = "0.9.2 — Streamlit-интерфейс мультифизического САПР.

Обновления:
  - Модульные чекбоксы (включение/выключение)
  - Базовые модули (термалка, плазма) — всегда включены
  - Проверка зависимостей между модулями
  - Пресеты (сохранить/загрузить набор модулей)
  - Индикатор активных модулей
  - session_state для сохранения галочек
  - Селектор режима для графиков

Запуск:
    streamlit run app.py
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Версия ────────────────────────────────────────────────────────────
VERSION = "0.9.2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Импорт модулей ────────────────────────────────────────────────────
try:
    from core.config_loader import ConfigLoader
    from physical_modules.osmosis_monitor import OsmosisMonitor
    from physical_modules.thermal_monitor import ThermalMonitor
    from physical_modules.acoustic_monitor import AcousticMonitor
    from physical_modules.born_collapse_monitor import BornCollapseMonitor
    from physical_modules.magnon_monitor import MagnonMonitor
    from physical_modules.em_resonance_monitor import EMResonanceMonitor
    from physical_modules.plasma_monitor import PlasmaMonitor
    from physical_modules.vacuum_monitor import VacuumMonitor
    from physical_modules.quantum_vacuum import GAKQuantumVacuum
    from physical_modules.bio_bridge import BioBridge
    from physical_modules.coupling_monitor import CouplingMonitor
except ImportError as e:
    st.error(f"❌ Критическая ошибка импорта модулей: {e}")
    st.stop()

try:
    from optuna_optimizer import GAKOptunaOptimizer
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

st.set_page_config(
    page_title="GAK-WaveCAD",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 GAK-WaveCAD — мультифизический САПР")
st.caption(f"Версия {VERSION}")

# ── Загрузка конфига ──────────────────────────────────────────────────
config_path = os.path.join(SCRIPT_DIR, "configs", "config.yaml")


@st.cache_data
def load_config(path):
    return ConfigLoader.load(path)


try:
    base_config = load_config(config_path)
except Exception as e:
    st.error(f"❌ Не удалось загрузить config.yaml: {e}")
    st.stop()

# ── Определение модулей ──────────────────────────────────────────────
# Базовые модули — нельзя отключить
BASE_MODULES = ["thermal", "plasma"]

# Опциональные модули: ключ -> (метка, имя класса в конфиге)
OPTIONAL_MODULES = {
    "osmosis":   ("💧 Осмос",          "osmosis_monitor"),
    "acoustic":  ("🔊 Акустика",       "acoustic_monitor"),
    "collapse":  ("🕳️ Коллапс (Борн)",  "born_collapse_monitor"),
    "magnon":    ("🧲 Магноны",         "magnon_monitor"),
    "em":        ("📡 EM-резонанс",     "em_resonance_monitor"),
    "vacuum":    ("🌀 Вакуум",          "vacuum_monitor"),
    "quantum":   ("⚡ Квант. вакуум",   "quantum_vacuum"),
    "bio":       ("🧠 Био-мост",       "bio_bridge"),
}

ALL_MODULE_KEYS = list(OPTIONAL_MODULES.keys())
TOTAL_MODULES = len(BASE_MODULES) + len(OPTIONAL_MODULES)

# Зависимости: модуль -> список модулей, которые должны быть включены
DEPENDENCIES = {
    "collapse": ["osmosis"],      # коллапс использует pi_Pa из осмоса
    "bio":      ["vacuum"],       # био использует P_gas из вакуума
    "quantum":  [],               # квантовый вакуум работает с плазмой (всегда включена)
    "acoustic": [],               # акустика использует термалку (всегда включена)
    "magnon":   [],                # магноны используют термалку (всегда включена)
    "em":       [],                # EM использует термалку (всегда включена)
    "osmosis":  [],                # осмос самостоятелен
    "vacuum":   [],                # вакуум самостоятелен
}

# ── session_state: инициализация галочек ─────────────────────────────
if "module_states" not in st.session_state:
    st.session_state.module_states = {
        "osmosis": True,
        "acoustic": True,
        "collapse": False,
        "magnon": False,
        "em": False,
        "vacuum": True,
        "quantum": True,
        "bio": False,
    }

if "presets" not in st.session_state:
    st.session_state.presets = {}


# ── Сайдбар ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")

    # ── Группа 1: Точность симуляции ──
    st.subheader("📊 Точность симуляции")

    accuracy = st.radio(
        "Режим",
        ("🚀 Быстро", "⚖️ Стандарт", "🔬 Точно"),
        index=0,
    )

    if accuracy == "🚀 Быстро":
        n_steps = 1000
        dt = 1.0e-5
        profile_name = "fast"
    elif accuracy == "⚖️ Стандарт":
        n_steps = 10000
        dt = 1.0e-6
        profile_name = "standard"
    else:
        n_steps = 50000
        dt = 1.0e-7
        profile_name = "accurate"

    st.caption(f"Шагов: **{n_steps:,}** | dt: **{dt:.0e}** | Время: **{n_steps * dt:.2e}** с")

    if accuracy == "🔬 Точно":
        st.warning("⚠️ Расчёт может занять 15–30 секунд")

    st.divider()

    # ── Группа 2: Модули (чекбоксы) ──
    st.subheader("🔧 Модули")

    # Базовые — показываем как включённые, нельзя снять
    for mod_key in BASE_MODULES:
        labels = {"thermal": "🌡️ Термалка", "plasma": "🔥 Плазма"}
        st.checkbox(f"{labels[mod_key]} (базовый)", value=True, disabled=True, key=f"base_{mod_key}")

    # Опциональные
    for mod_key, (label, _) in OPTIONAL_MODULES.items():
        st.checkbox(
            label,
            value=st.session_state.module_states.get(mod_key, False),
            key=f"mod_{mod_key}",
            on_change=lambda k=mod_key: st.session_state.module_states.update(
                {k: st.session_state[f"mod_{k}"]}
            ),
        )

    # Проверка зависимостей
    active_optional = [
        k for k in OPTIONAL_MODULES
        if st.session_state.get(f"mod_{k}", st.session_state.module_states.get(k, False))
    ]
    active_all = BASE_MODULES + active_optional
    n_active = len(active_all)

    for mod_key in active_optional:
        deps = DEPENDENCIES.get(mod_key, [])
        missing = [d for d in deps if d not in active_all]
        if missing:
            missing_labels = [OPTIONAL_MODULES[d][0] if d in OPTIONAL_MODULES else d for d in missing]
            st.warning(f"⚠️ {OPTIONAL_MODULES[mod_key][0]} требует: {', '.join(missing_labels)}")

    st.caption(f"Активно: **{n_active}/{TOTAL_MODULES}** модулей")

    # ── Пресеты ──
    st.divider()
    st.subheader("💾 Пресеты")

    preset_col1, preset_col2 = st.columns(2)

    with preset_col1:
        preset_name_input = st.text_input("Название пресета", key="preset_name", placeholder="напр. плазма-мин")

    with preset_col2:
        if st.button("💾 Сохранить"):
            if preset_name_input.strip():
                st.session_state.presets[preset_name_input.strip()] = dict(st.session_state.module_states)
                st.success(f"Сохранён: {preset_name_input.strip()}")
            else:
                st.warning("Введите название")

    if st.session_state.presets:
        preset_names = list(st.session_state.presets.keys())
        selected_preset = st.selectbox("Загрузить пресет", [""] + preset_names)
        if st.button("📂 Загрузить") and selected_preset:
            saved = st.session_state.presets[selected_preset]
            st.session_state.module_states.update(saved)
            for k in OPTIONAL_MODULES:
                st.session_state[f"mod_{k}"] = saved.get(k, False)
            st.success(f"Загружен: {selected_preset}")
            st.rerun()

    st.divider()

    # ── Группа 3: Параметры активных модулей ──
    st.subheader("📐 Параметры")

    # Геометрия (всегда)
    with st.expander("Геометрия и материалы", expanded=True):
        th_cfg = base_config.get("thermal_monitor", {})
        R_mm = st.slider("Радиус сферы, мм", 10, 100, int(th_cfg.get("R", 0.05) * 1000), step=1)
        h_mm = st.slider("Толщина стенки, мм", 0.5, 10.0, float(th_cfg.get("h", 0.002)) * 1000, step=0.5)
        T_inner = st.number_input("T внутр., K", value=float(th_cfg.get("T_inner", 310.0)))
        T_outer = st.number_input("T внеш., K", value=float(th_cfg.get("T_outer", 293.0)))
        k_thermal = st.number_input("Теплопроводность, Вт/(м·K)", value=float(th_cfg.get("k_thermal", 15.0)))

    # Плазма (всегда)
    with st.expander("🔥 Плазма", expanded=True):
        pl_cfg = base_config.get("plasma_monitor", {})
        pl_radius = st.slider("Радиус плазмы, мм", 1, 50, int(pl_cfg.get("radius", 0.01) * 1000), step=1)
        T_plasma_init = st.number_input("Начальная T плазмы, K", value=float(pl_cfg.get("temperature", 5e6)), format="%.0f")
        n_density = st.number_input("Плотность, м⁻³", value=float(pl_cfg.get("density_number", 1e20)), format="%.2e")
        fuel_type = st.selectbox("Топливо", ["D-He3", "D-T", "D-D"], index=["D-He3", "D-T", "D-D"].index(pl_cfg.get("fuel_type", "D-He3")))

    # Осмос
    if "osmosis" in active_all:
        with st.expander("💧 Осмос", expanded=False):
            osm_cfg = base_config.get("osmosis_monitor", {})
            concentration = st.number_input("Концентрация, моль/м³", value=float(osm_cfg.get("concentration", 300.0)))
            membrane_thickness = st.number_input("Толщина мембраны, м", value=float(osm_cfg.get("membrane_thickness", 1e-9)), format="%.1e")

    # Вакуум
    if "vacuum" in active_all:
        with st.expander("🌀 Вакуум", expanded=False):
            vac_cfg = base_config.get("vacuum_monitor", {})
            P_initial = st.number_input("Давление старт, Па", value=float(vac_cfg.get("P_initial", 1e5)), format="%.2e")
            pump_speed = st.number_input("Скорость откачки, м³/с", value=float(vac_cfg.get("pump_speed", 0.01)), format="%.4f")

    # Квантовый вакуум
    if "quantum" in active_all:
        with st.expander("⚡ Квантовый вакуум", expanded=False):
            qvac_cfg = base_config.get("quantum_vacuum", {})
            E_field = st.number_input("E-поле, В/м", value=float(qvac_cfg.get("E_field", 1e18)), format="%.2e")
            gap_nm = st.slider("Зазор, нм", 1, 50, int(qvac_cfg.get("gap", 3.5e-9) * 1e9), step=1)

    # Био
    if "bio" in active_all:
        with st.expander("🧠 Био-мост", expanded=False):
            bio_cfg = base_config.get("bio_bridge", {})
            n_neurons = st.number_input("Нейроны", value=int(bio_cfg.get("n_neurons", 100)))
            firing_rate = st.number_input("Частота спайков, Гц", value=float(bio_cfg.get("firing_rate", 10.0)))

    # Акустика
    if "acoustic" in active_all:
        with st.expander("🔊 Акустика", expanded=False):
            ac_cfg = base_config.get("acoustic_monitor", {})
            rho_acoustic = st.number_input("Плотность, кг/м³", value=float(ac_cfg.get("rho", 2700.0)))
            E_acoustic = st.number_input("Модуль Юнга, Па", value=float(ac_cfg.get("E", 7e10)), format="%.2e")

    st.divider()
    save_button = st.button("💾 Сохранить результаты", type="secondary")


# ── Сборка конфига ────────────────────────────────────────────────────
def build_config():
    cfg = dict(base_config)
    R_m = R_mm / 1000
    h_m = h_mm / 1000

    for mod_key in ("thermal_monitor", "acoustic_monitor", "born_collapse_monitor", "magnon_monitor", "em_resonance_monitor"):
        mod = dict(base_config.get(mod_key, {}))
        mod["R"] = R_m
        mod["h"] = h_m
        cfg[mod_key] = mod

    cfg["thermal_monitor"]["T_inner"] = T_inner
    cfg["thermal_monitor"]["T_outer"] = T_outer
    cfg["thermal_monitor"]["k_thermal"] = k_thermal

    cfg["plasma_monitor"] = {**base_config.get("plasma_monitor", {}),
                             "radius": pl_radius / 1000,
                             "temperature": T_plasma_init,
                             "density_number": n_density,
                             "fuel_type": fuel_type}

    if "osmosis" in active_all:
        cfg["osmosis_monitor"] = {**base_config.get("osmosis_monitor", {}),
                                  "concentration": concentration,
                                  "membrane_thickness": membrane_thickness}

    if "vacuum" in active_all:
        cfg["vacuum_monitor"] = {**base_config.get("vacuum_monitor", {}),
                                 "P_initial": P_initial,
                                 "pump_speed": pump_speed}

    if "quantum" in active_all:
        cfg["quantum_vacuum"] = {**base_config.get("quantum_vacuum", {}),
                                 "E_field": E_field,
                                 "gap": gap_nm * 1e-9}

    if "bio" in active_all:
        cfg["bio_bridge"] = {**base_config.get("bio_bridge", {}),
                              "n_neurons": n_neurons,
                              "firing_rate": firing_rate}

    if "acoustic" in active_all:
        cfg["acoustic_monitor"] = {**base_config.get("acoustic_monitor", {}),
                                    "R": R_m, "h": h_m,
                                    "rho": rho_acoustic, "E": E_acoustic}

    cfg["simulation"] = {"n_steps": n_steps, "dt": dt}
    cfg["_active_modules"] = active_all
    return cfg


# ── Прогон цепочки ────────────────────────────────────────────────────
def run_chain(cfg):
    t0 = time.time()
    sim_n = cfg["simulation"]["n_steps"]
    sim_dt = cfg["simulation"]["dt"]
    active = cfg.get("_active_modules", BASE_MODULES + list(OPTIONAL_MODULES.keys()))

    results = {}
    total_stress = 0.0

    # ── Осмос ──
    if "osmosis" in active:
        osmosis = OsmosisMonitor(cfg.get("osmosis_monitor", {}))
        osmosis.init()
        osmosis.run()
        results["osmosis"] = osmosis.get_results()
        total_stress += osmosis.get_stress_Pa()

    # ── Термалка (базовый) ──
    thermal = ThermalMonitor(cfg.get("thermal_monitor", {}))
    thermal.init()
    thermal.run()
    results["thermal"] = thermal.get_results()
    total_stress += thermal.get_thermal_stress_Pa()

    # ── Акустика ──
    if "acoustic" in active:
        acoustic = AcousticMonitor(cfg.get("acoustic_monitor", {}))
        acoustic.set_internal_stress(total_stress)
        acoustic.init()
        acoustic.run()
        results["acoustic"] = acoustic.get_results()

    # ── Коллапс ──
    if "collapse" in active and "osmosis" in results:
        collapse = BornCollapseMonitor(cfg.get("born_collapse_monitor", {}))
        collapse.set_external_pressure(results["osmosis"].get("pi_Pa", 0.0))
        collapse.init()
        collapse.run()
        results["collapse"] = collapse.get_results()

    # ── Магноны ──
    if "magnon" in active:
        magnon = MagnonMonitor(cfg.get("magnon_monitor", {}))
        magnon.set_mechanical_stress(total_stress)
        magnon.init()
        magnon.run()
        results["magnon"] = magnon.get_results()

    # ── EM-резонанс ──
    if "em" in active:
        em = EMResonanceMonitor(cfg.get("em_resonance_monitor", {}))
        em.set_mechanical_stress(total_stress)
        em.init()
        em.run()
        results["em"] = em.get_results()

    # ── Плазма (базовый) ──
    plasma = PlasmaMonitor(cfg.get("plasma_monitor", {}))
    plasma.init()
    plasma.run()
    results["plasma"] = plasma.get_results()

    # ── Вакуум ──
    if "vacuum" in active:
        vacuum = VacuumMonitor(cfg.get("vacuum_monitor", {}))
        vacuum.init()
        vacuum.run(external={})
        results["vacuum"] = vacuum.get_results()

    # ── Квантовый вакуум ──
    if "quantum" in active:
        q_vacuum = GAKQuantumVacuum(cfg.get("quantum_vacuum", {}))
        q_vacuum.run()
        results["quantum"] = q_vacuum.get_results()

    # ── Био ──
    if "bio" in active:
        bio = BioBridge(cfg.get("bio_bridge", {}))
        bio.init()
        bio.run()
        results["bio"] = bio.get_results()

    V_plasma = (4.0 / 3.0) * np.pi * (plasma.radius ** 3)
    pair_energy_gain = plasma.pair_energy_gain

    # ── Цикл по времени ──
    history = {
        "step": [], "T_plasma": [], "P_gas": [], "pair_rate": [],
        "P_cond": [], "P_heat_d": [], "ratio": [], "bio_activity": [],
        "osm_pressure": [], "T_inner_wall": [],
    }

    stride = max(1, sim_n // 500)

    for step_i in range(sim_n):
        step_result = plasma.step(sim_dt)
        pl_results = plasma.get_results()
        T_plasma = pl_results.get("temperature_plasma", 0.0)
        T_wall = pl_results.get("T_wall", T_inner)

        # Вакуум
        if "vacuum" in active:
            vacuum.step(sim_dt, Q_in=0.0, S_pump=vacuum.pump_speed)
            vacuum.run(external={})
            vac_results = vacuum.get_results()
            P_gas = vac_results.get("pressure_gas", 0.0)
        else:
            P_gas = 0.0

        # Квантовый вакуум
        if "quantum" in active:
            q_vacuum.step(sim_dt, T_plasma=T_plasma)
            q_results = q_vacuum.run()
            pair_rate = q_results.get("pair_rate", 0.0)
            plasma.run(external_stress={"pair_rate": pair_rate})
        else:
            pair_rate = 0.0

        # Био
        if "bio" in active:
            bio.step(sim_dt, neuron_activity=None, T_plasma=T_plasma, P_gas=P_gas)
            bio.run()
            bio_activity = bio.get_results().get("activity", 0.0)
        else:
            bio_activity = 0.0

        if step_i % stride == 0 or step_i == sim_n - 1:
            P_cond = step_result.get("P_cond", 0.0)
            P_heat = pair_rate * pair_energy_gain
            P_heat_d = P_heat / V_plasma if V_plasma > 0 else 0.0
            ratio = P_heat_d / P_cond if P_cond > 0 else 0.0
            osm_p = results.get("osmosis", {}).get("pi_Pa", 0.0)

            history["step"].append(step_i)
            history["T_plasma"].append(T_plasma)
            history["P_gas"].append(P_gas)
            history["pair_rate"].append(pair_rate)
            history["P_cond"].append(P_cond)
            history["P_heat_d"].append(P_heat_d)
            history["ratio"].append(ratio)
            history["bio_activity"].append(bio_activity)
            history["osm_pressure"].append(osm_p)
            history["T_inner_wall"].append(T_wall)

    # ── Coupling ──
    coupling = CouplingMonitor()
    coupling.init()
    coupling_kwargs = {}
    if "osmosis" in results:
        coupling_kwargs["osmosis"] = results["osmosis"]
    coupling_kwargs["thermal"] = results.get("thermal", {})
    if "acoustic" in results:
        coupling_kwargs["acoustic"] = results["acoustic"]
    if "collapse" in results:
        coupling_kwargs["collapse"] = results["collapse"]
    if "magnon" in results:
        coupling_kwargs["magnon"] = results["magnon"]
    if "em" in results:
        coupling_kwargs["em"] = results["em"]
    coupling_kwargs["plasma"] = results.get("plasma", {})
    if "vacuum" in results:
        coupling_kwargs["vacuum"] = results["vacuum"]
    if "bio" in results:
        coupling_kwargs["bio"] = results["bio"]
    if "quantum" in results:
        coupling_kwargs["quantum"] = results["quantum"]
    coupling.set_results(**coupling_kwargs)
    coupling.run()
    results["coupling"] = coupling.get_results()

    elapsed = time.time() - t0
    results["history"] = history
    results["elapsed"] = elapsed
    results["total_stress"] = total_stress
    results["active_modules"] = active

    return results


# ── Функция сохранения ───────────────────────────────────────────────
def save_results(results, cfg):
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(results_dir, f"gak_wavecad_log_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"GAK-WaveCAD v{VERSION} — Результаты симуляции\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Профиль: {profile_name} ({accuracy})\n")
        f.write(f"Время расчёта: {results['elapsed']:.2f} с\n")
        f.write(f"Шагов: {n_steps} | dt: {dt:.0e}\n")
        f.write(f"Активные модули: {', '.join(results.get('active_modules', []))}\n")
        f.write(f"Σ-напряжение: {results['total_stress']:.0f} Па\n")
        f.write("=" * 60 + "\n\n")
        for key, data in results.items():
            if key in ("history", "elapsed", "total_stress", "active_modules"):
                continue
            f.write(f"--- {key} ---\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        f.write(f"  {k}: {float(v):.6e}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            f.write("\n")

    csv_path = os.path.join(results_dir, f"gak_wavecad_history_{timestamp}.csv")
    pd.DataFrame(results["history"]).to_csv(csv_path, index=False, encoding="utf-8")

    json_path = os.path.join(results_dir, f"gak_wavecad_config_{timestamp}.json")
    cfg_serializable = {}
    for k, v in cfg.items():
        if k == "_active_modules":
            cfg_serializable[k] = v
        elif isinstance(v, dict):
            cfg_serializable[k] = {
                kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else str(vv)
                for kk, vv in v.items()
            }
        else:
            cfg_serializable[k] = str(v)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cfg_serializable, f, indent=2, ensure_ascii=False)

    return [os.path.basename(log_path), os.path.basename(csv_path), os.path.basename(json_path)]


# ── Вспомогательные функции ──────────────────────────────────────────
def _safe_float(v):
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return v

def _show_json(label, data):
    st.markdown(f"**{label}**")
    if isinstance(data, dict):
        clean = {k: _safe_float(v) for k, v in data.items() if k != "history"}
        st.json(clean)
    else:
        st.write(str(data))


# ── Вкладки ──────────────────────────────────────────────────────────
tab_sim, tab_opt, tab_save, tab_about, tab_val = st.tabs([
    "🚀 Симуляция",
    "🎯 Оптимизация",
    "💾 Результаты",
    "ℹ️ О проекте",
    "🧪 Валидация",
])

# ═══════════════════════════════════════════════════════════════════════
# Вкладка: Симуляция
# ═══════════════════════════════════════════════════════════════════════
with tab_sim:
    PLOT_MODES = {
        "plasma":   "🔥 Плазма — температура",
        "vacuum":   "🌀 Вакуум — давление газа",
        "quantum":  "⚡ Квантовый вакуум — рождение пар",
        "bio":      "🧠 Био — активность нейросети",
        "thermal":  "🌡️ Термалка — температура стенки",
        "osmosis":  "💧 Осмос — осмотическое давление",
        "ratio":    "⚖️ Баланс тепла (Швингер/проводимость)",
    }

    available_plots = {k: v for k, v in PLOT_MODES.items() if k in ["plasma", "ratio"] or k in active_all}

    selected_plot = st.selectbox(
        "📈 Выберите график",
        options=list(available_plots.keys()),
        format_func=lambda x: available_plots[x],
        index=0,
    )

    with st.spinner("⚡ Расчёт цепочки сред..."):
        r = run_chain(build_config())

    # Метрики
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Время расчёта", f"{r['elapsed']:.2f} с")
    col2.metric("Шагов", f"{n_steps:,}")
    col3.metric("Топливо", fuel_type)
    col4.metric("Σ-напряж. (Па)", f"{r['total_stress']:.0f}")
    col5.metric("Модулей", f"{len(r.get('active_modules', []))}/{TOTAL_MODULES}")

    st.divider()
    st.subheader("📈 Эволюция во времени")
    hist = r["history"]
    df_hist = pd.DataFrame(hist)

    col_a, col_b = st.columns(2)

    with col_a:
        if selected_plot == "plasma":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(df_hist["step"], df_hist["T_plasma"], color="crimson", linewidth=2)
            ax.set_title("Температура плазмы")
            ax.set_ylabel("T, K")
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

        elif selected_plot == "thermal":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_hist["step"], df_hist["T_inner_wall"], color="orange", linewidth=2)
            ax.set_title("Температура внутренней стенки")
            ax.set_ylabel("T, K")
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

        elif selected_plot == "osmosis":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_hist["step"], df_hist["osm_pressure"], color="teal", linewidth=2)
            ax.set_title("Осмотическое давление")
            ax.set_ylabel("π, Па")
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

        elif selected_plot == "ratio":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_hist["step"], df_hist["ratio"], color="purple", linewidth=2)
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="равновесие")
            ax.set_title("Баланс тепла")
            ax.set_ylabel("ratio")
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

    with col_b:
        if selected_plot == "vacuum":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(df_hist["step"], df_hist["P_gas"], color="navy", linewidth=2)
            ax.set_title("Давление газа")
            ax.set_ylabel("P, Па")
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

        elif selected_plot == "quantum":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(df_hist["step"], df_hist["pair_rate"], color="darkgreen", linewidth=2)
            ax.set_title("Рождение пар (Швингер)")
            ax.set_ylabel("пары/с")
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

        elif selected_plot == "bio":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_hist["step"], df_hist["bio_activity"], color="purple", linewidth=2)
            ax.set_title("Активность био-нейросети")
            ax.set_ylabel("активность")
            ax.grid(True, alpha=0.3, linestyle="--")
            st.pyplot(fig); plt.close(fig)

        else:
            st.info("Выберите режим слева для графика. Здесь будет второй график при выборе вакуума, квантов или био.")

    st.divider()
    st.subheader("🗂️ Результаты по модулям")

    col_mod1, col_mod2 = st.columns(2)

    with col_mod1:
        if "osmosis" in r:
            _show_json("💧 Осмос", r["osmosis"])
        _show_json("🌡️ Термалка", r["thermal"])
        if "acoustic" in r:
            st.markdown("**🔊 Акустика (таблица мод)**")
            ac = r["acoustic"]
            if isinstance(ac, dict):
                ac_df = pd.DataFrame([
                    {"mode": k, "f_ref": v.get("f_reference", 0), "f_meas": v.get("f_measured", 0),
                     "dF": v.get("delta_f", 0), "status": v.get("status", "?")}
                    for k, v in ac.items() if isinstance(v, dict)
                ])
                st.dataframe(ac_df, use_container_width=True, hide_index=True)
        if "collapse" in r:
            _show_json("🕳️ Коллапс", r["collapse"])
        if "magnon" in r:
            _show_json("🧲 Магноны", r["magnon"])

    with col_mod2:
        if "em" in r:
            _show_json("📡 EM-резонанс", r["em"])
        _show_json("🔥 Плазма", r["plasma"])
        if "vacuum" in r:
            _show_json("🌀 Вакуум", r["vacuum"])
        if "quantum" in r:
            _show_json("⚡ Квантовый вакуум", r["quantum"])
        if "bio" in r:
            _show_json("🧠 Био-мост", r["bio"])

    st.divider()
    st.subheader("🔗 Кросс-связи")
    _show_json("CouplingMonitor", r.get("coupling", {}))


# ═══════════════════════════════════════════════════════════════════════
# Вкладка: Оптимизация
# ═══════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("🎯 Оптимизация параметров (Optuna)")

    if not OPTUNA_AVAILABLE:
        st.error("Модуль optuna_optimizer не найден. Убедитесь, что файл optuna_optimizer.py лежит в корне проекта.")
    else:
        opt_mode = st.radio(
            "Режим оптимизации",
            ("🔥 Зажигание", "🔋 Усиление", "📊 Парето"),
            index=0,
            horizontal=True,
        )
        mode_key = {"🔥 Зажигание": "ignition", "🔋 Усиление": "gain", "📊 Парето": "pareto"}[opt_mode]

        if mode_key == "ignition":
            st.info("Зажигание — максимизация критерия Лоусона: n·T·τ_E → max")
        elif mode_key == "gain":
            st.info("Усиление — максимум Q-фактора: Q = P_fusion / P_cond → max")
        else:
            st.info("Парето — многокритериальный фронт: pair_rate → max при отклонении от порога → min")

        n_trials = st.slider("Количество попыток", 10, 200, 50, step=10)

        if st.button("🎯 Подобрать параметры", type="primary"):
            with st.spinner(f"Optuna ({mode_key}, {n_trials} попыток)..."):
                optimizer = GAKOptunaOptimizer(base_config)
                optimizer.optimize(mode=mode_key, n_trials=n_trials)
            st.success("✅ Оптимизация завершена!")
            st.markdown("### Лучшие параметры")
            summary_df = optimizer.get_results_summary()
            if summary_df is not None:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.markdown("### График сходимости")
            fig, ax = plt.subplots(figsize=(8, 4))
            optimizer.plot_convergence(fig=fig, ax=ax)
            st.pyplot(fig); plt.close(fig)
            export_df = optimizer.export_best_csv()
            if export_df is not None:
                st.dataframe(export_df, use_container_width=True, hide_index=True)
                csv_data = export_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Скачать CSV", data=csv_data,
                                  file_name=f"optuna_{mode_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                  mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════
# Вкладка: Результаты
# ═══════════════════════════════════════════════════════════════════════
with tab_save:
    st.subheader("💾 Сохранение результатов")

    if save_button:
        try:
            saved_files = save_results(r, build_config())
            st.success("✅ Сохранено в папку results/:")
            for fname in saved_files:
                st.write(f"• `{fname}`")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("Нажмите «💾 Сохранить результаты» в сайдбаре")

    results_dir = os.path.join(SCRIPT_DIR, "results")
    if os.path.exists(results_dir):
        existing = sorted(os.listdir(results_dir), reverse=True)
        if existing:
            st.markdown("### Сохранённые файлы")
            for fname in existing[:20]:
                fsize = os.path.getsize(os.path.join(results_dir, fname))
                st.write(f"• `{fname}` ({fsize:,} байт)")


# ═══════════════════════════════════════════════════════════════════════
# Вкладка: О проекте
# ═══════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"## 🌊 GAK-WaveCAD v{VERSION}")
    st.markdown("Мультифизический САПР для моделирования цепочки физических сред.")
    st.markdown("")
    st.markdown("**Модули:**")
    st.markdown("- 🌡️ Термалка — теплопроводность оболочки (базовый)")
    st.markdown("- 🔥 Плазма — термоядерные реакции (базовый)")
    st.markdown("- 💧 Осмос — ионный транспорт через мембрану")
    st.markdown("- 🔊 Акустика — моды сферической оболочки")
    st.markdown("- 🕳️ Коллапс — гравитационный коллапс (Борн)")
    st.markdown("- 🧲 Магноны — спинтронные моды")
    st.markdown("- 📡 EM-резонанс — электромагнитные моды")
    st.markdown("- 🌀 Вакуум — откачка и газовыделение")
    st.markdown("- ⚡ Квантовый вакуум — рождение пар (Швингер)")
    st.markdown("- 🧠 Био-мост — нейросетевая активность")
    st.markdown("")
    st.markdown("**Технологии:** Python + Streamlit, NumPy, Matplotlib, Optuna")
    st.markdown("")
    st.markdown("---")
    st.markdown("🌊 Суверенное ядро GAK-WaveCAD")

# ═══════════════════════════════════════════════════════════════════════
# Вкладка: Валидация
# ═══════════════════════════════════════════════════════════════════════
with tab_val:
    try:
        from core.validation import render_validation_tab
        render_validation_tab()
    except ImportError as e:
        st.error(f"❌ Модуль валидации не найден: {e}")
        st.info("Положите validation.py в папку core/")
    except Exception as e:
        st.error(f"❌ Ошибка валидации: {e}")
        st.code(traceback.format_exc())
