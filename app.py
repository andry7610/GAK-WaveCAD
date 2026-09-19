#!/usr/bin/env python3
"""
GAK-WaveCAD — Streamlit Web Interface
Интерактивная панель управления симуляцией.

Запуск:
    pip install streamlit
    streamlit run app.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.thermal_monitor import ThermalMonitor
from physical_modules.acoustic_monitor import AcousticMonitor
from physical_modules.born_collapse_monitor import BornCollapseMonitor
from physical_modules.magnon_monitor import MagnonMonitor
from physical_modules.em_resonance_monitor import EMResonanceMonitor
from physical_modules.coupling_monitor import CouplingMonitor


# ============================================================
# Конфигурация страницы
# ============================================================
st.set_page_config(
    page_title="GAK-WaveCAD",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 GAK-WaveCAD — Мультифизическая симуляция")
st.markdown("Осмос → Термалка → Акустика → Коллапс → Магноны → EM → Кросс-связи")

# ============================================================
# Боковая панель — параметры
# ============================================================
st.sidebar.header("⚙️ Параметры оболочки")

R = st.sidebar.slider("Радиус оболочки R, м", 0.01, 0.20, 0.05, 0.01)
h = st.sidebar.slider("Толщина стенки h, мм", 0.1, 5.0, 1.0, 0.1) / 1000
E = st.sidebar.slider("Модуль Юнга E, ГПа", 1.0, 200.0, 70.0, 1.0) * 1e9
nu = st.sidebar.slider("Коэффициент Пуассона ν", 0.1, 0.49, 0.3, 0.01)

st.sidebar.header("🧪 Химия")

C_in = st.sidebar.slider("Концентрация внутри C_in, моль/л", 0.01, 5.0, 1.0, 0.05)
C_out = st.sidebar.slider("Концентрация снаружи C_out, моль/л", 0.001, 2.0, 0.01, 0.01)

st.sidebar.header("🌡️ Температура")

T_inner = st.sidebar.slider("Т внутри, К", 200, 500, 310, 5)
T_outer = st.sidebar.slider("Т снаружи, К", 200, 500, 300, 5)

st.sidebar.header("🧲 Магноны (YIG)")

B0 = st.sidebar.slider("Поле B₀, Т", 0.05, 2.0, 0.3, 0.01)
lambda_s = st.sidebar.slider("Магнитострикция λ_s, ppm", 1.0, 50.0, 25.0, 1.0) * 1e-6

st.sidebar.header("⚡ EM-резонанс")

eps_r = st.sidebar.slider("ε_r (диэлектрик)", 1.0, 100.0, 10.0, 1.0)
d_piezo = st.sidebar.slider("d_piezo, пм/В", 10, 500, 220, 10) * 1e-12

# ============================================================
# Запуск симуляции
# ============================================================
# --- Осмос ---
osm_cfg = {'R': R, 'h': h, 'E': E, 'nu': nu, 'C_in': C_in, 'C_out': C_out}
osmosis = OsmosisMonitor(osm_cfg)
osmosis.init()
osmosis.run()
osm_stress = osmosis.get_stress_Pa()
osm_pressure = osmosis.results['pi_Pa']

# --- Термалка ---
th_cfg = {'R': R, 'h': h, 'E': E, 'nu': nu, 'T_inner': T_inner, 'T_outer': T_outer}
thermal = ThermalMonitor(th_cfg)
thermal.init()
thermal.run()
th_stress = thermal.get_thermal_stress_Pa()

total_stress = osm_stress + th_stress

# --- Акустика ---
ac_cfg = {'R': R, 'h': h, 'E': E, 'nu': nu}
acoustic = AcousticMonitor(ac_cfg)
acoustic.set_internal_stress(total_stress)
acoustic.init()
acoustic.run()
ac_results = acoustic.get_results()

# --- Коллапс ---
bc_cfg = {'R': R, 'h': h, 'E': E, 'nu': nu}
collapse = BornCollapseMonitor(bc_cfg)
collapse.set_external_pressure(osm_pressure)
collapse.init()
collapse.run()
coll_results = collapse.get_results()

# --- Магноны ---
mag_cfg = {'R': R, 'h': h, 'B0': B0, 'lambda_s': lambda_s}
magnon = MagnonMonitor(mag_cfg)
magnon.set_mechanical_stress(total_stress)
magnon.init()
magnon.run()
mag_results = magnon.get_results()

# --- EM ---
em_cfg = {'R': R, 'eps_r': eps_r, 'd_piezo': d_piezo}
em = EMResonanceMonitor(em_cfg)
em.set_mechanical_stress(total_stress)
em.init()
em.run()
em_results = em.get_results()

# --- Кросс-связи ---
coupling = CouplingMonitor()
coupling.init()
coupling.set_results(
    osmosis=osmosis.get_results(),
    thermal=thermal.get_results(),
    acoustic=ac_results,
    collapse=coll_results,
    magnon=mag_results,
    em=em_results,
)
coupling.run()
coupl = coupling.get_results()

# ============================================================
# Метрики — верхняя строка
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("σ суммарное", f"{total_stress / 1e6:.2f} МПа")
with col2:
    phase = coll_results['phase']
    phase_emoji = {"STABLE": "🟢", "WARNING": "🟡", "COLLAPSE": "🔴", "INFLATION": "🔵"}.get(phase, "❓")
    st.metric("Фаза коллапса", f"{phase_emoji} {phase}")
with col3:
    status = coupl['system_status']
    status_emoji = {"HEALTHY": "🟢", "DEGRADED": "🟡", "CRITICAL": "🔴"}.get(status, "❓")
    st.metric("Статус системы", f"{status_emoji} {status}")
with col4:
    st.metric("Stability Index", f"{coupl['stability_index']:.3f}")

st.divider()

# ============================================================
# Графики — две колонки
# ============================================================
col_left, col_right = st.columns(2)

# --- Акустика ---
with col_left:
    st.subheader("🔊 Акустические моды")
    fig, ax = plt.subplots(figsize=(7, 4))
    ac_labels = list(ac_results.keys())
    f0 = [r['f_reference'] for r in ac_results.values()]
    f1 = [r['f_measured'] for r in ac_results.values()]
    x = np.arange(len(ac_labels))
    w = 0.35
    ax.bar(x - w/2, f0, w, label='f₀', color='#4a90d9', alpha=0.8)
    ax.bar(x + w/2, f1, w, label='f (σ)', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in ac_labels], fontsize=8)
    ax.set_ylabel('Частота, Гц')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Таблица
    st.markdown("**Сдвиги частот:**")
    ac_data = []
    for key, r in ac_results.items():
        ac_data.append({
            'Мода': key,
            'f₀, Гц': f"{r['f_reference']:.1f}",
            'f, Гц': f"{r['f_measured']:.1f}",
            'Δf, Гц': f"{r['delta_f']:+.1f}",
        })
    st.table(ac_data)

# --- Магноны ---
with col_right:
    st.subheader("🧲 Магнонный спектр (YIG)")
    fig, ax = plt.subplots(figsize=(7, 4))
    mag_labels = list(mag_results.keys())
    f_mag = [r['f'] / 1e9 for r in mag_results.values()]
    f_mag_ns = [(r['f'] - r['df_stress']) / 1e9 for r in mag_results.values()]
    x = np.arange(len(mag_labels))
    ax.bar(x - w/2, f_mag_ns, w, label='f₀ (без σ)', color='#2ecc71', alpha=0.8)
    ax.bar(x + w/2, f_mag, w, label='f (с σ)', color='#e67e22', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in mag_labels], fontsize=8)
    ax.set_ylabel('Частота, ГГц')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Сдвиги и добротность:**")
    mag_data = []
    for key, r in mag_results.items():
        mag_data.append({
            'Мода': key,
            'f, ГГц': f"{r['f'] / 1e9:.3f}",
            'Δf, МГц': f"{r['df_stress'] / 1e6:+.1f}",
            'Q': f"{r['Q']:.0f}",
        })
    st.table(mag_data)

st.divider()

col_left2, col_right2 = st.columns(2)

# --- EM ---
with col_left2:
    st.subheader("⚡ EM-резонанс (TM/TE)")
    fig, ax = plt.subplots(figsize=(7, 4))
    em_labels = list(em_results.keys())
    f_em = [r['f_shifted'] / 1e9 for r in em_results.values()]
    f_em0 = [r['f_base'] / 1e9 for r in em_results.values()]
    x = np.arange(len(em_labels))
    ax.bar(x - w/2, f_em0, w, label='f₀ (без σ)', color='#9b59b6', alpha=0.8)
    ax.bar(x + w/2, f_em, w, label='f (с σ)', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in em_labels], fontsize=8)
    ax.set_ylabel('Частота, ГГц')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Пьезосдвиг:**")
    em_data = []
    for key, r in em_results.items():
        em_data.append({
            'Мода': key,
            'f, ГГц': f"{r['f_shifted'] / 1e9:.3f}",
            'Δf, МГц': f"{r['df_stress'] / 1e6:+.1f}",
            'Q': f"{r['Q']:.0f}",
        })
    st.table(em_data)

# --- Кросс-связи ---
with col_right2:
    st.subheader("🔗 Кросс-связи")
    fig, ax = plt.subplots(figsize=(7, 4))
    c_labels = ['mag-elastic\n(ак↔маг)', 'piezo\n(ак↔EM)', 'mag-electric\n(маг↔EM)', 'thermal-osm\n(т↔осм)']
    c_vals = [
        min(coupl['k_magnetoelastic'], 1.0),
        min(coupl['k_piezoelectric'], 1.0),
        min(coupl['k_magnetoelectric'], 1.0),
        min(coupl['k_thermal_osmosis'], 1.0),
    ]
    c_colors = ['#e74c3c', '#9b59b6', '#2ecc71', '#4a90d9']
    ax.barh(range(4), c_vals, color=c_colors, alpha=0.8, height=0.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(c_labels, fontsize=9)
    ax.set_xlabel('Коэффициент связи')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Параметры системы:**")
    st.markdown(f"""
    - σ_осмос: **{osm_stress / 1e6:.2f} МПа**
    - σ_термалка: **{th_stress / 1e6:.2f} МПа**
    - π (осмотическое давление): **{osm_pressure / 1e6:.3f} МПа**
    - P/P_cr (коллапс): **{coll_results['ratio']:.3f}**
    - geom_factor: **{coupl['geom_factor']:.3f}**
    """)

# ============================================================
# Нижняя секция — профиль напряжений
# ============================================================
st.divider()
st.subheader("📊 Профиль напряжений по цепочке")

chain_labels = ['σ_осмос', 'σ_терм', 'Σ σ', 'P/P_cr', 'k_mag-el', 'k_piezo', 'k_me', 'k_th-osm']
chain_vals = [
    osm_stress / 1e6,
    th_stress / 1e6,
    total_stress / 1e6,
    coll_results['ratio'],
    coupl['k_magnetoelastic'],
    coupl['k_piezoelectric'],
    coupl['k_magnetoelectric'],
    coupl['k_thermal_osmosis'],
]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#4a90d9', '#e67e22', '#e74c3c', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#3498db']
bars = ax.bar(range(len(chain_labels)), chain_vals, color=colors, alpha=0.8)
ax.set_xticks(range(len(chain_labels)))
ax.set_xticklabels(chain_labels, fontsize=9, rotation=30)
ax.set_ylabel('Значение')
ax.set_title('Все параметры цепочки на одном графике')
ax.grid(axis='y', alpha=0.3)

# Подписи значений
for bar, val in zip(bars, chain_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f'{val:.3f}', ha='center', fontsize=8)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ============================================================
# Экспандеры — детали
# ============================================================
with st.expander("📋 Детали — осмос и термалка"):
    st.json({
        'osmosis': osmosis.get_results(),
        'thermal': thermal.get_results(),
    })

with st.expander("📋 Детали — коллапс"):
    st.json(coll_results)

with st.expander("📋 Детали — кросс-связи"):
    st.json(coupl)
