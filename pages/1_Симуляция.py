"""
Графики эволюции плазмы: T, P, ratio, Q по шагам.
"""
import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.session_state import init_session_state

st.set_page_config(page_title="Симуляция", page_icon="📊", layout="wide")
init_session_state()

st.title("📊 Графики симуляции")

history = st.session_state.get("last_history")

if history is None or not history.get("step"):
    st.info("Сначала запустите симуляцию на главной странице")
    st.stop()

steps = history["step"]

# ── T_plasma ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(steps, [t/1e6 for t in history["T_plasma"]], color="red")
axes[0,0].set_title("T_plasma, МК")
axes[0,0].set_ylabel("T, ×10⁶ K")
axes[0,0].grid(True, alpha=0.3)

# ── P_fusion, P_cond, P_sync ──
axes[0,1].plot(steps, [p/1e6 for p in history["P_fusion"]], label="P_fusion", color="green")
axes[0,1].plot(steps, [p/1e6 for p in history["P_cond"]], label="P_cond", color="blue")
axes[0,1].plot(steps, [p/1e6 for p in history["P_sync"]], label="P_sync", color="orange")
axes[0,1].set_title("Мощности, МВт")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# ── Q-фактор ──
axes[1,0].plot(steps, history["Q"], color="purple")
axes[1,0].set_title("Q-фактор")
axes[1,0].grid(True, alpha=0.3)

# ── ratio ──
axes[1,1].plot(steps, history["ratio"], color="teal")
axes[1,1].axhline(y=1.0, color="red", linestyle="--", label="ignition")
axes[1,1].set_title("ratio (P_heat / P_loss)")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()
