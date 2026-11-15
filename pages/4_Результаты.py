"""
Сохранённые результаты и экспорт.
"""
import streamlit as st
import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.session_state import init_session_state

st.set_page_config(page_title="Результаты", page_icon="💾", layout="wide")
init_session_state()

st.title("💾 Результаты")

# ── Последний результат ──
r = st.session_state.get("last_result")
if r:
    st.markdown("### Последний прогон")
    st.json({k: v for k, v in r.items() if k != "history"})

# ── Сохранённые прогонами ──
st.markdown("---")
st.markdown("### Сохранённые прогонами")
script_dir = st.session_state.get("script_dir", ".")
results_dir = os.path.join(script_dir, "results")

if os.path.exists(results_dir):
    csvs = sorted(glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True))
    if csvs:
        for csv_path in csvs[-20:]:  # последние 20
            with st.expander(os.path.relpath(csv_path, results_dir)):
                df = pd.read_csv(csv_path)
                st.dataframe(df)
    else:
        st.info("Нет сохранённых результатов")
else:
    st.info("Папка results/ не найдена")
