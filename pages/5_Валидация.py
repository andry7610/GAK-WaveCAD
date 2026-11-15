"""
Валидация: 62 теста, сравнение с аналитикой.
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.session_state import init_session_state

st.set_page_config(page_title="Валидация", page_icon="✅", layout="wide")
init_session_state()

st.title("✅ Валидация")

st.markdown("### ИИ-валидация трансформера")
st.info("Сравнение прогноза ИИ с реальной симуляцией на данных, не использованных при обучении.")

try:
    from core.ai_validation import render_ai_validation_tab
    render_ai_validation_tab()
except ImportError as e:
    st.error(f"Модуль ai_validation не найден: {e}")
except Exception as e:
    st.error(f"Ошибка: {e}")
    import traceback
    st.code(traceback.format_exc())
