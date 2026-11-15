"""
Страница остановки прогона оптимизации GAK-WaveCAD.
Ставит файл-флаг models/STOP_FLAG — оптимизатор проверяет его между триалами
и аккуратно завершает прогон, сохранив все завершённые триалы в базу.
v1: первоначальная версия.
"""

import os
import streamlit as st

st.set_page_config(page_title="Стоп прогона", page_icon="🛑", layout="centered")

FLAG_NAME = "STOP_FLAG"


def _find_models_dir():
    """Ищем папку models: рядом с pages/ или в текущей рабочей директории."""
    candidates = []
    # От этой страницы: pages/ -> корень проекта
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(root, "models"))
    except NameError:
        pass
    # Текущая рабочая директория (Streamlit обычно стартует из корня проекта)
    candidates.append(os.path.join(os.getcwd(), "models"))
    for c in candidates:
        if os.path.isdir(c):
            return c
    # Ничего не нашли — берём первый кандидат (создадим при установке флага)
    return candidates[0] if candidates else "models"


MODELS_DIR = _find_models_dir()
FLAG_PATH = os.path.join(MODELS_DIR, FLAG_NAME)

st.title("🛑 Остановка прогона оптимизации")

st.markdown(
    "Ставит файл-флаг `models/STOP_FLAG`. Оптимизатор проверяет его **между триалами** "
    "и аккуратно завершает прогон: все уже завершённые триалы остаются в базе, "
    "потеряется максимум текущий (около минуты счёта)."
)

st.caption(f"Файл-флаг: `{FLAG_PATH}`")

col1, col2 = st.columns(2)

with col1:
    if st.button("🛑 Остановить прогон", type="primary", use_container_width=True):
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(FLAG_PATH, "w", encoding="utf-8") as f:
                f.write("stop requested")
            st.success("Флаг установлен. Прогон остановится после текущего триала (до ~70 секунд).")
        except Exception as e:
            st.error(f"Не удалось создать флаг: {e}")

with col2:
    if os.path.exists(FLAG_PATH):
        st.warning("Сейчас флаг остановки **активен** — новые прогоны остановятся сразу после первого триала.")
        if st.button("✅ Снять флаг", use_container_width=True):
            try:
                os.remove(FLAG_PATH)
                st.success("Флаг снят — прогоны снова идут до конца.")
            except Exception as e:
                st.error(f"Не удалось удалить флаг: {e}")
    else:
        st.info("Флаг не активен — прогоны идут без остановки.")

st.divider()
st.markdown(
    "**Памятка.** Нажимать кнопку лучше из второй вкладки браузера, пока в первой "
    "идёт прогон: Streamlit занят расчётом и не обрабатывает нажатия в той же вкладке. "
    "Все завершённые триалы сохраняются в базу автоматически."
)
