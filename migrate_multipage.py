"""
migrate_multipage.py — Миграция на multipage архитектуру.

Что делает:
1. Бэкапит старый app.py → app_backup_monolith.py
2. Копирует новые файлы из output/ в GAK-WaveCAD-main/
3. Создаёт папку pages/ с 6 страницами
4. Копирует core-модули в core/

Запуск: python migrate_multipage.py
"""

import os
import shutil

# Папки
SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT = SRC  # GAK-WaveCAD-main

print("=" * 60)
print("GAK-WaveCAD Multipage Migration")
print("=" * 60)

# 1. Бэкап app.py
app_path = os.path.join(PROJECT, "app.py")
backup_path = os.path.join(PROJECT, "app_backup_monolith.py")
if os.path.exists(app_path):
    shutil.copy2(app_path, backup_path)
    print(f"✅ Бэкап: app.py → app_backup_monolith.py")
else:
    print("⚠️ app.py не найден — пропускаем бэкап")

# 2. Копируем core-модули
core_files = [
    "session_state.py",
    "config_builder.py",
    "simulation.py",
    "save_results.py",
    "utils.py",
    "render_sidebar.py",
]

core_dir = os.path.join(PROJECT, "core")
os.makedirs(core_dir, exist_ok=True)

for fname in core_files:
    src = os.path.join(SRC, "output", "core", fname)
    dst = os.path.join(core_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ core/{fname}")
    else:
        print(f"❌ Не найден: output/core/{fname}")

# 3. Копируем pages
pages_dir = os.path.join(PROJECT, "pages")
os.makedirs(pages_dir, exist_ok=True)

page_files = [
    "1_Симуляция.py",
    "2_Оптимизация.py",
    "3_ИИ.py",
    "4_Результаты.py",
    "5_Валидация.py",
    "6_О_проекте.py",
]

for fname in page_files:
    src = os.path.join(SRC, "output", "pages", fname)
    dst = os.path.join(pages_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ pages/{fname}")
    else:
        print(f"❌ Не найден: output/pages/{fname}")

# 4. Копируем новый app.py
new_app = os.path.join(SRC, "output", "app.py")
if os.path.exists(new_app):
    shutil.copy2(new_app, app_path)
    print(f"✅ app.py (новый пульт)")
else:
    print(f"❌ Не найден: output/app.py")

print()
print("=" * 60)
print("✅ Миграция завершена!")
print()
print("Структура:")
print("  GAK-WaveCAD-main/")
print("  ├── app.py                    ← пульт (новый)")
print("  ├── app_backup_monolith.py    ← бэкап (старый)")
print("  ├── core/")
print("  │   ├── session_state.py")
print("  │   ├── config_builder.py")
print("  │   ├── simulation.py")
print("  │   ├── save_results.py")
print("  │   ├── utils.py")
print("  │   ├── render_sidebar.py")
print("  │   ├── config_loader.py      (существующий)")
print("  │   ├── validation.py          (существующий)")
print("  │   └── ai_validation.py       (существующий)")
print("  └── pages/")
print("      ├── 1_Симуляция.py")
print("      ├── 2_Оптимизация.py")
print("      ├── 3_ИИ.py")
print("      ├── 4_Результаты.py")
print("      ├── 5_Валидация.py")
print("      └── 6_О_проекте.py")
print()
print("Запуск: streamlit run app.py")
print("=" * 60)
