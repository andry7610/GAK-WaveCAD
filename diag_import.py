"""
Диагностика импорта ai_validation.
Запустить: python diag_import.py
"""
import os, sys, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

print(f"Script dir: {SCRIPT_DIR}")
print(f"core/ exists: {os.path.isdir(os.path.join(SCRIPT_DIR, 'core'))}")
print(f"core/__init__.py exists: {os.path.exists(os.path.join(SCRIPT_DIR, 'core', '__init__.py'))}")
print(f"core/ai_validation.py exists: {os.path.exists(os.path.join(SCRIPT_DIR, 'core', 'ai_validation.py'))}")
print(f"core/ai_validation.py size: {os.path.getsize(os.path.join(SCRIPT_DIR, 'core', 'ai_validation.py')) if os.path.exists(os.path.join(SCRIPT_DIR, 'core', 'ai_validation.py')) else 'N/A'} bytes")
print()

# Пробуем импорт
print("=== Попытка импорта ===")
try:
    from core.ai_validation import render_ai_validation_tab
    print("✅ Импорт успешен!")
    print(f"   Функция: {render_ai_validation_tab}")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    traceback.print_exc()
except SyntaxError as e:
    print(f"❌ SyntaxError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
    traceback.print_exc()
