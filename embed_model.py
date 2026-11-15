"""
embed_model.py — вшивает обученную модель в ai_transformer_predictor.py
Запускать из корневой папки GAK-WaveCAD: python embed_model.py
"""

import json
import zlib
import base64
import os

# Пути
MODEL_PATH = os.path.join("models", "ai_predictor_DT.json")
PREDICTOR_PATH = os.path.join("physical_modules", "ai_transformer_predictor.py")
OUTPUT_PATH = os.path.join("physical_modules", "ai_transformer_predictor_v12.py")

# 1. Загружаем модель
print(f"Loading model from {MODEL_PATH}...")
with open(MODEL_PATH, "r") as f:
    full_data = json.load(f)

print(f"  is_trained={full_data.get('is_trained')}, "
      f"exp={full_data.get('n_experiences')}, "
      f"err={full_data.get('last_error_pct', 0):.2f}%")

# 2. Извлекаем только веса + нормализацию (без experience)
slim = {}
for key in ["_model_type", "_model_version",
            "n_features", "n_targets",
            "hidden1", "hidden2", "hidden3",
            "lr", "max_grad_norm",
            "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4",
            "feature_mean", "feature_std",
            "target_mean", "target_std",
            "is_trained", "n_experiences",
            "last_error_pct", "last_error_linear_pct",
            "_train_steps"]:
    if key in full_data:
        slim[key] = full_data[key]

# Принудительно обнуляем experience — AutoLearn накопит заново
slim["experience_configs"] = []
slim["experience_targets"] = []

# 3. Сжимаем
slim_json = json.dumps(slim, separators=(",", ":"))
full_kb = len(json.dumps(full_data, separators=(",", ":"))) / 1024
slim_kb = len(slim_json) / 1024

compressed = zlib.compress(slim_json.encode("utf-8"), level=9)
b64 = base64.b64encode(compressed).decode("ascii")
b64_kb = len(b64) / 1024

print(f"  Full: {full_kb:.1f} KB, Slim: {slim_kb:.1f} KB, B64: {b64_kb:.1f} KB")

# 4. Читаем исходный файл
print(f"Reading {PREDICTOR_PATH}...")
with open(PREDICTOR_PATH, "r", encoding="utf-8") as f:
    src = f.read()

# 5. Вставляем _DEFAULT_MODEL_B64 перед классом
b64_block = '_DEFAULT_MODEL_B64 = "' + b64 + '"\n\n\n'

class_marker = "class GAKTransformerPredictor:"
idx = src.find(class_marker)
if idx == -1:
    print("ERROR: class GAKTransformerPredictor not found!")
    exit(1)

patched = src[:idx] + b64_block + src[idx:]

# 6. Добавляем load_default() после load()
load_default_method = (
    "\n"
    "    @classmethod\n"
    "    def load_default(cls):\n"
    "        \"\"\"Загрузка заводской модели, вшитой в код.\"\"\"\n"
    "        import json\n"
    "        obj = cls()\n"
    "        try:\n"
    "            raw = zlib.decompress(base64.b64decode(_DEFAULT_MODEL_B64))\n"
    "            data = json.loads(raw.decode('utf-8'))\n"
    "            for attr in ['n_features', 'n_targets', 'hidden1', 'hidden2', 'hidden3',\n"
    "                          'lr', 'max_grad_norm',\n"
    "                          'is_trained', 'n_experiences',\n"
    "                          'last_error_pct', 'last_error_linear_pct', '_train_steps']:\n"
    "                if attr in data:\n"
    "                    setattr(obj, attr, data[attr])\n"
    "            for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4',\n"
    "                          'feature_mean', 'feature_std', 'target_mean', 'target_std']:\n"
    "                if attr in data:\n"
    "                    setattr(obj, np.array(data[attr]))\n"
    "            obj.experience_configs = []\n"
    "            obj.experience_targets = []\n"
    "            obj._error_history = []\n"
    "            obj.n_predictions = 0\n"
    "            print('[AI] Default model loaded (factory preset)')\n"
    "            return obj\n"
    "        except Exception as e:\n"
    "            print(f'[AI] Failed to load default model: {e}')\n"
    "            return obj\n"
    "\n"
)

# Находим конец load() — последний "return True" в файле
last_return = src.rfind("        return True\n")
if last_return == -1:
    print("ERROR: end of load() not found!")
    exit(1)

end_of_load = last_return + len("        return True\n")
insert_idx = end_of_load + len(b64_block)
patched = patched[:insert_idx] + load_default_method + patched[insert_idx:]

# 7. Добавляем import zlib, base64 если их нет
if "import zlib" not in patched:
    patched = patched.replace("import os\n", "import os\nimport zlib\nimport base64\n", 1)

# 8. Сохраняем
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(patched)

print(f"\nDone! Saved as {OUTPUT_PATH}")
print(f"  Model: is_trained={slim['is_trained']}, exp={slim['n_experiences']}, err={slim.get('last_error_pct', 0):.2f}%")
print(f"  Embedded size: {b64_kb:.1f} KB")
print(f"  Experience: stripped (AutoLearn will re-accumulate)")
print(f"\nReplace physical_modules/ai_transformer_predictor.py with ai_transformer_predictor_v12.py")
