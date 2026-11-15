"""Тест вшитой заводской модели."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "physical_modules"))

from ai_transformer_predictor import GAKTransformerPredictor

# 1. Грузим заводскую модель (без файла на диске)
p = GAKTransformerPredictor.load_default()
if p is None:
    print("FAIL: load_default() вернул None")
    sys.exit(1)

print(f"Factory model loaded:")
print(f"  is_trained:   {p.is_trained}")
print(f"  n_experiences: {p.n_experiences}")
print(f"  last_error:    {p.last_error_pct:.2f}%")
print(f"  W1 shape:      {p.W1.shape}")
print(f"  W2 shape:      {p.W2.shape}")
print(f"  W3 shape:      {p.W3.shape}")
print(f"  W4 shape:      {p.W4.shape}")

# 2. Тест предсказания
cfg = {
    "plasma_monitor": {
        "temperature": 1.5e8,
        "density_number": 1e21,
        "radius": 0.6,
        "external_B": 29.0,
    },
    "quantum_vacuum": {"E_field": 5e17},
}
forecast = p.predict(cfg)
if forecast:
    print(f"\nPredict test:")
    for k, v in forecast.items():
        print(f"  {k}: {v:.4e}")
    print("\nPASS: Factory model works!")
else:
    print("\nFAIL: predict() вернул None")
