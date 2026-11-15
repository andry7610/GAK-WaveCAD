"""
MLP-фильтр для pre-screening конфигов оптимизации.
Обучен на 50 trials (Q >= 2 → good, Q < 2 → bad).
F1 = 0.99, Recall = 1.0 (не отбрасывает хорошие конфиги).
"""
import os
import pickle
import numpy as np


class MLPFilter:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.config = None
        self.is_trained = False

    def load(self, model_dir):
        model_path = os.path.join(model_dir, "mlp_filter_model_v2.pkl")
        scaler_path = os.path.join(model_dir, "mlp_filter_scaler_v2.pkl")
        config_path = os.path.join(model_dir, "mlp_filter_config_v2.pkl")

        for p in [model_path, scaler_path, config_path]:
            if not os.path.exists(p):
                print(f"[MLP] Файл не найден: {p}")
            else:
                print(f"[MLP] Файл найден: {p}")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
            return False

        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            self.is_trained = True
            print(f"[MLP] Загружен. input_features: {self.config['input_features']}")
            return True
        except Exception as e:
            print(f"[MLP] Ошибка загрузки: {e}")
            return False

    def _extract_features(self, params_dict):
        feature_cols = self.config['feature_cols']
        log10_cols = self.config['log10_cols']

        features = {}
        for col in feature_cols:
            val = params_dict.get(col, 0)
            if col in log10_cols:
                features[col + '_log10'] = np.log10(max(val, 1e-30))
            else:
                features[col] = val

        return np.array([features[f] for f in self.config['input_features']]).reshape(1, -1)

    def predict(self, params_dict):
        if not self.is_trained:
            print("[MLP] Модель не загружена — пропускаю")
            return True

        try:
            X = self._extract_features(params_dict)
            X_scaled = self.scaler.transform(X)
            result = bool(self.model.predict(X_scaled)[0] == 1)
            print(f"[MLP] Predict: {result} | X shape: {X.shape}")
            return result
        except Exception as e:
            print(f"[MLP] Ошибка предсказания: {e}")
            return True
