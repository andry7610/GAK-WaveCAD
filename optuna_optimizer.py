"""
GAK-WaveCAD — Optuna-оптимизатор v37.2.
 v37.2: FIX-пакет (сирена ДипСика + аудит Алисы):
      (1) objective_ignition/objective_gain возвращают -P_net_W (ЧИСТАЯ мощность
          станции) — TPE впервые видит налог B^2*pump^2 на рециркуляцию;
      (2) FIX pump_speed: для золы и рециркуляции берётся из cfg["vacuum_monitor"]
          (был неверный словарь plasma_monitor — ползунок откачки не влиял НИ НА ЧТО,
          и зола, и ALIP считались с дефолтом 0.0665);
      (3) прунер репортит чистую мощность (зола + станционный баланс) — та же шкала,
          что и objective;
      (4) set_user_attr P_net_W/P_electric_W в оба objective (видны в CSV);
      (5) best_value и success-сообщения — на P_net.
 v37: ГРЯЗНАЯ ПЛАЗМА (совет трёх КБ):
      (1) гелиевая зола — равновесная доля f_He из баланса рождение/откачка:
          f_He = min(F_HE_MAX, C_ASH * R_burn / (n * V * pump_speed)); pump_speed
          впервые становится настоящей ручкой откачки, а не декорацией.
          Разбавление топлива: n_fuel = n*(1-f_He), мощность ∝ (1-f_He)^2,
          P_alpha ∝ (1-f_He)^2, Лоусон по n_fuel.
          Калибровка C_ASH: f_He≈0.15 на точке рекорда 1527 МВт
          (n=1.6e21, V=0.974 м3, pump=0.0615).
      (2) вольфрам: c_W=1e-4 константой → Z_eff = 1 + c_W*Z*(Z-1) ≈ 1.54,
          бремсштрулунг P_rad *= Z_eff (первое приближение, динамика T не пересчитывается).
      (3) pump_speed расширен до (0.01, 0.15) — BOUNDS, IGNITION_BOUNDS, burn-фаза.
      (4) границы v36.2/v36.3 включены: T_plasma_init до 1.6e8, n_density до 1.6e21,
          pl_radius до 620 (сейфти R*0.52 на месте).
      (5) сиды: чемпион trial-4 (1282 МВт, trial 31 старого прогона) и рекорд
          1527 МВт (n=1.6e21, pl=615) + разведка 1.55e21/615.
 v36.1: FIX (ревью ДипСика): pump_speed берётся из cfg["vacuum_monitor"] (был неверный
      словарь — рециркуляция ALIP не реагировала на pump_speed); убран дубль
      set_user_attr("beta") в objective_ignition; дефолт n_steps синхронизирован
      с _build_trial_config (pruner и триал всегда считают один и тот же прогон).
 v32 (историческая версия):
v22: FIX init() temp reset + cascade burn-phase MHD switch.
v26: BOUNDS k_thermal (20,90), n_density (6e20,1.3e21) — пробить брейкивен.
v31: objective=P_fusion (total power), Q>=4.8 + lawson constraints, sim_time=1s.
 v32: честный энергобаланс - Q_plasma (P_alpha/P_loss), f_ext (доля внешнего подогрева),
      каскадный КПД, тритий, когенерация. Патч КБ «Два звездабола».
 v33: FIX размерностей (замечания ОТК): f_ext и P_fusion — полные ватты, а не Вт/м3;
      дубли _run_quick убраны; user_attrs Q_plasma/f_ext/P_fusion_W; seed=43, n_startup=10.
 v34: прунер репортит полную мощность (Вт) — та же шкала, что objective; третий seed
      (trial 29: 460.7 МВт при f_ext=2.1%); сэмплер ignition 43/10.
 v35: мягкая остановка STOP_FLAG (models/STOP_FLAG → study.stop() между trial),
      catch=(Exception,) — сбой одного trial не роняет прогон.
 v36: судья на разогреве не работает — warmup 50% прогона, репорты на 50–87.5%;
      β-метрика (MHD) + мягкий штраф β>8%; n_density до 1.5e21 + разведка 1.5e21
      с дедупликацией; рециркуляция = криогеника + ALIP ~ (B·v)² (налог B²).
v21: FIX T_outer BOUNDS (273,300) — золотая зона, seeds внутри границ.
v20: динамический pl_radius с safety check.
v19: BOUNDS — золотая зона из 50 trials (trial 44, Q=4.54).
v15: каскадная оптимизация (зажигание → горение) с плавным переходом и штрафом за нестабильность.
v9.3: constraint pl_radius, синхронизация B, честные метрики.
v10: golden band B=28-30T, R>=600mm, n_steps from base_config.
v11: суженные BOUNDS (golden zone из 50 trials), MedianPruner, seed best params.
"""

import optuna
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import traceback

# === AI TRANSFORMER FILTER ===
try:
    from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
    _ai_predictor = None
    _ai_loaded = False

    def _get_ai_predictor(script_dir):
        global _ai_predictor, _ai_loaded
        if _ai_loaded:
            return _ai_predictor
        _ai_loaded = True
        model_path = os.path.join(script_dir, "models", "ai_predictor_DT.json")
        if os.path.exists(model_path):
            _ai_predictor = GAKTransformerPredictor()
            if _ai_predictor.load(model_path):
                try:
                    print("[AI Filter] Model loaded")
                except Exception:
                    pass
                return _ai_predictor
        try:
            print("[AI Filter] Model not found")
        except Exception:
            pass
        _ai_predictor = GAKTransformerPredictor.load_default()
        if _ai_predictor is not None and _ai_predictor.is_trained:
            print("[AI Filter] Factory model loaded")
        return _ai_predictor
except ImportError:
    _ai_predictor = None
    _ai_loaded = False

    def _get_ai_predictor(script_dir):
        return None
# === END AI TRANSFORMER FILTER ===


# v11: лучшие параметры из 50 trials (trial 44, Q=4.54)
BEST_SEED_PARAMS = {
    "R_mm":           1194.66,
    "h_mm":           14.33,
    "T_inner":        387.74,
    "T_outer":        287.89,
    "k_thermal":      56.49,
    "pl_radius_mm":   594.49,
    "T_plasma_init":  112850099.0,
    "n_density":      9.974e20,
    "P_initial":      0.0712,
    "pump_speed":     0.0665,
    "E_field":        5.756e17,
    "gap_nm":         1,
    "external_B":     28.71,
}

# v11: второй seed (trial 42, Q=4.47)
SEED_PARAMS_2 = {
    "R_mm":           1197.15,
    "h_mm":           14.29,
    "T_inner":        382.5,
    "T_outer":        287.3,
    "k_thermal":      65.95,
    "pl_radius_mm":   574.35,
    "T_plasma_init":  148584222.0,
    "n_density":      9.93e20,
    "P_initial":      1.084,
    "pump_speed":     0.0517,
    "E_field":        4.815e17,
    "gap_nm":         3,
    "external_B":     28.74,
}


class GAKOptunaOptimizer:
    """
    Три режима:
    1. Зажигание — максимизация Q (с Lawson constraint)
    2. Усиление   — максимизация Q
    3. Парето     — pair_rate → max + Q → max
    """

    FUEL_TYPES = ["D-He3", "D-T", "D-D"]

    LAWSON_THRESHOLDS = {
        "D-He3": 1.0e22,
        "D-T":   3.0e21,
        "D-D":   1.0e24,
    }

    # v11: суженные BOUNDS — золотая зона из 50 trials
    # Плохие (Q<3): pl_radius<530, n<6e20, P_initial>100, gap>10, R<1150, B>32
    # Хорошие (Q>4): pl_radius 530-595, n 7-10e20, P_initial<2, gap 1-8, R 1150-1200, B 28-31
    BOUNDS = {
        "R_mm":           (1140, 1210),    # v21: золотая зона (trial 44: 1194.66)
        "h_mm":           (10.0, 18.0),    # v21: золотая зона (trial 44: 14.33)
        "T_inner":        (360.0, 420.0),  # v21: золотая зона (trial 44: 387.74)
        "T_outer":        (273.0, 300.0),   # v21: FIX — было (273,280), но seed=287.89!
        "k_thermal":      (20.0, 90.0),     # v26: расширено вниз — пробить брейкивен
        "pl_radius_mm":    (500, 620),       # v36.3: 610->620 (рекорд 1527 МВт при 615)
        "T_plasma_init":  (1e8, 1.6e8),     # v36.2: 1.5e8->1.6e8 (над сид-чемпионом 155.7 МК)
        "n_density":      (6e20, 1.6e21),   # v36.3: 1.5e21->1.6e21 (β-штраф контролирует)
        "fuel_type_idx":  (0, 2),
        "P_initial":      (0.01, 3.0),      # v21: золотая зона (trial 44: 0.0712)
        "pump_speed":     (0.01, 0.15),     # v37: реальная ручка откачки золы (было 0.07)
        "E_field":        (2.5e17, 6e17),   # v25: сужено — плавный градиент Швингера
        "gap_nm":          (1, 8),            # v21: золотая зона (trial 44: 1)
        "external_B":     (28.0, 31.0),     # v21: золотая зона (trial 44: 28.71)
    }

    T_CAP = 1e9  # 1000 МК — выше это runaway

    # v15: Каскадные границы — раздельные для зажигания и горения
    # ОБЩИЕ параметры (одни и те же в обеих фазах): R, h, T_inner, T_outer, k_thermal,
    #   pl_radius, T_plasma_init, n_density, external_B, gap_nm
    # ФАЗО-СПЕЦИФИЧНЫЕ: E_field, P_initial, pump_speed

    IGNITION_BOUNDS = {
        # v21: золотая зона
        "R_mm":           (1140, 1210),
        "h_mm":           (10.0, 18.0),
        "T_inner":        (360.0, 420.0),
        "T_outer":        (273.0, 300.0),   # v21: FIX — было (273,282)
        "k_thermal":      (20.0, 90.0),  # v26: расширено вниз
        "pl_radius_mm":    (500, 620),       # v36.3: 610->620; динамический сейфти R*0.52
        "T_plasma_init":  (1e8, 1.6e8),     # v36.2
        "n_density":      (6e20, 1.6e21),   # v36.3
        "external_B":     (28.0, 31.0),
        "gap_nm":          (1, 8),
        # Фаза-специфичные (зажигание — нужен мощный импульс)
        "E_field":        (2.5e17, 6e17),   # v25: синхрон с BOUNDS
        "P_initial":      (0.01, 3.0),
        "pump_speed":     (0.01, 0.15),     # v37
    }

    # BURN_BOUNDS задаётся динамически вокруг top-N из Phase 1
    # Ширина окна: ±15% от значения из Phase 1
    BURN_BOUNDS_WIDTH = 0.15  # 15% в каждую сторону

    # v15: доля шагов на переход зажигание → горение
    TRANSITION_FRAC = 0.05  # 5% шагов — плавная интерполяция pair_rate



    # v13: анти-галлюцинация суррогата
    SURROGATE_CAP = 6.0   # жёсткий потолок Q_pred (1.3× от лучшего реального Q=4.58)
    MAX_TRAIN_Q = 4.6     # макс Q в обучающей выборке — граница доверия

    # v9.1: D-T зафиксирован
    FIXED_FUEL_IDX = 1  # D-T
    # -- v32: константы станции (съём энергии) --
    ETA_MHD = 0.22           # КПД МГД-топпинга (0.15-0.30, трактовка: от теплового потока)
    ETA_STEAM = 0.42         # КПД парового нижнего цикла
    RECIRC_FRAC = 0.04       # legacy-константа (дефолт, если B/v не переданы)
    # v36: рециркуляция = криогеника магнитов + ALIP-прокачка (налог МГД ~ (B·v)^2)
    RECIRC_CRYO_FRAC = 0.02
    RECIRC_PUMP_FRAC = 0.02
    RECIRC_REF_B = 28.71
    RECIRC_REF_PUMP = 0.0665
    BETA_MAX = 0.08          # v36: предел MHD-беты (штраф в objective)
    MU0 = 1.25663706212e-6   # магнитная постоянная
    SYNC_HARVEST_FRAC = 0.15 # доля нетто-синхротрона, отдаваемая ИК-конвертерам
    SYNC_CONV_EFF = 0.40     # КПД ИК-конвертеров
    T_BR = 1.05              # коэффициент воспроизводства трития
    N_A  = 6.02214076e23     # число Авогадро, 1/моль
    M_TRITIUM  = 3.016e-3 / N_A   # кг на ОДИН атом трития

    E_DT = 17.6e6 * 1.602176634e-19  # Дж на реакцию D-T
    SECONDS_PER_YEAR = 3.156e7
    F_EXT_MAX = 0.20         # constraint v32: доля внешнего подогрева <= 20%
    # v37: гелиевая зола и вольфрам
    E_FUSION_J = {           # энергия на реакцию, Дж
        "D-T":   2.8199e-12,   # 17.6 МэВ
        "D-He3": 2.9332e-12,   # 18.3 МэВ
        "D-D":   5.85e-13,     # ~3.65 МэВ в среднем по ветвям
    }
    C_ASH = 0.0266           # калибровка золы: f_He≈0.15 на рекорде 1527 МВт (pump=0.0615)
    F_HE_MAX = 0.30          # предельная доля золы (выше — топливо выгорает наглухо)
    C_W = 1.0e-4             # доля вольфрама (первая итерация — константа)
    Z_W = 74                 # атомный номер вольфрама

    @classmethod
    def _station_balance(cls, P_fusion_W, P_sync_net_W, B_T=None, pump_speed=None):
        # v32: энергобаланс станции. Зовётся из _run_quick с усреднёнными мощностями.
        rate = P_fusion_W / cls.E_DT if cls.E_DT > 0 else 0.0   # реакций/с
        m_T_burn = rate * cls.M_TRITIUM * cls.SECONDS_PER_YEAR   # кг трития сжигается/год
        m_T_net = m_T_burn * (cls.T_BR - 1.0) / cls.T_BR         # избыток на продажу/запас
        eta_cascade = cls.ETA_MHD + (1.0 - cls.ETA_MHD) * cls.ETA_STEAM
        P_sync_harvest = cls.SYNC_HARVEST_FRAC * max(P_sync_net_W, 0.0) * cls.SYNC_CONV_EFF
        P_electric = P_fusion_W * eta_cascade + P_sync_harvest
        # v36: честная рециркуляция — криогеника (константа) + ALIP-прокачка ~ (B·v)^2
        recirc = cls.RECIRC_FRAC
        if B_T is not None and pump_speed is not None:
            recirc = cls.RECIRC_CRYO_FRAC + cls.RECIRC_PUMP_FRAC * (
                (float(B_T) / cls.RECIRC_REF_B) ** 2) * (
                (float(pump_speed) / cls.RECIRC_REF_PUMP) ** 2)
            recirc = float(np.clip(recirc, cls.RECIRC_CRYO_FRAC, 0.12))
        P_net = P_electric * (1.0 - recirc)
        return {
            "eta_cascade": eta_cascade,
            "recirc_frac": recirc,
            "P_electric_W": P_electric,
            "P_net_W": P_net,
            "P_sync_harvest_W": P_sync_harvest,
            "P_thermal_reject_W": P_fusion_W * (1.0 - eta_cascade),
            "m_T_burn_kg_year": m_T_burn,
            "m_T_net_kg_year": m_T_net,
        }

    def __init__(self, base_config):
        self.base_config = base_config
        self.study = None
        self.results = None
        self.mode = None
        self.predictor = None  # суррогат (трансформер) для pre-screen
        self._prescreen_stats = {"screened": 0, "passed": 0, "rejected": 0}
        self._pruned_count = 0
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._predictor_save_path = os.path.join(script_dir, "models", "ai_predictor_DT.json")  # v11.3: абсолютный путь
        self._auto_learn_counter = 0  # v11.5: счётчикчик для batch-обучения
        self._auto_learn_interval = 5  # v11.5: учить каждые 5 trials
        # v15.2: SQLite storage — прогресс не теряется при перезагрузке страницы
        self._study_storage_path = f"sqlite:///{os.path.join(script_dir, 'models', 'optuna_study.db')}"
        # v35: мягкая остановка прогона (кнопка на странице остановки)
        self._stop_flag_path = os.path.join(script_dir, "models", "STOP_FLAG")


    def _make_stop_callback(self):
        """v35: мягкая остановка — models/STOP_FLAG → study.stop() между trial."""
        def _stop_cb(study, trial):
            if os.path.exists(self._stop_flag_path):
                print("[v35] STOP_FLAG обнаружен — останавливаем study после текущего trial")
                study.stop()
        return _stop_cb

    def set_predictor(self, predictor):
        """Привязка суррогатной модели для гибридного pre-screen."""
        self.predictor = predictor

    def _prescreen(self, cfg):
        """Гибридный pre-screen: трансформер отсеивает явный брак.
        Возвращает:
          None — конфиг прошёл pre-screen (запускать физику)
          'reject' — конфиг отбракован трансформером
        Физика — финальный арбитр. Трансформер отсеивает только явный мусор.
        """
        if self.predictor is None or not self.predictor.is_trained:
            return None

        self._prescreen_stats["screened"] += 1

        try:
            forecast = self.predictor.predict(cfg)
            if forecast is None:
                self._prescreen_stats["passed"] += 1
                return None

            T_pred = forecast.get("T_plasma", 0.0)
            pair_rate_pred = forecast.get("pair_rate", 0.0)

            # Отсеиваем только явный брак:
            # 1. T > T_CAP — runaway
            if T_pred > self.T_CAP:
                self._prescreen_stats["rejected"] += 1
                return 'reject'
            # 2. T < 1e6 — слишком холодная, реакция не пойдёт
            if T_pred < 1e6:
                self._prescreen_stats["rejected"] += 1
                return 'reject'
            # 3. pair_rate < 1 — нет рождения пар
            if pair_rate_pred < 1.0:
                self._prescreen_stats["rejected"] += 1
                return 'reject'

            self._prescreen_stats["passed"] += 1
            return None
        except Exception:
            self._prescreen_stats["passed"] += 1
            return None

    def _build_trial_config(self, trial):
        """Конфиг из trial. Constraint: pl_radius < R_mm * 0.5."""
        cfg = dict(self.base_config)

        R_mm = trial.suggest_float("R_mm", *self.BOUNDS["R_mm"])
        h_mm = trial.suggest_float("h_mm", *self.BOUNDS["h_mm"])
        T_inner = trial.suggest_float("T_inner", *self.BOUNDS["T_inner"])
        T_outer = trial.suggest_float("T_outer", *self.BOUNDS["T_outer"])
        k_thermal = trial.suggest_float("k_thermal", *self.BOUNDS["k_thermal"])

        # v21: Constraint: плазма внутри сферы (золотая зона + safety)
        pl_radius_lo = max(R_mm * 0.42, 500)
        pl_radius_hi = min(R_mm * 0.52, 620)
        if pl_radius_hi <= pl_radius_lo:
            pl_radius_hi = pl_radius_lo + 10.0  # safety: never flip
        pl_radius_mm = trial.suggest_float("pl_radius_mm",
                                           pl_radius_lo,
                                           pl_radius_hi)
        T_plasma = trial.suggest_float("T_plasma_init", *self.BOUNDS["T_plasma_init"], log=True)
        n_density = trial.suggest_float("n_density", *self.BOUNDS["n_density"], log=True)

        # v9.1: D-T зафиксирован
        fuel_idx = self.FIXED_FUEL_IDX
        fuel_type = self.FUEL_TYPES[fuel_idx]

        P_initial = trial.suggest_float("P_initial", *self.BOUNDS["P_initial"], log=True)
        pump_speed = trial.suggest_float("pump_speed", *self.BOUNDS["pump_speed"], log=True)
        E_field = trial.suggest_float("E_field", *self.BOUNDS["E_field"], log=True)
        gap_nm = trial.suggest_int("gap_nm", *self.BOUNDS["gap_nm"])
        external_B = trial.suggest_float("external_B", *self.BOUNDS["external_B"])

        R_m = R_mm / 1000
        h_m = h_mm / 1000
        pl_radius_m = pl_radius_mm / 1000

        for mod_key in ("thermal_monitor", "acoustic_monitor",
                        "born_collapse_monitor", "magnon_monitor",
                        "em_resonance_monitor"):
            mod = dict(self.base_config.get(mod_key, {}))
            mod["R"] = R_m
            mod["h"] = h_m
            cfg[mod_key] = mod

        cfg["thermal_monitor"]["T_inner"] = T_inner
        cfg["thermal_monitor"]["T_outer"] = T_outer
        cfg["thermal_monitor"]["k_thermal"] = k_thermal

        cfg["plasma_monitor"] = {
            **self.base_config.get("plasma_monitor", {}),
            "radius": pl_radius_m,
            "temperature": T_plasma,
            "density_number": n_density,
            "fuel_type": fuel_type,
            "auto_feed": False,
            "quantum_feedback": True,
            "external_B": external_B,
        }
        cfg["vacuum_monitor"] = {
            **self.base_config.get("vacuum_monitor", {}),
            "P_initial": P_initial,
            "pump_speed": pump_speed,
        }
        cfg["quantum_vacuum"] = {
            **self.base_config.get("quantum_vacuum", {}),
            "E_field": E_field,
            "gap": gap_nm * 1e-9,
        }
        cfg["simulation"] = {"n_steps": self.base_config.get("simulation", {}).get("n_steps", 100000), "dt": 1e-5}
        return cfg


    def _build_cascade_config(self, trial, phase, shared_params=None):
        """v15: Построение конфига для каскадной фазы.
        phase: 'ignition' или 'burn'
        shared_params: dict параметров из Phase 1 (для burn — общие параметры фиксированы)"""
        cfg = dict(self.base_config)

        if phase == 'ignition':
            bounds = self.IGNITION_BOUNDS
            # Все параметры свободны
            R_mm = trial.suggest_float("R_mm", *bounds["R_mm"])
            h_mm = trial.suggest_float("h_mm", *bounds["h_mm"])
            T_inner = trial.suggest_float("T_inner", *bounds["T_inner"])
            T_outer = trial.suggest_float("T_outer", *bounds["T_outer"])
            k_thermal = trial.suggest_float("k_thermal", *bounds["k_thermal"])
            pl_radius_lo = max(R_mm * 0.42, 500)
            pl_radius_hi = min(R_mm * 0.52, 620)
            if pl_radius_hi <= pl_radius_lo:
                pl_radius_hi = pl_radius_lo + 10.0  # safety: never flip
            pl_radius_mm = trial.suggest_float("pl_radius_mm",
                                                pl_radius_lo,
                                                pl_radius_hi)
            T_plasma = trial.suggest_float("T_plasma_init", *bounds["T_plasma_init"], log=True)
            n_density = trial.suggest_float("n_density", *bounds["n_density"], log=True)
            external_B = trial.suggest_float("external_B", *bounds["external_B"])
            gap_nm = trial.suggest_int("gap_nm", *bounds["gap_nm"])
            E_field = trial.suggest_float("E_field", *bounds["E_field"], log=True)
            P_initial = trial.suggest_float("P_initial", *bounds["P_initial"], log=True)
            pump_speed = trial.suggest_float("pump_speed", *bounds["pump_speed"], log=True)
        elif phase == 'burn':
            # Общие параметры — из shared_params (топ-N из Phase 1)
            # Фаза-специфичные — свободны в узком окне
            R_mm = shared_params["R_mm"]
            h_mm = shared_params["h_mm"]
            T_inner = shared_params["T_inner"]
            T_outer = shared_params["T_outer"]
            k_thermal = shared_params["k_thermal"]
            pl_radius_mm = shared_params["pl_radius_mm"]
            T_plasma = shared_params["T_plasma_init"]
            n_density = shared_params["n_density"]
            external_B = shared_params["external_B"]
            gap_nm = int(shared_params["gap_nm"])

            # Фаза-специфичные — горение (низкий E_field, низкое P)
            w = self.BURN_BOUNDS_WIDTH
            e_lo, e_hi = 1e16, 5e17   # НИЗКИЙ E_field для горения
            p_lo, p_hi = 1e-4, 1.0    # НИЗКОЕ давление
            ps_lo, ps_hi = 1e-3, 0.15  # v37

            E_field = trial.suggest_float("E_field_burn", e_lo, e_hi, log=True)
            P_initial = trial.suggest_float("P_initial_burn", p_lo, p_hi, log=True)
            pump_speed = trial.suggest_float("pump_speed_burn", ps_lo, ps_hi, log=True)
            # t_switch — момент переключения (30%–70% шагов)
            t_switch = trial.suggest_float("t_switch_frac", 0.3, 0.7)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        fuel_idx = self.FIXED_FUEL_IDX
        fuel_type = self.FUEL_TYPES[fuel_idx]

        R_m = R_mm / 1000
        h_m = h_mm / 1000
        pl_radius_m = pl_radius_mm / 1000

        for mod_key in ("thermal_monitor", "acoustic_monitor",
                        "born_collapse_monitor", "magnon_monitor",
                        "em_resonance_monitor"):
            mod = dict(self.base_config.get(mod_key, {}))
            mod["R"] = R_m
            mod["h"] = h_m
            cfg[mod_key] = mod

        cfg["thermal_monitor"]["T_inner"] = T_inner
        cfg["thermal_monitor"]["T_outer"] = T_outer
        cfg["thermal_monitor"]["k_thermal"] = k_thermal

        cfg["plasma_monitor"] = {
            **self.base_config.get("plasma_monitor", {}),
            "radius": pl_radius_m,
            "temperature": T_plasma,
            "density_number": n_density,
            "fuel_type": fuel_type,
            "auto_feed": False,
            "quantum_feedback": True,
            "external_B": external_B,
        }
        cfg["vacuum_monitor"] = {
            **self.base_config.get("vacuum_monitor", {}),
            "P_initial": P_initial,
            "pump_speed": pump_speed,
        }
        cfg["quantum_vacuum"] = {
            **self.base_config.get("quantum_vacuum", {}),
            "E_field": E_field,
            "gap": gap_nm * 1e-9,
        }
        cfg["simulation"] = {"n_steps": self.base_config.get("simulation", {}).get("n_steps", 100000), "dt": 1e-5}

        if phase == 'burn':
            return cfg, t_switch
        return cfg


    def _run_quick(self, cfg, trial=None):
        """Упрощённый прогон: plasma.step() + ручной Швингер. Без run().
        v9.2: Adaptive dt — dt_actual = min(sim_dt, 0.05 * tau_E).
        v11: поддержка MedianPruner — если передан trial, репортит промежуточный Q
             на 20%, 40%, 60%, 80% шагов и прерывает при should_prune()."""
        import numpy as np
        from physical_modules.plasma_monitor import PlasmaMonitor

        try:
            sim_n = cfg["simulation"]["n_steps"]
            sim_dt = cfg["simulation"]["dt"]

            plasma = PlasmaMonitor(cfg.get("plasma_monitor", {}))
            plasma.init()

            V_plasma = (4.0 / 3.0) * np.pi * (plasma.radius ** 3)
            pair_energy_gain = plasma.pair_energy_gain
            external_B = cfg["plasma_monitor"]["external_B"]
            fuel_type = cfg["plasma_monitor"]["fuel_type"]
            n_density = cfg["plasma_monitor"]["density_number"]

            ratios = []
            pair_rates = []
            T_vals = []
            P_fusion_vals = []
            P_cond_vals = []
            P_rad_vals = []
            P_sync_vals = []
            P_alpha_vals = []
            tau_E_vals = []
            Q_vals = []
            Q_plasma_vals = []
            f_ext_vals = []
            dt_actual_vals = []
            sim_time = 0.0

            runaway = False

            # ── T_CAP pre-check ──
            T_init_check = cfg["plasma_monitor"]["temperature"]
            if T_init_check > self.T_CAP:
                return None

            # ── pair_rate ──
            qv_cfg = cfg.get("quantum_vacuum", {})
            E_field_qv = qv_cfg.get("E_field", 0.0)
            gap_qv = qv_cfg.get("gap", 1e-9)
            E_c = 1.32e18
            pair_rate = 0.0
            if E_field_qv > 0.001 * E_c:
                schwinger_factor = np.exp(-np.pi * E_c / max(E_field_qv, 1e-30))
                pair_rate = 7.076e13 * (E_field_qv / E_c) ** 2 * schwinger_factor / (gap_qv * 1e9)
                pair_rate = min(pair_rate, 1e10)

            # v36: точки репорта для pruner (50%..87.5%) — не судим триал на разогреве
            report_steps = set()
            if trial is not None:
                for frac in [0.5, 0.625, 0.75, 0.875]:
                    report_steps.add(int(sim_n * frac))

            for step_i in range(sim_n):
                tau_E_current = plasma.compute_tau_E()
                if tau_E_current > 0:
                    dt_actual = min(sim_dt, 0.05 * tau_E_current)
                else:
                    dt_actual = sim_dt

                step_result = plasma.step(dt_actual, pair_rate=pair_rate)
                sim_time += dt_actual

                T_plasma = step_result["temperature"]
                P_fusion = step_result["P_fusion"]
                P_cond = step_result["P_cond"]
                P_rad = step_result["P_rad"]
                tau_E = step_result["tau_E"]

                P_sync = step_result.get("P_sync", 0.0)
                P_alpha = step_result.get("P_alpha", 0.2 * P_fusion)
                alpha_frac = step_result.get("alpha_frac", 0.2)

                if plasma.temperature > self.T_CAP:
                    runaway = True
                    break

                if np.isnan(plasma.temperature) or plasma.temperature <= 0:
                    break

                # Диагностический ratio
                P_heat = pair_rate * pair_energy_gain
                P_heat_d = P_heat / V_plasma if V_plasma > 0 else 0.0
                ratio = P_heat_d / P_cond if P_cond > 0 and P_cond < 1e30 else 0.0

                # Q с P_sync
                P_loss = P_cond + P_rad + P_sync
                Q = P_fusion / P_loss if P_loss > 0 and P_loss < 1e30 else 0.0
                # v33: честные метрики. P_loss — плотность Вт/м3, P_heat_loc — полные Вт,
                # поэтому f_ext считаем от ПОЛНЫХ потерь: P_loss * V_plasma
                P_heat_loc = pair_rate * pair_energy_gain
                Q_plasma = P_alpha / P_loss if P_loss > 0 and P_loss < 1e30 else 0.0
                P_loss_total = P_loss * V_plasma if V_plasma > 0 else 0.0
                f_ext = P_heat_loc / P_loss_total if P_loss_total > 0 and P_loss_total < 1e30 else 0.0

                ratios.append(ratio)
                pair_rates.append(pair_rate)
                T_vals.append(T_plasma)
                P_fusion_vals.append(P_fusion)
                P_cond_vals.append(P_cond)
                P_rad_vals.append(P_rad)
                P_sync_vals.append(P_sync)
                P_alpha_vals.append(P_alpha)
                tau_E_vals.append(tau_E)
                Q_vals.append(Q)
                Q_plasma_vals.append(Q_plasma)
                f_ext_vals.append(f_ext)
                dt_actual_vals.append(dt_actual)

                # v11: MedianPruner — репорт и проверка
                if trial is not None and step_i in report_steps and len(Q_vals) > 0:
                    # v37.2: репорт ЧИСТОЙ мощности — та же шкала, что у objective
                    recent_p = float(np.mean(P_fusion_vals[-max(1, len(P_fusion_vals)//10):])) * V_plasma
                    E_fus_rep = self.E_FUSION_J.get(fuel_type, self.E_DT)
                    R_burn_rep = recent_p / E_fus_rep if E_fus_rep > 0 else 0.0
                    pump_rep = float(cfg.get("vacuum_monitor", {}).get(
                        "pump_speed", self.RECIRC_REF_PUMP))
                    f_He_rep = min(self.F_HE_MAX, self.C_ASH * R_burn_rep /
                                   max(n_density * V_plasma * pump_rep, 1e-30))
                    P_net_rep = recent_p * (1.0 - f_He_rep) ** 2
                    P_elec_rep = P_net_rep * (self.ETA_MHD +
                                              (1.0 - self.ETA_MHD) * self.ETA_STEAM)
                    recirc_rep = self.RECIRC_CRYO_FRAC + self.RECIRC_PUMP_FRAC * \
                        (external_B / self.RECIRC_REF_B) ** 2 * \
                        (pump_rep / self.RECIRC_REF_PUMP) ** 2
                    recirc_rep = float(np.clip(recirc_rep, self.RECIRC_CRYO_FRAC, 0.12))
                    trial.report(-P_elec_rep * (1.0 - recirc_rep), step_i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if len(T_vals) < 10:
                st.warning(
                    f"_run_quick: только {len(T_vals)} шагов | "
                    f"T_init={cfg['plasma_monitor']['temperature']:.2e} | "
                    f"T_last={plasma.temperature:.2e} | "
                    f"tau_E={tau_E_current:.2e} | "
                    f"dt_actual={dt_actual:.2e} | "
                    f"P_fusion={P_fusion:.2e} | "
                    f"P_cond={P_cond:.2e} | "
                    f"E_field={E_field_qv:.2e} | "
                    f"pair_rate={pair_rate:.2e}"
                )
                return None

            # v10: отбраковка упавших симуляций
            if len(T_vals) < 0.9 * sim_n:
                return None

            n_avg = max(1, len(T_vals) // 5)
            avg_ratio = float(np.mean(ratios[-n_avg:]))
            avg_pair_rate = float(np.mean(pair_rates[-n_avg:]))
            avg_T = float(np.mean(T_vals[-n_avg:]))
            avg_P_fusion = float(np.mean(P_fusion_vals[-n_avg:]))
            avg_P_cond = float(np.mean(P_cond_vals[-n_avg:]))
            avg_P_rad = float(np.mean(P_rad_vals[-n_avg:]))
            avg_P_sync = float(np.mean(P_sync_vals[-n_avg:]))
            avg_P_alpha = float(np.mean(P_alpha_vals[-n_avg:]))
            avg_tau_E = float(np.mean(tau_E_vals[-n_avg:]))
            avg_Q = float(np.mean(Q_vals[-n_avg:]))
            avg_dt_actual = float(np.mean(dt_actual_vals[-n_avg:]))
            avg_Q_plasma = float(np.mean(Q_plasma_vals[-n_avg:])) if Q_plasma_vals else 0.0
            avg_f_ext = float(np.mean(f_ext_vals[-n_avg:])) if f_ext_vals else 0.0
            # v36: MHD-бета = давление плазмы / магнитное давление (2nkT против B²/2μ0)
            avg_beta = (2.0 * n_density * 1.380649e-23 * avg_T) / (
                external_B ** 2 / (2.0 * self.MU0)) if external_B > 0 else 0.0

            # v33: объёмные плотности (Вт/м3) -> полные мощности (Вт)
            avg_P_fusion_W = avg_P_fusion * V_plasma
            avg_P_alpha_W = avg_P_alpha * V_plasma
            avg_P_cond_W = avg_P_cond * V_plasma
            avg_P_rad_W = avg_P_rad * V_plasma
            avg_P_sync_W = avg_P_sync * V_plasma

            # v37: гелиевая зола — равновесие рождение/откачка
            E_fus = self.E_FUSION_J.get(fuel_type, self.E_DT)
            R_burn = avg_P_fusion_W / E_fus if E_fus > 0 else 0.0
            pump_eff = float(cfg["vacuum_monitor"].get("pump_speed", self.RECIRC_REF_PUMP))
            f_He = min(self.F_HE_MAX,
                       self.C_ASH * R_burn / max(n_density * V_plasma * pump_eff, 1e-30))
            dil = (1.0 - f_He) ** 2
            n_fuel = n_density * (1.0 - f_He)
            # v37: вольфрам — Z_eff и бремсштрулунг
            Z_eff = 1.0 + self.C_W * (self.Z_W * self.Z_W - self.Z_W)
            P_loss_old_W = avg_P_cond_W + avg_P_rad_W + avg_P_sync_W
            avg_P_fusion_W *= dil
            avg_P_alpha_W *= dil
            avg_P_rad_W *= Z_eff
            P_loss_new_W = avg_P_cond_W + avg_P_rad_W + avg_P_sync_W
            avg_Q_plasma = avg_P_alpha_W / P_loss_new_W if P_loss_new_W > 0 else 0.0
            avg_f_ext = avg_f_ext * P_loss_old_W / P_loss_new_W if P_loss_new_W > 0 else avg_f_ext

            T_keV = avg_T * 1.380649e-23 / (1.602176634e-19 * 1e3)

            # v37: Лоусон по плотности ТОПЛИВА (зола разбавляет)
            lawson_product = n_fuel * T_keV * avg_tau_E
            lawson_threshold = self.LAWSON_THRESHOLDS.get(fuel_type, 3e21)
            lawson_achieved = lawson_product >= lawson_threshold

            return {
                "avg_ratio": avg_ratio,
                "avg_pair_rate": avg_pair_rate,
                "avg_T": avg_T,
                "T_keV": T_keV,
                "avg_P_cond": avg_P_cond,
                "avg_P_fusion": avg_P_fusion,
                "avg_P_rad": avg_P_rad,
                "avg_P_sync": avg_P_sync,
                "avg_P_alpha": avg_P_alpha,
                "avg_tau_E": avg_tau_E,
                "avg_Q": avg_Q,
                "lawson_product": lawson_product,
                "lawson_threshold": lawson_threshold,
                "lawson_achieved": lawson_achieved,
                "Q": avg_Q,
                "n": n_density,
                "fuel_type": fuel_type,
                "external_B": external_B,
                "V_plasma": V_plasma,
                "plasma_radius_actual_m": plasma.radius,
                "runaway": runaway,
                "steps_done": len(T_vals),
                "sim_time": sim_time,
                "avg_dt_actual": avg_dt_actual,
                "avg_Q_plasma": avg_Q_plasma,
                "avg_f_ext": avg_f_ext,
                "avg_beta": avg_beta,
                # v37: грязная плазма
                "f_He": f_He,
                "n_fuel": n_fuel,
                "Z_eff": Z_eff,
                # v33: полные мощности (Вт) для станции и objective
                "avg_P_fusion_W": avg_P_fusion_W,
                "avg_P_alpha_W": avg_P_alpha_W,
                "avg_P_cond_W": avg_P_cond_W,
                "avg_P_rad_W": avg_P_rad_W,
                "avg_P_sync_W": avg_P_sync_W,
                "station": self._station_balance(avg_P_fusion_W, avg_P_sync_W,
                                                 B_T=external_B,
                                                 pump_speed=cfg["vacuum_monitor"].get("pump_speed", self.RECIRC_REF_PUMP)),
            }

        except optuna.TrialPruned:
            raise
        except Exception as e:
            try:
                st.warning(f"_run_quick exception: {e}")
                tb = traceback.format_exc()
                tb_safe = tb.encode('utf-8', errors='replace').decode('utf-8')
                st.code(tb_safe)
            except Exception:
                pass
            return None



    def _compute_pair_rate(self, cfg):
        """Вычисление pair_rate из E_field и gap (фактор Швингера)."""
        qv_cfg = cfg.get("quantum_vacuum", {})
        E_field_qv = qv_cfg.get("E_field", 0.0)
        gap_qv = qv_cfg.get("gap", 1e-9)
        E_c = 1.32e18
        if E_field_qv > 0.001 * E_c:
            schwinger_factor = np.exp(-np.pi * E_c / max(E_field_qv, 1e-30))
            pr = 7.076e13 * (E_field_qv / E_c) ** 2 * schwinger_factor / (gap_qv * 1e9)
            return min(pr, 1e10)
        return 0.0

    def _run_quick_cascade(self, cfg_ignition, cfg_burn, t_switch_frac, trial=None):
        """v15: Каскадный прогон — зажигание → переход → горение.
        Фаза 1 (0..t_switch): конфиг зажигания (высокий pair_rate)
        Переход (t_switch..t_switch+5%): линейная интерполяция pair_rate
        Фаза 2 (t_switch+5%..end): конфиг горения (низкий pair_rate)
        Objective: Q в фазе горения + штраф за нестабильность."""
        import numpy as np
        from physical_modules.plasma_monitor import PlasmaMonitor

        try:
            sim_n = cfg_ignition["simulation"]["n_steps"]
            sim_dt = cfg_ignition["simulation"]["dt"]

            # PlasmaMonitor инициализируется с ignition-конфигом
            plasma = PlasmaMonitor(cfg_ignition.get("plasma_monitor", {}))
            plasma.init()

            V_plasma = (4.0 / 3.0) * np.pi * (plasma.radius ** 3)
            pair_energy_gain = plasma.pair_energy_gain
            external_B = cfg_ignition["plasma_monitor"]["external_B"]
            fuel_type = cfg_ignition["plasma_monitor"]["fuel_type"]
            n_density = cfg_ignition["plasma_monitor"]["density_number"]

            # pair_rate для каждой фазы
            pr_ignition = self._compute_pair_rate(cfg_ignition)
            pr_burn = self._compute_pair_rate(cfg_burn)

            # Точки перехода
            t_switch = int(sim_n * t_switch_frac)
            trans_steps = max(1, int(sim_n * self.TRANSITION_FRAC))
            t_burn_start = t_switch + trans_steps

            # Сбор данных
            ratios = []
            pair_rates_log = []
            T_vals = []
            P_fusion_vals = []
            P_cond_vals = []
            P_rad_vals = []
            P_sync_vals = []
            P_alpha_vals = []
            tau_E_vals = []
            Q_vals = []
            dt_actual_vals = []
            sim_time = 0.0
            runaway = False

            # T_CAP pre-check
            T_init_check = cfg_ignition["plasma_monitor"]["temperature"]
            if T_init_check > self.T_CAP:
                return None

            # Pruner
            report_steps = set()
            if trial is not None:
                for frac in [0.2, 0.4, 0.6, 0.8]:
                    report_steps.add(int(sim_n * frac))

            for step_i in range(sim_n):
                # Определяем текущий pair_rate
                if step_i < t_switch:
                    # Фаза зажигания
                    current_pr = pr_ignition
                elif step_i < t_burn_start:
                    # Переход: линейная интерполяция
                    frac = (step_i - t_switch) / trans_steps
                    current_pr = pr_ignition * (1.0 - frac) + pr_burn * frac
                else:
                    # Фаза горения
                    current_pr = pr_burn
                    # v22: накатываем параметры горения на живой plasma объект
                    if step_i == t_burn_start:
                        burn_pm = cfg_burn.get("plasma_monitor", {})
                        if "density_number" in burn_pm:
                            plasma.density_number = float(burn_pm["density_number"])
                        if "external_B" in burn_pm:
                            plasma.external_B = float(burn_pm["external_B"])
                        if "fuel_type" in burn_pm:
                            plasma.fuel_type = burn_pm["fuel_type"]
                            plasma._apply_fuel_preset()
                        # MHD-паспорт: переключаем auto_feed из burn-конфига
                        burn_vm = cfg_burn.get("vacuum_monitor", {})
                        plasma.auto_feed_enabled = burn_vm.get("auto_feed_enabled", True)
                        if hasattr(plasma, 'config') and plasma.config:
                            if "vacuum_monitor" not in plasma.config:
                                plasma.config["vacuum_monitor"] = {}
                            plasma.config["vacuum_monitor"]["P_initial"] = burn_vm.get("P_initial", 0.0)
                            plasma.config["vacuum_monitor"]["pump_speed"] = burn_vm.get("pump_speed", 0.0)
                        print(f"[Cascade] Burn phase: n={plasma.density_number:.2e}, "
                              f"B={plasma.external_B:.1f}, auto_feed={plasma.auto_feed_enabled}")

                tau_E_current = plasma.compute_tau_E()
                if tau_E_current > 0:
                    dt_actual = min(sim_dt, 0.05 * tau_E_current)
                else:
                    dt_actual = sim_dt

                step_result = plasma.step(dt_actual, pair_rate=current_pr)
                sim_time += dt_actual

                T_plasma = step_result["temperature"]
                P_fusion = step_result["P_fusion"]
                P_cond = step_result["P_cond"]
                P_rad = step_result["P_rad"]
                tau_E = step_result["tau_E"]
                P_sync = step_result.get("P_sync", 0.0)
                P_alpha = step_result.get("P_alpha", 0.2 * P_fusion)

                if plasma.temperature > self.T_CAP:
                    runaway = True
                    break

                if np.isnan(plasma.temperature) or plasma.temperature <= 0:
                    break

                # Диагностика
                P_heat = current_pr * pair_energy_gain
                P_heat_d = P_heat / V_plasma if V_plasma > 0 else 0.0
                ratio = P_heat_d / P_cond if P_cond > 0 and P_cond < 1e30 else 0.0

                P_loss = P_cond + P_rad + P_sync
                Q = P_fusion / P_loss if P_loss > 0 and P_loss < 1e30 else 0.0

                ratios.append(ratio)
                pair_rates_log.append(current_pr)
                T_vals.append(T_plasma)
                P_fusion_vals.append(P_fusion)
                P_cond_vals.append(P_cond)
                P_rad_vals.append(P_rad)
                P_sync_vals.append(P_sync)
                P_alpha_vals.append(P_alpha)
                tau_E_vals.append(tau_E)
                Q_vals.append(Q)
                dt_actual_vals.append(dt_actual)

                # Pruner
                if trial is not None and step_i in report_steps and len(Q_vals) > 0:
                    recent_q = float(np.mean(Q_vals[-max(1, len(Q_vals)//10):]))
                    trial.report(-recent_q, step_i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if len(T_vals) < 10:
                st.warning(
                    f"_run_quick_cascade: только {len(T_vals)} шагов | "
                    f"T_last={plasma.temperature:.2e} | "
                    f"pr_ign={pr_ignition:.2e} | pr_burn={pr_burn:.2e}"
                )
                return None

            if len(T_vals) < 0.9 * sim_n:
                return None

            # ── Метрики фазы горения (последние 20% шагов) ──
            burn_start_idx = int(len(T_vals) * 0.8)
            burn_Q = Q_vals[burn_start_idx:]
            burn_T = T_vals[burn_start_idx:]

            if len(burn_Q) < 5:
                return None

            avg_Q_burn = float(np.mean(burn_Q))
            avg_T_burn = float(np.mean(burn_T))

            # ── Стабильность: коэффициент вариации Q в фазе горения ──
            Q_std = float(np.std(burn_Q))
            Q_cv = Q_std / max(abs(avg_Q_burn), 1e-30)  # coefficient of variation

            # Штраф за нестабильность: если Q скачет, снижаем оценку
            # Q_cv < 0.1 — стабильно, Q_cv > 0.5 — нестабильно
            stability_penalty = min(Q_cv * 0.5, 0.5)  # макс 50% штраф
            effective_Q = avg_Q_burn * (1.0 - stability_penalty)

            # ── Проверка: плазма не погасла при переходе ──
            # T в фазе горения не должна упасть больше чем в 2× от пика
            T_peak = max(T_vals[:burn_start_idx]) if burn_start_idx > 0 else avg_T_burn
            T_dropped = T_peak / max(avg_T_burn, 1e-30)
            if T_dropped > 10.0:
                # Плазма погасла при переходе
                effective_Q *= 0.1  # Жёсткий штраф

            # ── Общие метрики (для совместимости) ──
            n_avg = max(1, len(T_vals) // 5)
            avg_ratio = float(np.mean(ratios[-n_avg:]))
            avg_pair_rate = float(np.mean(pair_rates_log[-n_avg:]))
            avg_T = float(np.mean(T_vals[-n_avg:]))
            avg_P_fusion = float(np.mean(P_fusion_vals[-n_avg:]))
            avg_P_cond = float(np.mean(P_cond_vals[-n_avg:]))
            avg_P_rad = float(np.mean(P_rad_vals[-n_avg:]))
            avg_P_sync = float(np.mean(P_sync_vals[-n_avg:]))
            avg_P_alpha = float(np.mean(P_alpha_vals[-n_avg:]))
            avg_tau_E = float(np.mean(tau_E_vals[-n_avg:]))
            avg_dt_actual = float(np.mean(dt_actual_vals[-n_avg:]))

            T_keV = avg_T * 1.380649e-23 / (1.602176634e-19 * 1e3)
            lawson_product = n_density * T_keV * avg_tau_E
            lawson_threshold = self.LAWSON_THRESHOLDS.get(fuel_type, 3e21)
            lawson_achieved = lawson_product >= lawson_threshold

            return {
                "avg_ratio": avg_ratio,
                "avg_pair_rate": avg_pair_rate,
                "avg_T": avg_T,
                "T_keV": T_keV,
                "avg_P_cond": avg_P_cond,
                "avg_P_fusion": avg_P_fusion,
                "avg_P_rad": avg_P_rad,
                "avg_P_sync": avg_P_sync,
                "avg_P_alpha": avg_P_alpha,
                "avg_tau_E": avg_tau_E,
                "avg_Q": effective_Q,
                "Q_burn_raw": avg_Q_burn,
                "Q_stability_cv": Q_cv,
                "T_peak": T_peak,
                "T_burn": avg_T_burn,
                "T_drop_ratio": T_dropped,
                "lawson_product": lawson_product,
                "lawson_threshold": lawson_threshold,
                "lawson_achieved": lawson_achieved,
                "Q": effective_Q,
                "n": n_density,
                "fuel_type": fuel_type,
                "external_B": external_B,
                "V_plasma": V_plasma,
                "plasma_radius_actual_m": plasma.radius,
                "runaway": runaway,
                "steps_done": len(T_vals),
                "sim_time": sim_time,
                "avg_dt_actual": avg_dt_actual,
                "cascade": True,
                "t_switch_frac": t_switch_frac,
                "pr_ignition": pr_ignition,
                "pr_burn": pr_burn,
            }

        except optuna.TrialPruned:
            raise
        except Exception as e:
            try:
                st.warning(f"_run_quick_cascade exception: {e}")
                tb = traceback.format_exc()
                tb_safe = tb.encode('utf-8', errors='replace').decode('utf-8')
                st.code(tb_safe)
            except Exception:
                pass
            return None


    def _staged_validate(self, top_params_list, sim_time_s=25.0, dt=1e-5):
        # v32: ступенчатая валидация - топ-кандидаты перепроверяются длинным прогоном.
        # Вызывается после optimize(): _opt._staged_validate([t.params for t in top_trials])
        results = []
        n_steps = int(sim_time_s / dt)
        for i, params in enumerate(top_params_list):
            try:
                ft = optuna.trial.FixedTrial(params)
                cfg = self._build_trial_config(ft)
                cfg["simulation"]["n_steps"] = n_steps
                m = self._run_quick(cfg)
                if m is None or m.get("runaway"):
                    results.append({"idx": i, "ok": False, "reason": "runaway/fail"})
                    continue
                st_m = m.get("station") or {}
                results.append({
                    "idx": i, "ok": True,
                    "Q_plasma": m.get("avg_Q_plasma", 0.0),
                    "f_ext": m.get("avg_f_ext", 1.0),
                    "P_fusion_W": m.get("avg_P_fusion_W", 0.0),
                    "P_net_W": st_m.get("P_net_W", 0.0),
                })
            except Exception as e:
                results.append({"idx": i, "ok": False, "reason": str(e)})
        return results
    def _make_progress_callback(self, total_trials):
        """Создаёт callback для обновления прогресс-бара Streamlit."""
        def callback(study, trial):
            try:
                best_val = study.best_value if study.best_value is not None else 0
                if self.mode != "pareto" and best_val != 0:
                    progress = (trial.number + 1) / total_trials
                    st.write(f"Trial {trial.number + 1}/{total_trials} | Best P={-best_val/1e6:.2f} MW")
                else:
                    st.write(f"Trial {trial.number + 1}/{total_trials}")
            except Exception:
                pass
        return callback

    def optimize(self, mode: str, n_trials: int = 50,
                 use_surrogate: bool = False,
                 surrogate_trials: int = 500,
                 refine_top: int = 10):
        self.mode = mode
        self._pruned_count = 0

        # v11.5: автозагрузка предиктора, если не установлен
        if self.predictor is None:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "models", "ai_predictor_DT.json")
                if os.path.exists(model_path):
                    from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
                    p = GAKTransformerPredictor()
                    if p.load(model_path):
                        self.predictor = p
                        if p.is_trained:
                            print(f"[AutoLearn] Predictor auto-loaded: {model_path} (exp={p.n_experiences})")
                        else:
                            print(f"[AutoLearn] Predictor loaded (not trained yet, exp={p.n_experiences}) — will continue learning")
                    else:
                        print(f"[AutoLearn] Model file corrupt — loading factory preset")
                        self.predictor = GAKTransformerPredictor.load_default()
                        if self.predictor and self.predictor.is_trained:
                            print(f"[AutoLearn] Factory model loaded (exp={self.predictor.n_experiences}, err={self.predictor.last_error_pct:.1f}%)")
                else:
                    print(f"[AutoLearn] No model file — loading factory preset")
                    from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
                    self.predictor = GAKTransformerPredictor.load_default()
                    if self.predictor and self.predictor.is_trained:
                        print(f"[AutoLearn] Factory model loaded (exp={self.predictor.n_experiences}, err={self.predictor.last_error_pct:.1f}%)")
            except Exception as e:
                print(f"[AutoLearn] Auto-load error: {e}")
                try:
                    from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
                    self.predictor = GAKTransformerPredictor.load_default()
                    if self.predictor and self.predictor.is_trained:
                        print(f"[AutoLearn] Fallback: factory model loaded (exp={self.predictor.n_experiences})")
                    else:
                        print(f"[AutoLearn] Fallback: factory model not trained")
                except Exception as e2:
                    print(f"[AutoLearn] Fallback failed: {e2}")

        # v11: MedianPruner — отсекает плохие trials на ранних шагах
        # v36: судья не трогает триал на разогреве — warmup = 50% прогона
        base_steps = int(self.base_config.get("simulation", {}).get("n_steps", 2500000))
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,                    # первые 10 не прерывать
            n_warmup_steps=int(0.5 * base_steps),   # первые 50% шагов — разогрев плазмы
            interval_steps=max(1, base_steps // 20),
            n_min_trials=3,                         # не судить по 1–2 точкам
        )

        study_name = f"gak_opt_{mode}"
        if mode == "pareto":
            self.study = optuna.create_study(
                directions=["maximize", "maximize"],
                sampler=optuna.samplers.TPESampler(seed=43, multivariate=True, group=True, n_startup_trials=10),
                pruner=pruner,
                storage=self._study_storage_path,
                study_name=study_name,
                load_if_exists=True,
            )
        else:
            self.study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=43, multivariate=True, group=True, n_startup_trials=10),
                pruner=pruner,
                storage=self._study_storage_path,
                study_name=study_name,
                load_if_exists=True,
            )

        # v15.2: учитываем уже завершённые trials из БД
        existing_complete = len([t for t in self.study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE])
        if existing_complete > 0:
            print(f"📂 Продолжаем study '{study_name}': уже {existing_complete} завершённых trials")
            n_trials = max(0, n_trials - existing_complete)
            surrogate_trials = max(0, surrogate_trials - existing_complete)
            if n_trials == 0 and not use_surrogate:
                print(f"✅ Study '{study_name}' уже завершён ({existing_complete} trials) — новых не нужно")
            elif use_surrogate and surrogate_trials == 0:
                print(f"✅ Study '{study_name}' уже имеет {existing_complete} trials — переходим к уточнению")

        # v11: seed — ставим 2 лучших известных конфига в очередь
        # TPE начнёт исследование от них, а не с нуля
        # v15.2: обёрнуто в try/except — seeds уже могут быть в очереди
        # v36: seeds + разведки 1.5e21, с дедупликацией (учитываем WAITING и прошлые триалы)
        SEED_PARAMS_3 = {
            "R_mm": 1198.4811181231032, "h_mm": 14.855574931299106,
            "T_inner": 387.0960840288169, "T_outer": 278.8421639835596,
            "k_thermal": 57.55357861478491, "pl_radius_mm": 607.8498234166981,
            "T_plasma_init": 155669662.20115376, "n_density": 9.60076528186974e20,
            "P_initial": 0.0920612703480718, "pump_speed": 0.0663469993815982,
            "E_field": 5.2261205271136064e17, "gap_nm": 2,
            "external_B": 30.20018059193997,
        }
        # v37: чемпион v36.3 (trial 4 / 31: 1282 МВт) — холодный режим 100.4 МК
        SEED_PARAMS_4 = {
            "R_mm": 1188.3442248257556, "h_mm": 13.278035024627311,
            "T_inner": 383.659372210686, "T_outer": 290.5678515591252,
            "k_thermal": 53.491263411352605, "pl_radius_mm": 609.8304755191414,
            "T_plasma_init": 100401682.7003209, "n_density": 1.4908333970852565e21,
            "P_initial": 0.06307008488692127, "pump_speed": 0.06151920095737816,
            "E_field": 5.772087101959802e17, "gap_nm": 1,
            "external_B": 28.72838584704643,
        }
        # v37: рекорд 1527 МВт — разведка (1.6e21, 615) на базе SEED_PARAMS_4
        SEED_PARAMS_5 = dict(SEED_PARAMS_4)
        SEED_PARAMS_5["n_density"] = 1.6e21
        SEED_PARAMS_5["pl_radius_mm"] = 615.0
        def _probe_n(base, n_new, pl_new=None):
            p = dict(base); p["n_density"] = n_new
            if pl_new is not None: p["pl_radius_mm"] = pl_new
            return p
        queued_cfgs = [BEST_SEED_PARAMS, SEED_PARAMS_2, SEED_PARAMS_3,
                       SEED_PARAMS_4, SEED_PARAMS_5,
                       _probe_n(SEED_PARAMS_4, 1.55e21, 615.0)]
        def _pkey(p):
            return (round(float(p.get("n_density", 0)), 3),
                    round(float(p.get("R_mm", 0)), 1),
                    round(float(p.get("pl_radius_mm", 0)), 1))
        seen_params = set()
        try:
            for t in self.study.get_trials(deepcopy=False):
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.WAITING,
                               optuna.trial.TrialState.RUNNING):
                    seen_params.add(_pkey(t.params))
        except Exception:
            pass
        for seed_params in queued_cfgs:
            try:
                if _pkey(seed_params) in seen_params:
                    continue
                self.study.enqueue_trial(seed_params)
                seen_params.add(_pkey(seed_params))
            except Exception:
                pass

        # ── Суррогатный режим ──
        if use_surrogate and self.predictor and self.predictor.is_trained and surrogate_trials > 0:
            st.info(f"\U0001f916 Суррогатный режим: {surrogate_trials} trials через ИИ + {refine_top} уточнение физикой")

            if mode == "ignition":
                self.study.optimize(self._surrogate_objective_ignition,
                                  n_trials=surrogate_trials, show_progress_bar=True,
                                  callbacks=[self._make_progress_callback(surrogate_trials)])
            elif mode == "gain":
                self.study.optimize(self._surrogate_objective_gain,
                                  n_trials=surrogate_trials, show_progress_bar=True,
                                  callbacks=[self._make_progress_callback(surrogate_trials)])
            elif mode == "pareto":
                self.study.optimize(self._surrogate_objective_pareto,
                                  n_trials=surrogate_trials, show_progress_bar=True,
                                  callbacks=[self._make_progress_callback(surrogate_trials)])

            st.info(f"\U0001f52c Уточнение топ-{refine_top} через полную симуляцию...")
            best_m, best_params = self._refine_top(mode, refine_top)

            if best_m is not None:
                self.results = {
                    "best_params": best_params,
                    "best_value": -float((best_m.get("station") or {}).get("P_net_W",
                                          best_m.get("avg_P_fusion_W", 0))) if mode != "pareto" else 0,
                    "metrics": best_m,
                    "history": [
                        {"number": t.number, "value": t.value}
                        for t in self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                    ],
                }
                st.success(f"✅ Уточнение завершено: P_net={float((best_m.get('station') or {}).get('P_net_W', 0))/1e6:.2f} MW")
            else:
                st.warning("\u26a0\ufe0f Ни один кандидат не прошёл физическую симуляцию")
                self._extract_results()

            # v11.3: сохраняем модель и выходим — не проваливаться в нормальный режим
            if self.predictor is not None and self.predictor.is_trained:
                try:
                    save_path = self._predictor_save_path
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.predictor.save(save_path)
                    print(f"[AutoLearn] Model saved: {save_path} (exp={self.predictor.n_experiences})")
                except Exception as e:
                    print(f"[AutoLearn] Save error: {e}")
            return  # v11.3: суррогат закончен — не запускать нормальный режим

        def objective_ignition(trial):
            cfg = self._build_trial_config(trial)
            if self._prescreen(cfg) == 'reject':
                return 1e10
            try:
                m = self._run_quick(cfg, trial=trial)
            except optuna.TrialPruned:
                self._pruned_count += 1
                raise
            if m is None:
                return 1e10
            if m.get("runaway", False):
                return 1e10
            # v33: метрики зажигания в атрибуты триала (видны в trials_dataframe)
            trial.set_user_attr("Q_plasma", float(m.get("avg_Q_plasma", 0.0)))
            trial.set_user_attr("f_ext", float(m.get("avg_f_ext", 1.0)))
            trial.set_user_attr("P_fusion_W", float(m.get("avg_P_fusion_W", 0.0)))
            trial.set_user_attr("P_electric_W",
                                float((m.get("station") or {}).get("P_electric_W", 0.0)))
            trial.set_user_attr("P_net_W",
                                float((m.get("station") or {}).get("P_net_W", 0.0)))
            trial.set_user_attr("beta", float(m.get("avg_beta", 0.0)))
            # v37: метрики грязной плазмы в атрибуты триала
            trial.set_user_attr("f_He", float(m.get("f_He", 0.0)))
            trial.set_user_attr("Z_eff", float(m.get("Z_eff", 1.0)))
            # v32: constraint - доля внешнего подогрева (зажигание/поддержка)
            if m.get("avg_f_ext", 1.0) > self.F_EXT_MAX:
                return 1e10
            if not m.get("lawson_achieved", False):
                return 1e10
            self._auto_learn(cfg, m)
            # v33: objective — ПОЛНАЯ мощность (Вт), а не плотность (Вт/м3)
            # v37.2: objective — ЧИСТАЯ мощность станции; налог на насосы виден TPE
            P_net_W = float((m.get("station") or {}).get("P_net_W",
                          m.get("avg_P_fusion_W", 0.0)))
            # v36: мягкий штраф за превышение MHD-беты (предел 8%)
            beta = float(m.get("avg_beta", 0.0))
            if beta > self.BETA_MAX:
                P_net_W *= (self.BETA_MAX / beta) ** 2
            return -P_net_W

        def objective_gain(trial):
            cfg = self._build_trial_config(trial)
            if self._prescreen(cfg) == 'reject':
                return 1e10
            try:
                m = self._run_quick(cfg, trial=trial)
            except optuna.TrialPruned:
                self._pruned_count += 1
                raise
            if m is None:
                return 1e10
            if m.get("runaway", False):
                return 1e10
            # v33: метрики зажигания в атрибуты триала (видны в trials_dataframe)
            trial.set_user_attr("Q_plasma", float(m.get("avg_Q_plasma", 0.0)))
            trial.set_user_attr("f_ext", float(m.get("avg_f_ext", 1.0)))
            trial.set_user_attr("P_fusion_W", float(m.get("avg_P_fusion_W", 0.0)))
            trial.set_user_attr("P_electric_W",
                                float((m.get("station") or {}).get("P_electric_W", 0.0)))
            trial.set_user_attr("P_net_W",
                                float((m.get("station") or {}).get("P_net_W", 0.0)))
            # v37: метрики грязной плазмы
            trial.set_user_attr("f_He", float(m.get("f_He", 0.0)))
            trial.set_user_attr("Z_eff", float(m.get("Z_eff", 1.0)))
            # v32: constraint - доля внешнего подогрева (зажигание/поддержка)
            if m.get("avg_f_ext", 1.0) > self.F_EXT_MAX:
                return 1e10
            if not m.get("lawson_achieved", False):
                return 1e10
            self._auto_learn(cfg, m)
            # v33: objective — ПОЛНАЯ мощность (Вт), а не плотность (Вт/м3)
            # v37.2: objective — ЧИСТАЯ мощность станции; налог на насосы виден TPE
            P_net_W = float((m.get("station") or {}).get("P_net_W",
                          m.get("avg_P_fusion_W", 0.0)))
            # v36: мягкий штраф за превышение MHD-беты (предел 8%)
            beta = float(m.get("avg_beta", 0.0))
            if beta > self.BETA_MAX:
                P_net_W *= (self.BETA_MAX / beta) ** 2
            return -P_net_W

        def objective_pareto(trial):
            cfg = self._build_trial_config(trial)
            if self._prescreen(cfg) == 'reject':
                return [-1e10, -1e10]
            try:
                m = self._run_quick(cfg, trial=trial)
            except optuna.TrialPruned:
                self._pruned_count += 1
                raise
            if m is None or m.get("runaway", False):
                return [-1e10, -1e10]
            self._auto_learn(cfg, m)
            return [m.get("avg_pair_rate", 0), m.get("avg_Q_plasma", 0)]

        if n_trials > 0:
            if mode == "ignition":
                self.study.optimize(objective_ignition, n_trials=n_trials, show_progress_bar=True,
                                   callbacks=[self._make_progress_callback(n_trials), self._make_stop_callback()],
                                   catch=(Exception,))
            elif mode == "gain":
                self.study.optimize(objective_gain, n_trials=n_trials, show_progress_bar=True,
                                callbacks=[self._make_progress_callback(n_trials), self._make_stop_callback()],
                                   catch=(Exception,))
            elif mode == "pareto":
                self.study.optimize(objective_pareto, n_trials=n_trials, show_progress_bar=True,
                                 callbacks=[self._make_progress_callback(n_trials), self._make_stop_callback()],
                                   catch=(Exception,))
            else:
                raise ValueError(f"Неизвестный режим: {mode}")
        else:
            print(f"⏭️ Study '{study_name}' уже завершён — пропускаем оптимизацию, переходим к результатам")

        # Логирование статистики
        stats = self._prescreen_stats
        total_screened = stats["screened"]
        total_rejected = stats["rejected"]
        total_passed = stats["passed"]
        pruned = self._pruned_count
        if total_screened > 0:
            reject_pct = total_rejected / total_screened * 100
            try:
                print(f"  Pre-screen: {total_screened} screened, "
                      f"{total_rejected} rejected ({reject_pct:.0f}%), "
                      f"{total_passed} to physics")
            except Exception:
                pass
        if pruned > 0:
            try:
                print(f"  Pruner: {pruned} trials pruned early (saved ~{pruned * 27}s)")
            except Exception:
                pass

        self._extract_results()

        # v11.5: сохраняем модель после автообучения (даже если not trained — опыт ценен)
        if self.predictor is not None and self.predictor.n_experiences > 0:
            try:
                save_path = self._predictor_save_path
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.predictor.save(save_path)
                print(f"[AutoLearn] Model saved: {save_path} (exp={self.predictor.n_experiences}, trained={self.predictor.is_trained})")
            except Exception as e:
                print(f"[AutoLearn] Save error: {e}")



    def optimize_cascade(self, n_trials_phase1=200, n_trials_phase2=200):
        """v15: Каскадная оптимизация — зажигание → горение.
        Phase 1: широкий поиск конфигов, которые зажигают плазму.
        Phase 2: горение с фиксированными общими параметрами из топ-N Phase 1."""
        self.mode = 'cascade'
        self._pruned_count = 0

        # Автозагрузка предиктора
        if self.predictor is None:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "models", "ai_predictor_DT.json")
                if os.path.exists(model_path):
                    from physical_modules.ai_transformer_predictor import GAKTransformerPredictor
                    p = GAKTransformerPredictor()
                    if p.load(model_path):
                        self.predictor = p
                        print(f"[Cascade] Predictor loaded (exp={p.n_experiences})")
                else:
                    self.predictor = GAKTransformerPredictor.load_default()
                    print("[Cascade] Factory predictor loaded")
            except Exception as e:
                print(f"[Cascade] Predictor load error: {e}")

        # === PHASE 1: ЗАЖИГАНИЕ ===
        st.info("🔵 Phase 1: Зажигание плазмы (поиск рабочих конфигов)")

        pruner1 = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20000,
            interval_steps=20000,
        )

        study1 = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=43, multivariate=True, group=True, n_startup_trials=10),
            pruner=pruner1,
            storage=self._study_storage_path,
            study_name="gak_cascade_phase1",
            load_if_exists=True,
        )

        # v15.2: учитываем уже завершённые trials
        existing_p1 = len([t for t in study1.trials
                           if t.state == optuna.trial.TrialState.COMPLETE])
        if existing_p1 > 0:
            print(f"📂 Продолжаем Phase 1: уже {existing_p1} завершённых trials")
            n_trials_phase1 = max(0, n_trials_phase1 - existing_p1)

        # Seeds для Phase 1
        for sp in [BEST_SEED_PARAMS, SEED_PARAMS_2]:
            try:
                study1.enqueue_trial(sp)
            except Exception:
                pass

        def objective_ignition_cascade(trial):
            cfg = self._build_cascade_config(trial, 'ignition')
            if self._prescreen(cfg) == 'reject':
                return 1e10
            try:
                m = self._run_quick(cfg, trial=trial)
            except optuna.TrialPruned:
                self._pruned_count += 1
                raise
            if m is None or m.get("runaway"):
                return 1e10
            self._auto_learn(cfg, m)
            # Лексикографический objective:
            # 1. lawson_achieved (×1000) — главное условие
            # 2. T_keV (×10) — температура как вторичный критерий
            # 3. Q (×1) — запас энергии как третий
            lawson = 1000.0 if m.get("lawson_achieved") else 0.0
            T_keV = m.get("T_keV", 0) * 10.0
            Q = m.get("avg_Q_plasma", 0)
            if m.get("avg_f_ext", 1.0) > self.F_EXT_MAX:
                return 1e10
            score = lawson + T_keV + Q
            return -score

        def progress1(study, trial):
            try:
                best_val = study.best_value if study.best_value is not None else 0
                progress = (trial.number + 1) / n_trials_phase1
                st.write(f"Phase 1 | Trial {trial.number + 1}/{n_trials_phase1} | Best score={-best_val:.2f}")
            except Exception:
                pass

        study1.optimize(objective_ignition_cascade, n_trials=n_trials_phase1,
                       show_progress_bar=True, callbacks=[progress1])

        st.info(f"Phase 1 завершена. Лучший score: {-study1.best_value:.2f}")

        # === ОТБОР ТОП-N ===
        st.info("🔵 Отбор топ-10 кандидатов из Phase 1")
        top_trials = self._select_top_cascade(study1, n_top=10)

        if len(top_trials) == 0:
            st.error("Ни один конфиг из Phase 1 не зажёг плазму")
            return

        st.write(f"Отобрано {len(top_trials)} кандидатов для Phase 2")

        # === PHASE 2: ГОРЕНИЕ ===
        st.info("🔴 Phase 2: Горение плазмы (максимизация Q со стабильностью)")

        best_overall_m = None
        best_overall_params = None
        best_overall_Q = -1e18
        all_phase2_results = []

        for cand_idx, base_trial in enumerate(top_trials):
            shared_params = dict(base_trial.params)

            st.write(f"  Кандидат {cand_idx + 1}/{len(top_trials)}: "
                     f"R={shared_params.get('R_mm', 0):.0f}mm "
                     f"B={shared_params.get('external_B', 0):.1f}T "
                     f"n={shared_params.get('n_density', 0):.2e}")

            pruner2 = optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=20000,
                interval_steps=20000,
            )

            study2_name = f"gak_cascade_phase2_cand{cand_idx}"
            study2 = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(
                    seed=42 + cand_idx,
                    multivariate=True, group=True, n_startup_trials=3
                ),
                pruner=pruner2,
                storage=self._study_storage_path,
                study_name=study2_name,
                load_if_exists=True,
            )
            existing_p2 = len([t for t in study2.trials
                              if t.state == optuna.trial.TrialState.COMPLETE])
            if existing_p2 > 0:
                print(f"📂 Продолжаем Phase 2 cand{cand_idx}: уже {existing_p2} trials")

            def objective_burn(trial, _shared=shared_params):
                cfg_ign, _ = self._build_cascade_config(
                    optuna.trial.FixedTrial(_shared), 'ignition')
                cfg_burn, t_switch = self._build_cascade_config(
                    trial, 'burn', shared_params=_shared)
                try:
                    m = self._run_quick_cascade(cfg_ign, cfg_burn, t_switch, trial=trial)
                except optuna.TrialPruned:
                    raise
                if m is None or m.get("runaway"):
                    return 1e10
                self._auto_learn(cfg_burn, m)
                return -m.get("Q", 0)

            def progress2(study, trial):
                try:
                    best_val = study.best_value if study.best_value is not None else 0
                    st.write(f"  Cand {cand_idx+1} | Trial {trial.number + 1}/{n_trials_phase2} | Q={-best_val:.4f}")
                except Exception:
                    pass

            trials_p2 = max(20, n_trials_phase2 // len(top_trials))
            study2.optimize(objective_burn, n_trials=trials_p2,
                          show_progress_bar=True, callbacks=[progress2])

            # Лучший результат для этого кандидата
            if study2.best_value is not None and study2.best_value < 1e9:
                best_p2 = study2.best_trial
                ft = optuna.trial.FixedTrial(best_p2.params)
                cfg_ign, _ = self._build_cascade_config(
                    optuna.trial.FixedTrial(shared_params), 'ignition')
                cfg_burn, t_switch = self._build_cascade_config(ft, 'burn', shared_params=shared_params)
                m = self._run_quick_cascade(cfg_ign, cfg_burn, t_switch)
                if m is not None and not m.get("runaway"):
                    Q = m.get("Q", 0)
                    all_phase2_results.append({
                        'cand_idx': cand_idx,
                        'params': {**shared_params, **best_p2.params},
                        'metrics': m,
                        'Q': Q,
                    })
                    if Q > best_overall_Q:
                        best_overall_Q = Q
                        best_overall_m = m
                        best_overall_params = {**shared_params, **best_p2.params}
                        st.success(f"  Новый рекорд: Q={Q:.4f} (cand {cand_idx+1})")

        # === ИТОГ ===
        if best_overall_m is not None:
            self.results = {
                "best_params": best_overall_params,
                "best_value": -best_overall_Q,
                "metrics": best_overall_m,
                "history": [
                    {"number": i, "value": -r['Q']}
                    for i, r in enumerate(all_phase2_results)
                ],
                "all_results": all_phase2_results,
            }
            st.success(f"✅ Каскад завершён: Q={best_overall_Q:.4f} "
                       f"(T_burn={best_overall_m.get('T_burn', 0):.2e}K, "
                       f"stability_cv={best_overall_m.get('Q_stability_cv', 0):.3f})")
        else:
            st.warning("⚠️ Ни один кандидат не дал стабильного горения")
            self.results = None

        # Сохранение модели
        if self.predictor is not None and self.predictor.n_experiences > 0:
            try:
                save_path = self._predictor_save_path
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.predictor.save(save_path)
                print(f"[Cascade] Model saved (exp={self.predictor.n_experiences})")
            except Exception as e:
                print(f"[Cascade] Save error: {e}")


    def _auto_learn(self, cfg, m):
        """Автообучение суррогата на реальных результатах физики.
        v11.5: опыт добавляется каждый trial, обучение — каждые _auto_learn_interval trials.
        Это снижает overhead с ~16с/trial до ~3с/trial."""
        if self.predictor is None or m is None:
            return
        try:
            history = {
                "T_plasma": m.get("avg_T", 0.0),
                "P_gas": m.get("avg_P_fusion", 0.0),
                "pair_rate": m.get("avg_pair_rate", 0.0),
                "ratio": m.get("avg_Q", 0.0),  # v25: фикс — суррогат учится Q, не P_heat/P_cond
            }
            # v11.5: добавляем опыт каждый trial (быстро, ~0.1с)
            added = self.predictor.add_experience(cfg, history)
            
            self._auto_learn_counter += 1
            
            # Обучаем только каждые N trials
            if self._auto_learn_counter >= self._auto_learn_interval and self.predictor.n_experiences >= 2:
                loss = self.predictor.train(cfg, history)
                self._auto_learn_counter = 0
                try:
                    print(f"[AutoLearn] TRAINED exp={self.predictor.n_experiences} "
                          f"trained={self.predictor.is_trained} loss={loss:.6f} "
                          f"err={self.predictor.last_error_pct:.1f}%")
                except Exception:
                    pass
            else:
                try:
                    if added:
                        print(f"[AutoLearn] exp buffered: {self.predictor.n_experiences} "
                              f"(train in {self._auto_learn_interval - self._auto_learn_counter})")
                except Exception:
                    pass
        except Exception as e:
            try:
                print(f"[AutoLearn] error: {e}")
            except Exception:
                pass


    def _surrogate_predict(self, cfg):
        """Прогон через predict() трансформера вместо _run_quick()."""
        if self.predictor is None or not self.predictor.is_trained:
            return None
        try:
            forecast = self.predictor.predict(cfg)
            if forecast is None:
                return None
            return {
                "T_plasma": forecast.get("T_plasma", 0.0),
                "ratio": forecast.get("ratio", 0.0),
                "pair_rate": forecast.get("pair_rate", 0.0),
                "P_gas": forecast.get("P_gas", 0.0),
            }
        except Exception:
            return None

    def _surrogate_objective_ignition(self, trial):
        cfg = self._build_trial_config(trial)
        pred = self._surrogate_predict(cfg)
        if pred is None:
            return 1e10
        T = pred["T_plasma"]
        if T > self.T_CAP or T < 1e6:
            return 1e10
        ratio = pred["ratio"]
        # v13: анти-галлюцинация — жёсткий кап + trust radius
        if ratio > self.SURROGATE_CAP or ratio < 0:
            return 1e10
        # Штраф за экстраполяцию за пределы обучающей выборки
        if ratio > self.MAX_TRAIN_Q:
            penalty = (ratio - self.MAX_TRAIN_Q) ** 2 * 2.0
            return -(ratio - penalty)
        return -ratio

    def _surrogate_objective_gain(self, trial):
        cfg = self._build_trial_config(trial)
        pred = self._surrogate_predict(cfg)
        if pred is None:
            return 1e10
        T = pred["T_plasma"]
        if T > self.T_CAP or T < 1e6:
            return 1e10
        ratio = pred["ratio"]
        # v13: анти-галлюцинация — жёсткий кап + trust radius
        if ratio > self.SURROGATE_CAP or ratio < 0:
            return 1e10
        if ratio > self.MAX_TRAIN_Q:
            penalty = (ratio - self.MAX_TRAIN_Q) ** 2 * 2.0
            return -(ratio - penalty)
        return -ratio

    def _surrogate_objective_pareto(self, trial):
        cfg = self._build_trial_config(trial)
        pred = self._surrogate_predict(cfg)
        if pred is None:
            return [-1e10, -1e10]
        T = pred["T_plasma"]
        if T > self.T_CAP or T < 1e6:
            return [-1e10, -1e10]
        ratio = pred["ratio"]
        # v13: анти-галлюцинация — жёсткий кап + trust radius
        if ratio > self.SURROGATE_CAP or ratio < 0:
            return [-1e10, -1e10]
        if ratio > self.MAX_TRAIN_Q:
            penalty = (ratio - self.MAX_TRAIN_Q) ** 2 * 2.0
            return [pred["pair_rate"], ratio - penalty]
        return [pred["pair_rate"], ratio]


    def _select_top_cascade(self, study, n_top=10):
        """v15: Отбор топ-N из Phase 1 по разным критериям.
        3 лучших по lawson, 3 по T_plasma, 4 по разнообразию."""
        all_trials = [t for t in study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                      if t.value < 1e9]  # отбрасываем 1e10 (мусор)

        if len(all_trials) < n_top:
            return all_trials

        # Создаём словарь trial → metrics (перебегаем _run_quick с FixedTrial)
        trial_metrics = []
        for t in all_trials:
            ft = optuna.trial.FixedTrial(t.params)
            cfg = self._build_cascade_config(ft, 'ignition')
            m = self._run_quick(cfg)
            if m is None or m.get("runaway"):
                continue
            trial_metrics.append({
                'trial': t,
                'params': t.params,
                'lawson': m.get('lawson_product', 0),
                'lawson_achieved': m.get('lawson_achieved', False),
                'T_plasma': m.get('avg_T', 0),
                'Q': m.get('Q', 0),
            })

        if len(trial_metrics) < n_top:
            return [tm['trial'] for tm in trial_metrics]

        selected = []
        selected_params = []

        # 3 лучших по lawson
        by_lawson = sorted(trial_metrics, key=lambda x: x['lawson'], reverse=True)
        for tm in by_lawson:
            if len(selected) >= 3:
                break
            selected.append(tm)
            selected_params.append(tm['params'])

        # 3 лучших по T_plasma (не из уже выбранных)
        by_T = sorted([tm for tm in trial_metrics if tm not in selected],
                      key=lambda x: x['T_plasma'], reverse=True)
        for tm in by_T:
            if len(selected) >= 6:
                break
            selected.append(tm)
            selected_params.append(tm['params'])

        # 4 по разнообразию (максимально удалённые в пространстве параметров)
        remaining = [tm for tm in trial_metrics if tm not in selected]
        # Нормализуем параметры для расчёта расстояния
        param_keys = ['R_mm', 'h_mm', 'T_plasma_init', 'n_density', 'external_B', 'pl_radius_mm']
        all_vals = np.array([[tm['params'].get(k, 0) for k in param_keys] for tm in remaining])
        if len(all_vals) > 0:
            # z-score нормализация
            means = all_vals.mean(axis=0)
            stds = all_vals.std(axis=0)
            stds[stds < 1e-30] = 1.0
            norm_vals = (all_vals - means) / stds

            # Жадный выбор: добавляем точку, наиболее удалённую от уже выбранных
            chosen_indices = []
            for _ in range(min(4, len(remaining))):
                best_idx = -1
                best_dist = -1
                for i in range(len(remaining)):
                    if i in chosen_indices:
                        continue
                    if not chosen_indices:
                        best_idx = i
                        best_dist = 0
                        continue
                    # Минимальное расстояние до выбранных
                    min_dist = min(np.sum((norm_vals[i] - norm_vals[j])**2)
                                   for j in chosen_indices)
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_idx = i
                if best_idx >= 0:
                    chosen_indices.append(best_idx)

            for idx in chosen_indices:
                if len(selected) >= n_top:
                    break
                selected.append(remaining[idx])

        return [tm['trial'] for tm in selected[:n_top]]


    def _refine_top(self, mode, n_refine):
        """Уточнение топ-N кандидатов через полную симуляцию.
        Логирует MLP vs физика для Дипа."""
        if mode == "pareto":
            trials = sorted(self.study.best_trials,
                          key=lambda t: t.values[1], reverse=True)[:n_refine]
        else:
            all_trials = self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            trials = sorted(all_trials, key=lambda t: t.value)[:n_refine]

        best_m = None
        best_score = -1e18
        best_params = None
        self._refine_log_data = []

        for idx, t in enumerate(trials):
            trial = optuna.trial.FixedTrial(t.params)
            cfg = self._build_trial_config(trial)

            # MLP prediction for comparison
            mlp_pred = self._surrogate_predict(cfg)
            mlp_ratio = mlp_pred.get("ratio", 0) if mlp_pred else None
            mlp_Q = -t.value if mode != "pareto" else t.values[1]

            m = self._run_quick(cfg)
            if m is None or m.get("runaway"):
                continue

            self._auto_learn(cfg, m)

            real_ratio = m.get("avg_ratio", 0)
            real_Q = m.get("avg_Q", m.get("Q", 0))
            real_f_ext = m.get("avg_f_ext", 0)

            # Log MLP vs physics
            ratio_diff = None
            Q_diff = None
            if mlp_ratio is not None and abs(real_ratio) > 1e-30:
                ratio_diff = abs(mlp_ratio - real_ratio) / abs(real_ratio) * 100
            if mlp_Q is not None and abs(real_Q) > 1e-10:
                Q_diff = abs(mlp_Q - real_Q) / abs(real_Q) * 100

            self._refine_log_data.append({
                "trial": idx,
                "MLP ratio": f"{mlp_ratio:.4e}" if mlp_ratio is not None else "N/A",
                "Физика ratio": f"{real_ratio:.4e}",
                "MLP Q": f"{mlp_Q:.4f}" if mlp_Q is not None else "N/A",
                "Физика Q": f"{real_Q:.4f}",
                "Разница ratio, %": f"{ratio_diff:.1f}" if ratio_diff is not None else "N/A",
                "Разница Q, %": f"{Q_diff:.1f}" if Q_diff is not None else "N/A",
                "f_ext": f"{real_f_ext:.3f}",
            })

            if mode == "pareto":
                score = m.get("avg_pair_rate", 0)
            else:
                score = float((m.get("station") or {}).get("P_net_W",
                                  m.get("avg_P_fusion_W", 0)))
            if score > best_score:
                best_score = score
                best_m = m
                best_params = t.params

        return best_m, best_params

    def _extract_results(self):
        if self.mode == "pareto":
            trials = self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            self.results = {
                "trials": [
                    {"params": t.params, "values": t.values, "number": t.number}
                    for t in trials
                ],
                "best_trials": [
                    {"params": t.params, "values": t.values, "number": t.number}
                    for t in self.study.best_trials
                ],
            }
        else:
            best = self.study.best_trial
            trial = optuna.trial.FixedTrial(best.params)
            cfg = self._build_trial_config(trial)
            m = self._run_quick(cfg)
            self.results = {
                "best_params": best.params,
                "best_value": best.value,
                "metrics": m,
                "history": [
                    {"number": t.number, "value": t.value}
                    for t in self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                ],
            }

    def get_results_summary(self):
        if self.results is None:
            return None

        if self.mode == "pareto":
            rows = []
            for t in self.results["best_trials"]:
                p = t["params"]
                rows.append({
                    "Радиус сф., мм": f"{p.get('R_mm', 0):.1f}",
                    "Радиус плазмы, мм": f"{p.get('pl_radius_mm', 0):.1f}",
                    "B, Тл": f"{p.get('external_B', 0):.1f}",
                    "T_init, MK": f"{p.get('T_plasma_init', 0)/1e6:.1f}",
                    "Топливо": self.FUEL_TYPES[self.FIXED_FUEL_IDX],
                    "pair_rate": f"{t['values'][0]:.3e}",
                    "Q": f"{t['values'][1]:.4f}",
                })
            return pd.DataFrame(rows)

        p = self.results["best_params"]
        m = self.results["metrics"]
        fuel = self.FUEL_TYPES[self.FIXED_FUEL_IDX]
        rows = [
            ("Радиус сферы, мм", f"{p.get('R_mm', 0):.2f}"),
            ("Толщина стенки, мм", f"{p.get('h_mm', 0):.3f}"),
            ("T внутренняя, K", f"{p.get('T_inner', 0):.1f}"),
            ("T внешняя, K", f"{p.get('T_outer', 0):.1f}"),
            ("Теплопроводность", f"{p.get('k_thermal', 0):.2f}"),
            ("Радиус плазмы, мм", f"{p.get('pl_radius_mm', 0):.3f}"),
            ("T плазмы (init), K", f"{p.get('T_plasma_init', 0):.3e}"),
            ("Плотность, м^-3", f"{p.get('n_density', 0):.3e}"),
            ("Топливо", fuel),
            ("Давление старт, Па", f"{p.get('P_initial', 0):.3e}"),
            ("Откачка, м3/с", f"{p.get('pump_speed', 0):.4e}"),
            ("E-поле, В/м", f"{p.get('E_field', 0):.3e}"),
            ("Зазор, нм", f"{p.get('gap_nm', 0):.1f}"),
            ("B внешнее, Тл", f"{p.get('external_B', 0):.1f}"),
            ("", ""),
            ("=== Метрики ===", ""),
            ("Q-фактор (средн.)", f"{m.get('avg_Q', m.get('Q', 0)):.6f}" if m else "N/A"),
            ("Lawson n*T*tau", f"{m.get('lawson_product', 0):.3e}" if m else "N/A"),
            ("Порог Лоусона", f"{m.get('lawson_threshold', 0):.3e}" if m else "N/A"),
            ("Lawson/threshold", f"{m.get('lawson_product', 0)/m.get('lawson_threshold', 1):.4f}" if m else "N/A"),
            ("Лоусон достигнут?", "ДА" if m and m.get("lawson_achieved") else "НЕТ"),
            ("Runaway (T > cap)?", "ДА" if m and m.get("runaway") else "НЕТ"),
            ("T плазмы (равн.), K", f"{m.get('avg_T', m.get('T_plasma', 0)):.3e}" if m else "N/A"),
            ("T_keV", f"{m.get('T_keV', 0):.2f}" if m else "N/A"),
            ("P_fusion, Вт (полная)", f"{m.get('avg_P_fusion_W', 0):.4e}" if m else "N/A"),
            ("P_alpha, Вт (полная)", f"{m.get('avg_P_alpha_W', 0):.4e}" if m else "N/A"),
            ("P_cond, Вт (полная)", f"{m.get('avg_P_cond_W', 0):.4e}" if m else "N/A"),
            ("P_rad, Вт (полная)", f"{m.get('avg_P_rad_W', 0):.4e}" if m else "N/A"),
            ("P_sync, Вт (полная)", f"{m.get('avg_P_sync_W', 0):.4e}" if m else "N/A"),
            ("tau_E, с", f"{m.get('avg_tau_E', 0):.4e}" if m else "N/A"),
            ("pair_rate (средн.)", f"{m.get('avg_pair_rate', 0):.4e}" if m else "N/A"),
            ("ratio (диагн.)", f"{m.get('avg_ratio', 0):.4f}" if m else "N/A"),
            ("B внешнее (метр.), Тл", f"{m.get('external_B', 0):.1f}" if m else "N/A"),
            ("Радиус плазмы (факт.), м", f"{m.get('plasma_radius_actual_m', 0):.4f}" if m else "N/A"),
            ("V плазмы, м3", f"{m.get('V_plasma', 0):.4e}" if m else "N/A"),
            ("Шагов выполнено", f"{m.get('steps_done', 0)}" if m else "N/A"),
            ("Время симуляции, с", f"{m.get('sim_time', 0):.4f}" if m else "N/A"),
            ("Ср. dt_actual, с", f"{m.get('avg_dt_actual', 0):.4e}" if m else "N/A"),
            ("Q_plasma (P_alpha/P_loss)", f"{m.get('avg_Q_plasma', 0):.4f}" if m else "N/A"),
            ("f_ext (внеш. подогрев)", f"{m.get('avg_f_ext', 0):.4f}" if m else "N/A"),
            ("Доля золы f_He (v37)", f"{m.get('f_He', 0):.4f}" if m else "N/A"),
            ("n_fuel (после разбавления)", f"{m.get('n_fuel', 0):.3e}" if m else "N/A"),
            ("Z_eff (вольфрам c_W=1e-4)", f"{m.get('Z_eff', 1.0):.4f}" if m else "N/A"),
            ("P_electric, Вт", f"{(m.get('station') or {}).get('P_electric_W', 0):.4e}" if m else "N/A"),
            ("P_net, Вт", f"{(m.get('station') or {}).get('P_net_W', 0):.4e}" if m else "N/A"),
            ("Тритий нетто, кг/год", f"{(m.get('station') or {}).get('m_T_net_kg_year', 0):.3f}" if m else "N/A"),
        ]
        return pd.DataFrame(rows, columns=["Параметр", "Значение"])

    def plot_convergence(self, fig=None, ax=None):
        if self.mode == "pareto":
            return self._plot_pareto(fig, ax)

        if self.results is None or "history" not in self.results:
            return None

        hist = self.results["history"]
        if not hist:
            return None

        df = pd.DataFrame(hist)

        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        if self.mode == "ignition":
            ax.plot(df["number"], -df["value"], color="crimson", linewidth=2)
            ax.set_ylabel("P_fusion, МВт")
            ax.set_title("Сходимость: P_fusion -> max (режим зажигания)")
        elif self.mode == "gain":
            ax.plot(df["number"], -df["value"], color="navy", linewidth=2)
            ax.set_ylabel("P_fusion, МВт")
            ax.set_title("Сходимость: P_fusion -> max (режим усиления)")

        ax.set_xlabel("Попытка №")
        ax.grid(True, alpha=0.3, linestyle="--")
        return fig

    def _plot_pareto(self, fig=None, ax=None):
        if self.results is None:
            return None

        trials = self.results["trials"]
        best_numbers = {t["number"] for t in self.results["best_trials"]}

        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        for t in trials:
            color = "gold" if t["number"] in best_numbers else "steelblue"
            size = 80 if t["number"] in best_numbers else 25
            alpha = 1.0 if t["number"] in best_numbers else 0.5
            ax.scatter(t["values"][1], t["values"][0],
                       color=color, s=size, alpha=alpha, edgecolors="black", linewidth=0.5)

        ax.set_xlabel("Q-фактор")
        ax.set_ylabel("pair_rate (рождение пар)")
        ax.set_title("Фронт Парето: pair_rate vs Q")
        ax.grid(True, alpha=0.3, linestyle="--")
        return fig

    def get_best_params_details(self):
        """Alias for get_results_summary() — совместимость с UI."""
        return self.get_results_summary()

    def get_refine_log(self):
        """Лог MLP vs физика из топ-N."""
        return getattr(self, '_refine_log_data', [])

    def export_best_csv(self) -> pd.DataFrame:
        if self.results is None:
            return None

        if self.mode == "pareto":
            rows = []
            for i, t in enumerate(self.results["best_trials"]):
                p = t["params"]
                row = {
                    "trial": i,
                    "R_mm": p.get("R_mm", 0),
                    "h_mm": p.get("h_mm", 0),
                    "T_inner": p.get("T_inner", 0),
                    "T_outer": p.get("T_outer", 0),
                    "k_thermal": p.get("k_thermal", 0),
                    "pl_radius_mm": p.get("pl_radius_mm", 0),
                    "T_plasma_init": p.get("T_plasma_init", 0),
                    "n_density": p.get("n_density", 0),
                    "fuel_type": self.FUEL_TYPES[self.FIXED_FUEL_IDX],
                    "P_initial": p.get("P_initial", 0),
                    "pump_speed": p.get("pump_speed", 0),
                    "E_field": p.get("E_field", 0),
                    "gap_nm": p.get("gap_nm", 0),
                    "external_B": p.get("external_B", 0),
                    "pair_rate": t["values"][0],
                    "Q": t["values"][1],
                }
                rows.append(row)
            return pd.DataFrame(rows)

        p = self.results["best_params"]
        m = self.results["metrics"]
        fuel = self.FUEL_TYPES[self.FIXED_FUEL_IDX]
        row = {
            "R_mm": p.get("R_mm", 0),
            "h_mm": p.get("h_mm", 0),
            "T_inner": p.get("T_inner", 0),
            "T_outer": p.get("T_outer", 0),
            "k_thermal": p.get("k_thermal", 0),
            "pl_radius_mm": p.get("pl_radius_mm", 0),
            "T_plasma_init": p.get("T_plasma_init", 0),
            "n_density": p.get("n_density", 0),
            "fuel_type": fuel,
            "P_initial": p.get("P_initial", 0),
            "pump_speed": p.get("pump_speed", 0),
            "E_field": p.get("E_field", 0),
            "gap_nm": p.get("gap_nm", 0),
            "external_B": p.get("external_B", 0),
            "Q": m.get("avg_Q", m.get("Q", 0)) if m else 0,
            "lawson": m.get("lawson_product", 0) if m else 0,
            "lawson_threshold": m.get("lawson_threshold", 0) if m else 0,
            "lawson_achieved": m.get("lawson_achieved", False) if m else False,
            "runaway": m.get("runaway", False) if m else False,
            "T_plasma_eq": m.get("avg_T", m.get("T_plasma", 0)) if m else 0,
            "P_fusion_W": m.get("avg_P_fusion_W", 0) if m else 0,
            "V_plasma": m.get("V_plasma", 0) if m else 0,
            "P_alpha_W": m.get("avg_P_alpha_W", 0) if m else 0,
            "P_cond_W": m.get("avg_P_cond_W", 0) if m else 0,
            "P_rad_W": m.get("avg_P_rad_W", 0) if m else 0,
            "P_sync_W": m.get("avg_P_sync_W", 0) if m else 0,
            "tau_E": m.get("avg_tau_E", 0) if m else 0,
            "plasma_radius_actual_m": m.get("plasma_radius_actual_m", 0) if m else 0,
            "steps_done": m.get("steps_done", 0) if m else 0,
            "sim_time": m.get("sim_time", 0) if m else 0,
            "avg_dt_actual": m.get("avg_dt_actual", 0) if m else 0,
            "Q_plasma": m.get("avg_Q_plasma", 0) if m else 0,
            "f_ext": m.get("avg_f_ext", 0) if m else 0,
            "f_He": m.get("f_He", 0) if m else 0,
            "n_fuel": m.get("n_fuel", 0) if m else 0,
            "Z_eff": m.get("Z_eff", 1.0) if m else 1.0,
            "P_electric_W": (m.get("station") or {}).get("P_electric_W", 0) if m else 0,
            "P_net_W": (m.get("station") or {}).get("P_net_W", 0) if m else 0,
            "T_net_kg_year": (m.get("station") or {}).get("m_T_net_kg_year", 0) if m else 0,
        }
        return pd.DataFrame([row])