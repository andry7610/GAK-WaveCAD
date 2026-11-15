"""
GAK-WaveCAD — Optuna-оптимизатор v46Q (флаги экономики + ось B=15 + фикс ямы разведки).

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


# === v46G: SEED-AWARE PRUNER ===
class SeedAwarePruner(optuna.pruners.BasePruner):
    """v46G: медиана отсечения — только по не-сидовым COMPLETE-триалам.
    Сиды (user_attrs["seed"]) — эталонные точки смоука: в выборку отсечения
    не входят и сами не прунятся никогда (смоук обязан доходить до финала
    бит в бит). Защитный режим: не-сидовых COMPLETE меньше n_min_trials —
    прунер молчит, кандидаты добегают до финала (run9: не-сидовых COMPLETE
    ноль — TPE нечему было учиться)."""

    def __init__(self, n_startup_trials=10, n_warmup_steps=0, interval_steps=1,
                 n_min_trials=3):
        self.n_startup_trials = int(n_startup_trials)
        self.n_warmup_steps = int(n_warmup_steps)
        self.interval_steps = max(1, int(interval_steps))
        self.n_min_trials = max(2, int(n_min_trials))

    def prune(self, study, trial):
        try:
            complete = [t for t in study.get_trials(deepcopy=False)
                        if t.state == optuna.trial.TrialState.COMPLETE]
        except Exception:
            return False
        if len(complete) < self.n_startup_trials:
            return False
        # сиды — не планка: смоук не задаёт планку отсечения
        nonseed = [t for t in complete
                   if not (t.user_attrs or {}).get("seed", False)]
        if len(nonseed) < self.n_min_trials:
            return False
        # сам сид не прунится никогда — смоук бит в бит
        if (trial.user_attrs or {}).get("seed", False):
            return False
        step = trial.last_step
        if step is None or step < self.n_warmup_steps:
            return False
        if step % self.interval_steps != 0:
            return False
        vals = []
        for t in nonseed:
            if t.number == trial.number:
                continue
            v = t.intermediate_values.get(step)
            if v is not None:
                vals.append(float(v))
        if len(vals) < self.n_min_trials:
            return False
        med = float(np.median(vals))
        return float(trial.intermediate_values[step]) > med
# === END SEED-AWARE PRUNER ===


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
        "R_mm":           (1200, 5500),    # v45R: реалистичное окно (станция, а не игрушечный шар)
        "h_mm":           (10.0, 18.0),    # v21: золотая зона (trial 44: 14.33)
        "T_inner":        (360.0, 420.0),  # v21: золотая зона (trial 44: 387.74)
        "T_outer":        (273.0, 300.0),   # v21: FIX — было (273,280), но seed=287.89!
        "k_thermal":      (20.0, 90.0),     # v26: расширено вниз — пробить брейкивен
        "pl_radius_mm":    (500, 2860),      # v45R: кламп 620 снят (0.52*5500)
        "T_plasma_init":  (1e8, 1.6e8),     # v36.2: 1.5e8->1.6e8 (над сид-чемпионом 155.7 МК)
        "n_density":      (1e20, 3.6e21),  # v45R: потолок динамический (0.8*n_G), низ 1e20
        "fuel_type_idx":  (0, 2),
        "P_initial":      (0.01, 3.0),      # v21: золотая зона (trial 44: 0.0712)
        "pump_speed":     (0.01, 0.22),  # v43     # v42: 0.18->0.20 — зола растёт с плотностью 3.2e21
        "E_field":        (9e16, 2.34e17),  # v45R: масштаб по полю (0.9-1.8)e16 * B
        "gap_nm":          (1, 8),            # v21: золотая зона (trial 44: 1)
        "external_B":     (10.0, 13.0),     # v45R: реалистичное окно (SPARC 12.2 Тл, ITER-класс)
    }

    T_CAP = 1e9  # 1000 МК — выше это runaway

    # v15: Каскадные границы — раздельные для зажигания и горения
    # ОБЩИЕ параметры (одни и те же в обеих фазах): R, h, T_inner, T_outer, k_thermal,
    #   pl_radius, T_plasma_init, n_density, external_B, gap_nm
    # ФАЗО-СПЕЦИФИЧНЫЕ: E_field, P_initial, pump_speed

    IGNITION_BOUNDS = {
        # v21: золотая зона
        "R_mm":           (1200, 5500),    # v45R: синхрон с BOUNDS
        "h_mm":           (10.0, 18.0),
        "T_inner":        (360.0, 420.0),
        "T_outer":        (273.0, 300.0),   # v21: FIX — было (273,282)
        "k_thermal":      (20.0, 90.0),  # v26: расширено вниз
        "pl_radius_mm":    (500, 2860),      # v45R: кламп снят
        "T_plasma_init":  (1e8, 1.6e8),     # v36.2
        "n_density":      (1e20, 3.6e21),  # v45R: потолок динамический (0.8*n_G), низ 1e20
        "external_B":     (10.0, 13.0),     # v45R: синхрон с BOUNDS
        "gap_nm":          (1, 8),
        # Фаза-специфичные (зажигание — нужен мощный импульс)
        "E_field":        (9e16, 2.34e17),  # v45R: масштаб по полю (0.9-1.8)e16 * B
        "P_initial":      (0.01, 3.0),
        "pump_speed":     (0.01, 0.22),  # v43     # v42: 0.18->0.20 (синхрон с BOUNDS)
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
    ETA_STEAM = 0.42         # КПД парового нижнего цикла (legacy)
    ETA_BRAYTON_HE = 0.48    # v46B: КПД гелиевого Брайтона (верх комбинированного цикла)
    ETA_SCO2_BOTTOMING = 0.185  # v46B: КПД sCO2-низа на выхлопе Брайтона
    COOLANT = "he_brayton_sco2"  # v46B: "steam" | "he_brayton" | "he_brayton_sco2"
    RECIRC_FRAC = 0.04       # legacy-константа (дефолт, если B/v не переданы)

    @classmethod
    def _eta_low(cls):
        """v46B: единая формула нижнего каскада — станция и прунер считают одинаково."""
        if cls.COOLANT == "he_brayton_sco2":
            return cls.ETA_BRAYTON_HE + (1.0 - cls.ETA_BRAYTON_HE) * cls.ETA_SCO2_BOTTOMING
        return cls.ETA_BRAYTON_HE if cls.COOLANT == "he_brayton" else cls.ETA_STEAM
    # v36: рециркуляция = криогеника магнитов + ALIP-прокачка (налог МГД ~ (B·v)^2)
    RECIRC_CRYO_FRAC = 0.02
    RECIRC_PUMP_FRAC = 0.02
    RECIRC_REF_B = 28.71
    RECIRC_REF_PUMP = 0.0665
    BETA_MAX = 0.15          # v42: было 0.12 (v39) — сдвинут по экспорту v41:
                         # raw растёт с beta монотонно (0.17->2187 МВт), физической стены
                         # в модели НЕТ (MHD не моделируется). 0.15 — эмпирический потолок,
                         # предохранитель от нефизичного runaway, а не физический лимит
    MU0 = 1.25663706212e-6   # магнитная постоянная
    SAFETY_Q = 2.0           # v41: прокси q_95 для лимита Гринвальда (справочная метрика)
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

    # ── v44: ЭКОНОМИКА (режим "lcoe") ──
    WALL_LIMIT_W_M2 = 18.0e6      # v46D.1 (46.2 зонд): чешуя держит 18 МВт/м2 (было 12)
    WALL_SOFT_PENALTY = False     # v46D.1: мягкий штраф снят — зонд меряет сырой потолок
    WALL_DIAMOND_ONSET = 12.0e6   # v46D.1: надбавка чешуи только выше старого лимита
    C_DIAMOND = 0.0               # v46D.1: цена чешуи — КАЛИБРУЕТСЯ ПОСЛЕ ЗОНДА (брейк-ивен ~0.06)
    C_ALIGN = 0.15                # v46C (46.8): цена изолирующих покрытий alignment
    PRESCREEN_ENABLED = False     # v46C: гейт инертен (AutoLearn учится, не стреляет)
    COIL_H_M = 0.5                # эффективная толщина обмотки, м
    COIL_COST_PER_MJ = 80e3       # $/МДж запасённой энергии (якорь: ITER-класс)
    BOP_COST_PER_KW = 3900.0      # v46B: $/кВт эл. — пар 3500 + второй контур sCO2 (+400; брейк-ивен 4050)
    BLANKET_COST_PER_KW = 1500.0  # $/кВт тепловой (бланкет + первая стенка; power-режим)
    BLANKET_COST_PER_M2 = 16029966.545501301  # v46H: $/м2 (якорь: S9 run10, невязка 0.0)
    MATERIALS = {"steel": (12.0e6, 0.0), "tpg": (18.0e6, 0.05), "c_c_sic": (22.0e6, 0.10), "c_c_sic_adv": (25.0e6, 0.15)}  # v46L: adv по ТЗ Дипсика — потолок 25, наценка 15%
    CRYO_HEAT_BASE_W = 1.0e4      # статический теплоприход на катушки при 4.2 K, Вт
    CRYO_PENETRATION = 1e-5       # нейтронный нагрев катушек: доля P_fusion
    CRYO_CARNOT = 70.0            # Вт электричества на 1 Вт холода (класс 4-20 K)
    ELEC_PRICE_PER_KWH = 0.05     # внутренняя цена электричества, $/кВт·ч
    OPEX_FRAC = 0.05              # прочий OPEX: 5% CAPEX в год
    CRF = 0.0806                  # capital recovery factor (7%, 30 лет)
    CAPACITY_FACTOR = 0.85
    TRITIUM_PRICE_PER_KG = 30e3   # тритий — актив: избыток продаём

    @classmethod
    def greenwald_limit(cls, B_T, pl_radius_m):
        """v41: справочный лимит Гринвальда [м^-3].
        Прокси тока плазмы из равновесия: I_p = 2*pi*a*B/(mu0*q), q=SAFETY_Q.
        n_G = I_p[MA] * 1e20 / (pi * a^2). Метрика для user_attrs, не constraint."""
        if not B_T or not pl_radius_m or pl_radius_m <= 0:
            return float("nan")
        a = float(pl_radius_m)
        I_p_A = 2.0 * np.pi * a * float(B_T) / (cls.MU0 * cls.SAFETY_Q)
        return (I_p_A / 1e6) * 1e20 / (np.pi * a * a)

    def _economics(self, cfg, m):
        """v44: LCOE и нагрузка на первую стенку (вызывается при mode="lcoe").
        Якоря — порядки величин ITER/DEMO-класса, НЕ точная смета: цель — дать
        оптимизатору правильный градиент цены, а не рублёвую точность."""
        station = m.get("station") or {}
        P_fusion = float(m.get("avg_P_fusion_W", 0.0) or 0.0)
        P_elec = float(station.get("P_electric_W", 0.0) or 0.0)
        P_net = float(station.get("P_net_W", 0.0) or 0.0)
        m_T_net = float(station.get("m_T_net_kg_year", 0.0) or 0.0)
        tm = cfg.get("thermal_monitor") or {}
        R_m = float(tm.get("R", 0.0) or 0.0)
        if R_m <= 0:
            R_m = float(tm.get("R_mm", 0.0) or 0.0) / 1000.0
        B_T = float((cfg.get("plasma_monitor") or {}).get("external_B", 0.0) or 0.0)
        E_mag_J = (B_T ** 2) / (2.0 * self.MU0) * 4.0 * np.pi * R_m * R_m * self.COIL_H_M
        cost_mag = E_mag_J / 1e6 * self.COIL_COST_PER_MJ
        # v46Q: внутренний эмиттер — магнитная система дешевле вдвое
        if (cfg.get("economy_flags") or {}).get("emitter", False):
            cost_mag *= 0.5
        cost_bop = P_elec / 1000.0 * self.BOP_COST_PER_KW
        # v46H: бланкет ПО ПЛОЩАДИ (4*pi*R^2) с наценкой материала
        _mat = (cfg.get("vacuum_monitor") or {}).get("wall_material", "steel")
        _mat_mult = 1.0 + self.MATERIALS.get(_mat, self.MATERIALS["steel"])[1]
        cost_blanket = 4.0 * np.pi * R_m * R_m * self.BLANKET_COST_PER_M2 * _mat_mult
        # v46D.1 (46.2): алмазная чешуя — надбавка только в зоне выше старого лимита 12
        if self.C_DIAMOND > 0.0 and P_fusion > 0 and R_m > 0:
            if P_fusion / (4.0 * np.pi * R_m * R_m) > self.WALL_DIAMOND_ONSET:
                cost_blanket *= (1.0 + self.C_DIAMOND)
        # v46C (46.8): изолирующие покрытия прокачки вдоль силовых линий
        al_vm = float((cfg.get("vacuum_monitor") or {}).get("alignment", 0.0) or 0.0)
        cost_blanket *= (1.0 + self.C_ALIGN * al_vm * al_vm)
        # v46Q: сварной силовой корпус RAFM (постановление штаба 15.11) — бланкет дешевле на 30%
        if (cfg.get("economy_flags") or {}).get("welded_vessel", False):
            cost_blanket *= 0.7
        capex = cost_mag + cost_bop + cost_blanket
        cryo_heat_W = self.CRYO_HEAT_BASE_W * (B_T / 12.0) ** 2 + self.CRYO_PENETRATION * P_fusion
        opex_cryo = cryo_heat_W * self.CRYO_CARNOT / 1000.0 * 8760.0 * self.ELEC_PRICE_PER_KWH
        opex_T = -m_T_net * self.TRITIUM_PRICE_PER_KG  # тритий — актив
        annual = self.CRF * capex + opex_cryo + self.OPEX_FRAC * capex + opex_T
        area = 4.0 * np.pi * R_m * R_m
        wall_load = P_fusion / area if area > 0 else float("inf")
        if self.WALL_SOFT_PENALTY and wall_load > self.WALL_LIMIT_W_M2:
            wall_mult = (wall_load / self.WALL_LIMIT_W_M2) ** 2
        else:
            wall_mult = 1.0  # v46D.1: штраф снят (зонд)
        E_year_MWh = P_net / 1e6 * 8760.0 * self.CAPACITY_FACTOR
        lcoe_raw = annual / E_year_MWh if E_year_MWh > 0 else 1e10
        return {"lcoe": lcoe_raw * wall_mult, "lcoe_raw": lcoe_raw,
                "wall_load_W_m2": wall_load, "wall_penalty_mult": wall_mult,
                "cost_mag": cost_mag, "cost_bop": cost_bop,
                "cost_blanket": cost_blanket, "cost_cryo_W": cryo_heat_W}

    def _wall_load(self, cfg, m):
        """v45R: нагрузка на первую стенку [Вт/м2] = P_fusion / (4*pi*R^2)."""
        tm = cfg.get("thermal_monitor") or {}
        R_m = float(tm.get("R", 0.0) or 0.0)
        if R_m <= 0:
            R_m = float(tm.get("R_mm", 0.0) or 0.0) / 1000.0
        P_fusion = float(m.get("avg_P_fusion_W", 0.0) or 0.0)
        area = 4.0 * np.pi * R_m * R_m
        return P_fusion / area if area > 0 else float("inf")

    @classmethod
    def _station_balance(cls, P_fusion_W, P_sync_net_W, B_T=None, pump_speed=None, alignment=0.0):  # v46A
        # v32: энергобаланс станции. Зовётся из _run_quick с усреднёнными мощностями.
        rate = P_fusion_W / cls.E_DT if cls.E_DT > 0 else 0.0   # реакций/с
        m_T_burn = rate * cls.M_TRITIUM * cls.SECONDS_PER_YEAR   # кг трития сжигается/год
        m_T_net = m_T_burn * (cls.T_BR - 1.0) / cls.T_BR         # избыток на продажу/запас
        # v46B: нижний каскад — единая формула (пар | Брайтон | Брайтон+sCO2)
        eta_low = cls._eta_low()
        eta_cascade = cls.ETA_MHD + (1.0 - cls.ETA_MHD) * eta_low
        P_sync_harvest = cls.SYNC_HARVEST_FRAC * max(P_sync_net_W, 0.0) * cls.SYNC_CONV_EFF
        P_electric = P_fusion_W * eta_cascade + P_sync_harvest
        # v36: честная рециркуляция — криогеника (константа) + ALIP-прокачка ~ (B·v)^2
        recirc = cls.RECIRC_FRAC
        recirc_raw = recirc  # v41: raw-налог до клипа — телеметрия насыщения
        if B_T is not None and pump_speed is not None:
            recirc_raw = cls.RECIRC_CRYO_FRAC + (1.0 - float(alignment)) * cls.RECIRC_PUMP_FRAC * (
                (float(B_T) / cls.RECIRC_REF_B) ** 2) * (
                (float(pump_speed) / cls.RECIRC_REF_PUMP) ** 2)
            recirc = float(recirc_raw)  # v43: клип 0.12 снят — честный налог B^2*pump^2
        P_net = P_electric * (1.0 - recirc)
        return {
            "eta_cascade": eta_cascade,
            "recirc_frac": recirc,
            "recirc_raw": recirc_raw,  # v41
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
                self._prescreen_last_reason = f"T_pred={T_pred:.4e} > T_CAP={self.T_CAP:.1e} (runaway)"
                return 'reject'
            # 2. T < 1e6 — слишком холодная, реакция не пойдёт
            if T_pred < 1e6:
                self._prescreen_stats["rejected"] += 1
                self._prescreen_last_reason = f"T_pred={T_pred:.4e} < 1e6 (холодная плазма)"
                return 'reject'
            # 3. pair_rate < 1 — нет рождения пар
            if pair_rate_pred < 1.0:
                self._prescreen_stats["rejected"] += 1
                self._prescreen_last_reason = f"pair_rate_pred={pair_rate_pred:.4e} < 1 (нет рождения пар)"
                return 'reject'

            self._prescreen_stats["passed"] += 1
            return None
        except Exception:
            self._prescreen_stats["passed"] += 1
            return None

    def _prescreen_checked(self, trial, cfg):
        """v46.7: прозрачность гейта — reject пишется в телеметрию,
        сиды обходят фильтр (эталонные точки защищены от суррогата)."""
        if not self.PRESCREEN_ENABLED:
            return None  # v46C: гейт инертен по умолчанию (AutoLearn продолжает учиться)
        # v46C FIX: двойная страховка сида — метка enqueue + ключ из trial.params
        # (в v46B ключ строился из плоских cfg.get — всегда (0,0,0), обход был мёртв)
        try:
            if trial.user_attrs.get("seed"):
                try:
                    trial.set_user_attr("prescreen_reject", "seed_bypass")
                except Exception:
                    pass
                return None
        except Exception:
            pass
        _tp = getattr(trial, "params", {}) or {}
        _efq = cfg.get("economy_flags") or {}
        key = (round(float(_tp.get("n_density", cfg.get("n_density", 0) or 0) or 0), 3),
               round(float(_tp.get("R_mm", cfg.get("R_mm", 0) or 0) or 0), 1),
               round(float(_tp.get("pl_radius_mm", cfg.get("pl_radius_mm", 0) or 0) or 0), 1),
               round(float(_tp.get("pump_speed", cfg.get("pump_speed", 0) or 0) or 0), 6),
               round(float(_tp.get("alignment", cfg.get("alignment", 0) or 0) or 0), 4),
               str(_tp.get("wall_material", cfg.get("wall_material", "steel") or "steel")),
               # v46Q: флаги в ключе прескрина — синхронно с _pkey
               bool(_tp.get("welded_vessel", _efq.get("welded_vessel", False))),
               bool(_tp.get("emitter", _efq.get("emitter", False))))
        if key in getattr(self, "_seed_pkeys", set()):
            try:
                trial.set_user_attr("prescreen_reject", "seed_bypass")
            except Exception:
                pass
            return None
        reason = self._prescreen(cfg)
        if reason:
            try:
                trial.set_user_attr("prescreen_reject",
                                    getattr(self, "_prescreen_last_reason", "reject"))
            except Exception:
                pass
        return reason

    def _active_bounds(self):
        """v44: границы зависят от режима. В lcoe окно R расширено — размер
        становится ручкой экономики (стенку разбавляют радиусом)."""
        # v45R: override снят — реалистичное окно живёт в самих BOUNDS
        return dict(self.BOUNDS)

    def _suggest_alignment(self, trial):
        """v46A: выравнивание прокачки. FixedTrial без ключа -> 0.0 (старая физика,
        обратная совместимость _refine_top/_staged_validate); живой Trial -> TPE."""
        if "alignment" in getattr(trial, "params", {}):
            return float(trial.params["alignment"])
        if isinstance(trial, optuna.trial.FixedTrial):
            return 0.0
        return trial.suggest_float("alignment", 0.0, 1.0)

    def _suggest_material(self, trial):
        """v46H: материал первой стенки. FixedTrial без ключа -> steel
        (обратная совместимость сидов/валидации); живой Trial -> TPE."""
        if "wall_material" in getattr(trial, "params", {}):
            return str(trial.params["wall_material"])
        if isinstance(trial, optuna.trial.FixedTrial):
            return "steel"
        return trial.suggest_categorical("wall_material", ["steel", "tpg", "c_c_sic", "c_c_sic_adv"])  # v46L: +adv

    def _suggest_flag(self, trial, name):
        """v46Q: экономические флаги (welded_vessel/emitter). FixedTrial без ключа
        -> False (обратная совместимость сидов/валидации); живой Trial -> TPE."""
        if name in getattr(trial, "params", {}):
            return bool(trial.params[name])
        if isinstance(trial, optuna.trial.FixedTrial):
            return False
        return bool(trial.suggest_categorical(name, [False, True]))

    def _build_trial_config(self, trial):
        """Конфиг из trial. Constraint: pl_radius < R_mm * 0.5."""
        cfg = dict(self.base_config)

        R_mm = trial.suggest_float("R_mm", *self._active_bounds()["R_mm"])  # v44
        h_mm = trial.suggest_float("h_mm", *self.BOUNDS["h_mm"])
        T_inner = trial.suggest_float("T_inner", *self.BOUNDS["T_inner"])
        T_outer = trial.suggest_float("T_outer", *self.BOUNDS["T_outer"])
        k_thermal = trial.suggest_float("k_thermal", *self.BOUNDS["k_thermal"])

        # v21: Constraint: плазма внутри сферы (золотая зона + safety)
        if self.mode == "lcoe":
            # v46Q: СТАТИЧНАЯ СИГНАТУРА TPE — pl/n едины для всех триалов, сиды
            # и разведка в одной группе joint-модели (лечение ямы разведки).
            # Ограничения (золотая зона, Гринвальд) — гейтами в objective_ignition.
            pl_radius_mm = trial.suggest_float("pl_radius_mm", 500.0, 2600.0)
            T_plasma = trial.suggest_float("T_plasma_init", *self.BOUNDS["T_plasma_init"], log=True)
            external_B = trial.suggest_float("external_B", *self.BOUNDS["external_B"])  # v46Q: ось до 15 Тл
            n_density = trial.suggest_float("n_density", 1e20, 3.6e21, log=True)
            welded_vessel = self._suggest_flag(trial, "welded_vessel")  # v46Q
            emitter = self._suggest_flag(trial, "emitter")              # v46Q
            cfg["economy_flags"] = {"welded_vessel": welded_vessel, "emitter": emitter}
        else:
            pl_radius_lo = max(R_mm * 0.42, 500)
            pl_radius_hi = R_mm * 0.52  # v45R: кламп 620 снят — реалистичная геометрия
            if pl_radius_hi <= pl_radius_lo:
                pl_radius_hi = pl_radius_lo + 10.0  # safety: never flip
            pl_radius_mm = trial.suggest_float("pl_radius_mm",
                                               pl_radius_lo,
                                               pl_radius_hi)
            T_plasma = trial.suggest_float("T_plasma_init", *self.BOUNDS["T_plasma_init"], log=True)
            external_B = trial.suggest_float("external_B", *self.BOUNDS["external_B"])  # v45R.2: ДО Гринвальда (FIX UnboundLocalError)
            # v45R: external_B предлагается ДО плотности — потолок Гринвальда зависит от B и a
            # v45R: динамический потолок плотности: n <= 0.8 * n_G(B, a), a — малый радиус плазмы
            _a_m = pl_radius_mm / 1000.0
            _n_G = self.greenwald_limit(external_B, _a_m)
            _n_hi = min(self.BOUNDS["n_density"][1], 0.8 * _n_G) if np.isfinite(_n_G) else self.BOUNDS["n_density"][1]
            _n_hi = max(_n_hi, self.BOUNDS["n_density"][0] * 1.2)  # safety: не вырождать окно
            n_density = trial.suggest_float("n_density", self.BOUNDS["n_density"][0], _n_hi, log=True)

        # v9.1: D-T зафиксирован
        fuel_idx = self.FIXED_FUEL_IDX
        fuel_type = self.FUEL_TYPES[fuel_idx]

        P_initial = trial.suggest_float("P_initial", *self.BOUNDS["P_initial"], log=True)
        pump_speed = trial.suggest_float("pump_speed", *self.BOUNDS["pump_speed"], log=True)
        E_field = trial.suggest_float("E_field", *self.BOUNDS["E_field"], log=True)
        gap_nm = trial.suggest_int("gap_nm", *self.BOUNDS["gap_nm"])
        alignment = self._suggest_alignment(trial)  # v46A
        wall_material = self._suggest_material(trial)  # v46H

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
            "alignment": alignment,  # v46A
            "wall_material": wall_material,  # v46H
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
            bounds = self._active_bounds() if self.mode == "lcoe" else self.IGNITION_BOUNDS  # v44
            # Все параметры свободны
            R_mm = trial.suggest_float("R_mm", *bounds["R_mm"])
            h_mm = trial.suggest_float("h_mm", *bounds["h_mm"])
            T_inner = trial.suggest_float("T_inner", *bounds["T_inner"])
            T_outer = trial.suggest_float("T_outer", *bounds["T_outer"])
            k_thermal = trial.suggest_float("k_thermal", *bounds["k_thermal"])
            pl_radius_lo = max(R_mm * 0.42, 500)
            pl_radius_hi = R_mm * 0.52  # v45R: кламп 620 снят — реалистичная геометрия
            if pl_radius_hi <= pl_radius_lo:
                pl_radius_hi = pl_radius_lo + 10.0  # safety: never flip
            pl_radius_mm = trial.suggest_float("pl_radius_mm",
                                                pl_radius_lo,
                                                pl_radius_hi)
            T_plasma = trial.suggest_float("T_plasma_init", *bounds["T_plasma_init"], log=True)
            # v45R: external_B до плотности — динамический потолок Гринвальда
            external_B = trial.suggest_float("external_B", *bounds["external_B"])
            _a_m = pl_radius_mm / 1000.0
            _n_G = self.greenwald_limit(external_B, _a_m)
            _n_hi = min(bounds["n_density"][1], 0.8 * _n_G) if np.isfinite(_n_G) else bounds["n_density"][1]
            _n_hi = max(_n_hi, bounds["n_density"][0] * 1.2)
            n_density = trial.suggest_float("n_density", bounds["n_density"][0], _n_hi, log=True)
            gap_nm = trial.suggest_int("gap_nm", *bounds["gap_nm"])
            E_field = trial.suggest_float("E_field", *bounds["E_field"], log=True)
            P_initial = trial.suggest_float("P_initial", *bounds["P_initial"], log=True)
            pump_speed = trial.suggest_float("pump_speed", *bounds["pump_speed"], log=True)
            alignment = self._suggest_alignment(trial)  # v46A
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
            alignment = float(shared_params.get("alignment", 0.0))  # v46A: геометрия общая

            # Фаза-специфичные — горение (низкий E_field, низкое P)
            w = self.BURN_BOUNDS_WIDTH
            e_lo, e_hi = 1e16, 5e17   # НИЗКИЙ E_field для горения
            p_lo, p_hi = 1e-4, 1.0    # НИЗКОЕ давление
            ps_lo, ps_hi = 1e-3, 0.22  # v43  # v42: 0.18->0.20 (синхрон с BOUNDS)

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
            "alignment": alignment,  # v46A
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
                    align_rep = float(cfg.get("vacuum_monitor", {}).get("alignment", 0.0))  # v46A
                    f_He_rep = min(self.F_HE_MAX, self.C_ASH * R_burn_rep /
                                   max(n_density * V_plasma * pump_rep, 1e-30))
                    P_net_rep = recent_p * (1.0 - f_He_rep) ** 2
                    P_elec_rep = P_net_rep * (self.ETA_MHD +
                                              (1.0 - self.ETA_MHD) * self._eta_low())  # v46B FIX: был жёсткий ETA_STEAM
                    recirc_rep = self.RECIRC_CRYO_FRAC + (1.0 - align_rep) * self.RECIRC_PUMP_FRAC * \
                        (external_B / self.RECIRC_REF_B) ** 2 * \
                        (pump_rep / self.RECIRC_REF_PUMP) ** 2
                    recirc_rep = float(recirc_rep)  # v43: клип снят — честный налог
                    # v46G: охрана зажигания — P_elec_rep < 1e6 Вт => конфигурация
                    # не зажглась (trial 34 run9: value -0.0 при last_intermediate
                    # -1.026e-18). Сентинел 1e10 вместо шума: value-колонка читается,
                    # Best-парсер не травится, прунер добивает мусор сразу.
                    if P_elec_rep < 1e6:
                        trial.report(1e10, step_i)
                        trial.set_user_attr("last_intermediate", 1e10)
                        trial.set_user_attr("no_ignition", True)
                    else:
                        p_net_step = -P_elec_rep * (1.0 - recirc_rep)  # текущее значение шага
                        trial.report(p_net_step, step_i)
                        trial.set_user_attr("last_intermediate", p_net_step)  # v40: телеметрия Дипа

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
                                                 pump_speed=cfg["vacuum_monitor"].get("pump_speed", self.RECIRC_REF_PUMP),
                                                 alignment=cfg["vacuum_monitor"].get("alignment", 0.0)),  # v46A
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
                    q_step = -recent_q  # текущее значение шага
                    trial.report(q_step, step_i)
                    trial.set_user_attr("last_intermediate", q_step)  # v40: телеметрия Дипа
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
                if self.mode == "lcoe" and best_val not in (None, 0, 1e10):
                    progress = (trial.number + 1) / total_trials
                    st.write(f"Trial {trial.number + 1}/{total_trials} | Best LCOE={best_val:.1f} $/МВт·ч")
                elif self.mode != "pareto" and best_val != 0:
                    progress = (trial.number + 1) / total_trials
                    st.write(f"Trial {trial.number + 1}/{total_trials} | Best P={-best_val/1e6:.2f} MW")
                else:
                    st.write(f"Trial {trial.number + 1}/{total_trials}")
            except Exception:
                pass
        return callback

    def _save_training_dataset(self):
        """v44: автосохранение датасета после прогона. COMPLETE и PRUNED-триалы
        мерджатся в models/training_dataset.csv по ключу (study, number) —
        экспорт и обучающая выборка больше никогда не расходятся."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(os.path.join(script_dir, "models"), exist_ok=True)
            path = os.path.join(script_dir, "models", "training_dataset.csv")
            import csv as _csv
            rows = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for r in _csv.DictReader(f):
                        rows[(r.get("study"), r.get("number"))] = r
            for t in self.study.get_trials(deepcopy=False):
                if t.state not in (optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.PRUNED):
                    continue
                row = {"study": self.study.study_name, "number": str(t.number),
                       "state": t.state.name, "physics": "v46Q", "mode": self.mode}
                if t.value is not None:
                    row["value"] = t.value
                row.update({f"param_{k}": v for k, v in (t.params or {}).items()})
                row.update({f"attr_{k}": v for k, v in (t.user_attrs or {}).items()})
                rows[(row["study"], row["number"])] = row
            fieldnames = sorted({k for r in rows.values() for k in r})
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows.values():
                    w.writerow(r)
            print(f"[Dataset] training_dataset.csv: {len(rows)} записей (physics=v46Q)")
        except Exception as e:
            print(f"[Dataset] save error: {e}")

    def optimize(self, mode: str = None, n_trials: int = 80,   # v39: 50->80; v45R: None -> lcoe
                 use_surrogate: bool = False,
                 surrogate_trials: int = 500,
                 refine_top: int = 10):
        mode = mode or "lcoe"  # v45R: LCOE по умолчанию
        self.mode = mode
        # v46O: ось B — расширение external_B ТОЛЬКО в lcoe
        # (ignition/gain/pareto/каскад живут на прежнем окне 10–13 Тл).
        # v46Q: до 15 Тл — S32/S33 диагностируют показатель d в P ~ B^d.
        if mode == "lcoe":
            self.BOUNDS = dict(self.BOUNDS, external_B=(10.0, 15.0))
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
        # v36: судья не трогает триал на разогреве; v42: warmup = 65% прогона
        # v46G: SeedAwarePruner — медиана ТОЛЬКО по не-сидовым COMPLETE: сиды —
        #       эталоны, а не планка (run9: 40/50 PRUNED, разведка без финишеров)
        base_steps = int(self.base_config.get("simulation", {}).get("n_steps", 2500000))
        pruner = SeedAwarePruner(
            n_startup_trials=10,                    # первые 10 не прерывать
            n_warmup_steps=int(0.65 * base_steps),  # v42: 50%->65% — гигиена прунера, выборка плотнее
            interval_steps=max(1, base_steps // 20),
            n_min_trials=3,                         # не судить по 1–2 точкам
        )
        # v46O (перенос из v46N): NopPruner в lcoe — пруннутые с отрицательным
        # last_intermediate (-9.6e7...-4.4e8) попадали в below-множество TPE
        # как "чемпионы" (run16: 71 из 83). Run17 (NopPruner): PRUNED=0, все
        # COMPLETE. Сиды не прунятся и так; ignition/gain/pareto — SeedAware.
        if mode == "lcoe":
            pruner = optuna.pruners.NopPruner()

        study_name = f"gak_opt_{mode}_v46Q"  # v46Q: флаги экономики + B=15 + статичная сигнатура + гейты
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
        # v39: чемпион v37.2 (trial 37: 597.24 МВт net) — старт из новой долины
        SEED_PARAMS_6 = {
            "R_mm": 1205.8618926411255, "h_mm": 12.302061995307,
            "T_inner": 379.3506434254803, "T_outer": 293.89135499073126,
            "k_thermal": 39.2815985066004, "pl_radius_mm": 616.1902130761636,
            "T_plasma_init": 123126866.94048074, "n_density": 2.0734839105972158e21,
            "P_initial": 0.06788464581116989, "pump_speed": 0.1440353424694812,
            "E_field": 5.5629162289043616e17, "gap_nm": 2,
            "external_B": 28.48095578305144,
        }
        # v37: рекорд 1527 МВт — разведка (2.2e21, 615) на базе SEED_PARAMS_4
        SEED_PARAMS_5 = dict(SEED_PARAMS_4)
        SEED_PARAMS_5["n_density"] = 3.0e21  # v41: разведка на новом потолке (2.6 -> 3.0)
        SEED_PARAMS_5["pl_radius_mm"] = 615.0
        def _probe_n(base, n_new, pl_new=None):
            p = dict(base); p["n_density"] = n_new
            if pl_new is not None: p["pl_radius_mm"] = pl_new
            return p
        # v41: чемпион v40 — trial 79 (1388.54 МВт net), первым в очереди
        SEED_PARAMS_7 = {
            "R_mm":           1196.4097311434175,
            "h_mm":           10.89644647097812,
            "T_inner":        388.4089122831374,
            "T_outer":        286.6778738247287,
            "k_thermal":      26.63317557121712,
            "pl_radius_mm":   616.1558096833119,
            "T_plasma_init":  130575281.82843617,
            "n_density":      2.336991641190688e+21,
            "P_initial":      0.1275864939496372,
            "pump_speed":     0.1700226148976479,
            "E_field":        5.0263975275730464e+17,
            "gap_nm":         1,
            "external_B":     30.379590947908213,
        }
        # v42: сиды — теневой чемпион v41 (trial 73: raw 2060.6 МВт, beta 0.1268)
        SEED_PARAMS_8 = {
            "R_mm":           1182.5697351317338,
            "h_mm":           12.818797906477275,
            "T_inner":        416.0718688935157,
            "T_outer":        299.7381203205847,
            "k_thermal":      26.936828269043488,
            "pl_radius_mm":   607.9276441094855,
            "T_plasma_init":  122644349.40333551,
            "n_density":      2.998470399013456e+21,
            "P_initial":      0.14631206124824597,
            "pump_speed":     0.15804003080267592,
            "E_field":        4.554029583448481e+17,
            "gap_nm":         3,
            "external_B":     33.76542305716652,
        }
        # v42: официальный чемпион v41 (trial 70: 1838.60 МВт net, beta 0.1140)
        SEED_PARAMS_9 = {
            "R_mm":           1189.5816795080914,
            "h_mm":           13.211449882412893,
            "T_inner":        419.12438355867016,
            "T_outer":        298.76996571985524,
            "k_thermal":      25.362521544543014,
            "pl_radius_mm":   616.2750488333286,
            "T_plasma_init":  122132463.5063297,
            "n_density":      2.852246398685251e+21,
            "P_initial":      0.08536927348431113,
            "pump_speed":     0.12219790380357026,
            "E_field":        4.327347925931941e+17,
            "gap_nm":         3,
            "external_B":     34.22039648428221,
        }
        # v43: чемпион прогона 01.11 (trial 31: 2419.50 МВт net при честном налоге)
        SEED_PARAMS_CHAMP43 = {
            "R_mm":           1192.4697953863817,
            "h_mm":           13.07100550538681,
            "T_inner":        413.54675585272474,
            "T_outer":        297.4151485918359,
            "k_thermal":      36.83919923086654,
            "pl_radius_mm":   609.7902584076783,
            "T_plasma_init":  113513494.03030033,
            "n_density":      3.4287805241632217e+21,
            "P_initial":      0.05474568408859994,
            "pump_speed":     0.12422406344336229,
            "E_field":        5.480526598115344e+17,
            "gap_nm":         3,
            "external_B":     34.7907691255818,
        }
        # v45R: сиды S1–S5 — точки реалистичного окна (B 10–13 Тл, R 1.2–5.5 м).
        # n каждого = 0.78*n_G(B, a) — под динамическим потолком Гринвальда.
        # S5 — скаут жёсткого лимита стенки (ожидаем reject 1e10).
        _S_BASE = {
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "k_thermal": 36.84, "T_plasma_init": 1.135e8,
            "P_initial": 0.0547, "pump_speed": 0.18, "gap_nm": 1,
            "alignment": 0.0,  # v46A: смоук — эталонная физика
        }
        def _nG(B_T, a_m):
            I_p = 2.0 * np.pi * a_m * B_T / (self.MU0 * self.SAFETY_Q)
            return (I_p / 1e6) * 1e20 / (np.pi * a_m * a_m)
        def _seed_v45R(B_T, R_mm_, wall_MW):
            # v45R.2: сиды под стенку — закон смоука wall = 2.86e-35 * n^2 * R (R в м)
            a_m = R_mm_ * 0.52 / 1000.0
            R_m = R_mm_ / 1000.0
            n_wall = (wall_MW * 1e6 / (2.86e-35 * R_m)) ** 0.5
            return {**_S_BASE, "R_mm": R_mm_, "pl_radius_mm": R_mm_ * 0.52,
                    "external_B": B_T, "E_field": 1.2e16 * B_T,
                    "n_density": min(0.78 * _nG(B_T, a_m), n_wall)}
        # v46K ЭТАЛОНЫ (пер-м² экономика; сверено по телеметрии run10/run13):
        #   S1(steel) = 115.1209736869   S2(steel) = 113.4016484507
        #   S3(steel) = 116.1618391886   S4(steel) = 124.7048229193
        #   S5(tpg)   = 113.4624627723   S6(steel) = 119.8067081467
        #   S7(steel) = 118.8687251815   S8(tpg)   = 104.5177472990
        #   S9(steel) = 114.8494689433409  S10(steel) = 114.7058633245
        #   S11(steel) = 113.7473154999 — лидер стального кластера
        #   S12(c_c_sic) = 100.6447775801 — ПРОБОЙ СОТНИ (run13, бит в бит)
        #   S13(c_c_sic) = 99.5891013901 — ЧЕМПИОН, сотня пробита (run14, бит в бит)'
        #   S14(c_c_sic) = 106.1617 честный (raw 97.1828) — ось плотности Дипсика
        #   S15/S16/S17(c_c_sic) = 103.7786 / 112.5979 / 122.2686 честные
        #     (сырые 98.5391 / 97.4948 / 96.4563 покупались багом беты)
        #   S18-S21(c_c_sic) = R=2500/2400/2300/2200 при n=5.309272075653e20
        #     (beta=0.149 на R=2500) — честная цепочка по потолку беты
        queued_cfgs = [_seed_v45R(12.0, 4300.0, 11.5),   # S1: под стенку (~11.5 МВт/м2)
                       _seed_v45R(11.0, 4000.0, 11.5),   # S2
                       _seed_v45R(12.5, 4600.0, 11.5),   # S3
                       _seed_v45R(13.0, 5000.0, 10.0),   # S4: дорого по магнитам
                       {**_seed_v45R(12.0, 3000.0, 12.5), "wall_material": "tpg"},  # S5: скаут градиента; v46H: tpg
                       {  # v46B: S6 — сид-инъекция чемпиона v46A (trial 42 run3,
                          # LCOE 117.63346736636007); параметры бит в бит из
                          # trials_v46A_run3_0611.csv — детерминизм проверяем бесплатно
                           "R_mm": 4663.101166321131,
                           "h_mm": 13.820513241074568,
                           "T_inner": 408.4747743556464,
                           "T_outer": 298.997129811904,
                           "k_thermal": 40.59063349797366,
                           "pl_radius_mm": 2193.9017773960154,
                           "T_plasma_init": 111794542.46325402,
                           "n_density": 3.247011140181622e+20,
                           "P_initial": 0.06061075445994368,
                           "pump_speed": 0.042368363394085964,
                           "E_field": 1.350090861403684e+17,
                           "gap_nm": 5,
                           "external_B": 11.266935972986923,
                           "alignment": 0.39134355828674217},
                       {  # v46D.1: S7 — чемпион S6 с alignment=0.0 (контрфакт 46.8:
                          # TPE не дошёл до близнеца без покрытий). Эталон
                          # 115.00129928973597 — проверяем седьмой тест детерминизма
                           "R_mm": 4663.101166321131,
                           "h_mm": 13.820513241074568,
                           "T_inner": 408.4747743556464,
                           "T_outer": 298.997129811904,
                           "k_thermal": 40.59063349797366,
                           "pl_radius_mm": 2193.9017773960154,
                           "T_plasma_init": 111794542.46325402,
                           "n_density": 3.247011140181622e+20,
                           "P_initial": 0.06061075445994368,
                           "pump_speed": 0.042368363394085964,
                           "E_field": 1.350090861403684e+17,
                           "gap_nm": 5,
                           "external_B": 11.266935972986923,
                           "alignment": 0.0},
                       {**_seed_v45R(13.0, 3000.0, 16.5), "wall_material": "tpg"},  # v46E: S8 — скаут алмазной зоны; v46H: tpg
                       {  # v46F: S9 = trial 31 run8 (115.6080) с alignment=0.0 —
                          # TPE держал 114.8494689433 и отдал 0.7585 за alignment-налог;
                          # параметры бит в бит из trials_v46E_run8_0811.csv
                            "R_mm": 4443.387577294695,
                            "h_mm": 14.675611588292389,
                            "T_inner": 419.4893243763244,
                            "T_outer": 297.7580834037615,
                            "k_thermal": 45.94708170333445,
                            "pl_radius_mm": 2178.4462037748535,
                            "T_plasma_init": 110086487.93450329,
                            "n_density": 3.2832237206200235e+20,
                            "P_initial": 0.05852128706978756,
                            "pump_speed": 0.041928387027360695,
                            "E_field": 1.2159635818550267e+17,
                            "gap_nm": 5,
                            "external_B": 11.641362497595981,
                            "alignment": 0.0},
                       {**_seed_v45R(12.0, 4300.0, 11.5),  # v46F: S10 = S1 с al=0.26 —
                        "alignment": 0.26},                # закрытие 46.8 на высоком насосе
        ]
        # v46G: СИД S11 = S9 с pump_speed=0.06 — сид-проба рекомендации Дипсика №3
        # ("дать больше откачки"). Физика от насоса меняется => эталона нет,
        # репорт-точка. Ожидание по run8 (trial 33, pump=0.066): чуть хуже S9 (~114.9).
        queued_cfgs.append({**queued_cfgs[8], "pump_speed": 0.06})
        # v46K: СИД S12 = S8, сжатый до R=2600 — скаут зоны C/C-SiC (ТЗ Дипсика).
        # n ∝ 1/R, pl ∝ R; стенка ~17.6 МВт/м2 — зона доступна только композиту.
        # Параметры — литералы бит в бит из run13 (v46J): эталон 100.6447775801,
        # пробой сотни подтверждён run13.
        queued_cfgs.append({
            "R_mm": 2600.0, "pl_radius_mm": 1352.0, "external_B": 13.0,
            "n_density": 5.05995011135e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46K: СИД S13 = S12, сжатый до R=2500 — репорт-точка нижнего края
        # композитной зоны. n = 5.262348115804e20 (продолжение ∝1/R),
        # pl=1300, стенка ~18.35 МВт/м2 (масштаб от измеренных 17.64 у S12):
        # выше лимита TPG (18), под потолком C/C-SiC (22). Эталон 99.5891013901 — run14.
        queued_cfgs.append({
            "R_mm": 2500.0, "pl_radius_mm": 1300.0, "external_B": 13.0,
            "n_density": 5.262348115804e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46L: СИД S14 — ось плотности (ТЗ Дипсика): R=2500, n=5.5e20 (доля
        # Гринвальда 0.691), проверка растёт ли P_net при подкачке плотности.
        # run15 честно: raw 97.1828, честный 106.1617 (бета 0.1568 > 0.15):
        # выигрыш дал тритиевый кредит opex_T, не мощность. Репорт-точка.
        queued_cfgs.append({
            "R_mm": 2500.0, "pl_radius_mm": 1300.0, "external_B": 13.0,
            "n_density": 5.5e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46L: СИДЫ S15/S16/S17 — тренд сжатия к нижнему краю композитной
        # зоны: R=2400/2300/2200, n ∝ 1/R от S13 (доля 0.6613 сохраняется),
        # pl = 0.52*R. Стенки ~19.0/19.9/20.8 — под потолком c_c_sic (22).
        # Run15 честно: S15=103.7786, S16=112.5979, S17=122.2686 — сырое сжатие
        # покупалось багом беты. Репорт-точки честного тренда.
        queued_cfgs.append({
            "R_mm": 2400.0, "pl_radius_mm": 1248.0, "external_B": 13.0,
            "n_density": 5.481612620629e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2300.0, "pl_radius_mm": 1196.0, "external_B": 13.0,
            "n_density": 5.719943604135e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2200.0, "pl_radius_mm": 1144.0, "external_B": 13.0,
            "n_density": 5.979941040686e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46M: СИДЫ S18-S21 — цепочка ПО ПОТОЛКУ БЕТЫ: R=2500/2400/2300/2200
        # при постоянной n=5.309272075653e+20 (beta=0.149 на R=2500, intercept
        # падает с R). Ошибка v46L исправлена: сжатие при постоянной плотности,
        # pl=0.52*R. Стенки 18.6/17.9/17.2/16.5 — под потолком c_c_sic (22).
        # Репорт-точки честного тренда (ожидания по сырому тренду ~99.1 и ниже,
        # но честное улучшение скромнее — P_net падает с R при постоянной n).
        queued_cfgs.append({
            "R_mm": 2500.0, "pl_radius_mm": 1300.0, "external_B": 13.0,
            "n_density": 5.309272075653e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2400.0, "pl_radius_mm": 1248.0, "external_B": 13.0,
            "n_density": 5.309272075653e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2300.0, "pl_radius_mm": 1196.0, "external_B": 13.0,
            "n_density": 5.309272075653e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2200.0, "pl_radius_mm": 1144.0, "external_B": 13.0,
            "n_density": 5.309272075653e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46O: СИДЫ S22–S26 — хребет кромки беты при B=13. Литералы бит в бит
        # из run17 (v46N-прогон): S25=96.6557 чемпион, S26 (Дипсик)=101.008.
        queued_cfgs.append({
            "R_mm": 2600.0, "pl_radius_mm": 1352.0, "external_B": 13.0,
            "n_density": 5.255653e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2700.0, "pl_radius_mm": 1404.0, "external_B": 13.0,
            "n_density": 5.201668e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2800.0, "pl_radius_mm": 1456.0, "external_B": 13.0,
            "n_density": 5.147682e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 3000.0, "pl_radius_mm": 1560.0, "external_B": 13.0,
            "n_density": 5.039711e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 2800.0, "pl_radius_mm": 1456.0, "external_B": 13.0,
            "n_density": 5.309272075653e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46O: СИД S27 — ось B: R=2500, B=14, n=6.10e20, c_c_sic_adv
        # (реванш adv: в run16 умер в яме зажигания, не от наценки). Стенка
        # ~24.6 при потолке 25, доля Гринвальда 0.711. Ожидание 93–96;
        # главный вопрос — сдвиг T_eq при B=14 (синхротрон ~ B^2).
        queued_cfgs.append({
            "R_mm": 2500.0, "pl_radius_mm": 1300.0, "external_B": 14.0,
            "n_density": 6.10e+20, "E_field": 1.68e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic_adv"})
        # v46O: СИДЫ S28/S29 — плато хребта B=13: плотность прижата кэпом
        # 0.8*n_G (доля ровно 0.8). Диагностика точки перегиба R~3300.
        queued_cfgs.append({
            "R_mm": 3400.0, "pl_radius_mm": 1768.0, "external_B": 13.0,
            "n_density": 4.681e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        queued_cfgs.append({
            "R_mm": 3600.0, "pl_radius_mm": 1872.0, "external_B": 13.0,
            "n_density": 4.421e+20, "E_field": 1.56e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic"})
        # v46Q: СИДЫ S30–S35 — эталоны run19 + ось B=15 + контроли флагов
        queued_cfgs.append({
            "R_mm": 2600.0, "pl_radius_mm": 1352.0, "external_B": 14.0,
            "n_density": 5.95e+20, "E_field": 1.68e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic_adv"})                                   # S30 = run19: 93.9327
        queued_cfgs.append({
            "R_mm": 2700.0, "pl_radius_mm": 1404.0, "external_B": 14.0,
            "n_density": 5.86e+20, "E_field": 1.68e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic_adv"})                                   # S31 = run19: 93.7031
        queued_cfgs.append({
            "R_mm": 2700.0, "pl_radius_mm": 1404.0, "external_B": 15.0,
            "n_density": 5.86e+20, "E_field": 1.80e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic_adv"})                                   # S32: ось B=15, диагностика d
        queued_cfgs.append({
            "R_mm": 2900.0, "pl_radius_mm": 1508.0, "external_B": 15.0,
            "n_density": 5.66e+20, "E_field": 1.80e+17,
            "P_initial": 0.0547, "pump_speed": 0.06, "alignment": 0.0,
            "h_mm": 13.07, "T_inner": 413.55, "T_outer": 297.42,
            "T_plasma_init": 113500000.0, "gap_nm": 1, "k_thermal": 36.84,
            "wall_material": "c_c_sic_adv"})                                   # S33: плато B=15
        # контроли флагов экономики (та же физика, что S31 — различие только флагом)
        queued_cfgs.append({**dict(queued_cfgs[30]), "welded_vessel": True})    # S34: ожидание 87.6627
        queued_cfgs.append({**dict(queued_cfgs[30]),
                            "welded_vessel": True, "emitter": True})            # S35: ожидание 85.9592
        # v46M: FIX смоука (run11/run12) — сид без ключа wall_material получает
        # steel; S5/S8 несут свой tpg, S12–S21 — c_c_sic явно.
        for _sd in queued_cfgs:
            _sd.setdefault("wall_material", "steel")
            # v46Q: флаги экономики — False по умолчанию (смоук старых сидов не тронут)
            _sd.setdefault("welded_vessel", False)
            _sd.setdefault("emitter", False)
        def _pkey(p):
            # v46E FIX: alignment и pump_speed в ключе — S6/S7 (близнецы по физике,
            # но не по экономике) имели одинаковый ключ, и дедупликация молча съела
            # S7 в run7. Теперь контрфакты проходят.
            return (round(float(p.get("n_density", 0)), 3),
                    round(float(p.get("R_mm", 0)), 1),
                    round(float(p.get("pl_radius_mm", 0)), 1),
                    round(float(p.get("pump_speed", 0)), 6),
                    round(float(p.get("alignment", 0)), 4),
                    str(p.get("wall_material", "steel")),
                    # v46Q: флаги в ключе — иначе дедуп съел бы S34/S35
                    bool(p.get("welded_vessel", False)),
                    bool(p.get("emitter", False)))
        seen_params = set()
        try:
            for t in self.study.get_trials(deepcopy=False):
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.WAITING,
                               optuna.trial.TrialState.RUNNING):
                    seen_params.add(_pkey(t.params))
        except Exception:
            pass
        # v46B: ключи сидов для обхода гейта (46.7) + телеметрическая метка seed
        self._seed_pkeys = {_pkey(p) for p in queued_cfgs}
        for _si, seed_params in enumerate(queued_cfgs):
            try:
                def _fmt(x):
                    return f"{x:.4e}" if isinstance(x, float) else str(x)
                if _pkey(seed_params) in seen_params:
                    print(f"[SEED {_si+1}] ПРОПУЩЕН дедупликацией: R={_fmt(seed_params.get('R_mm'))}, "
                          f"n={_fmt(seed_params.get('n_density'))}, al={seed_params.get('alignment')}, mat={seed_params.get('wall_material', 'steel')}")
                    continue
                try:
                    self.study.enqueue_trial(seed_params, user_attrs={"seed": True})
                except TypeError:
                    self.study.enqueue_trial(seed_params)
                seen_params.add(_pkey(seed_params))
                print(f"[SEED {_si+1}] в очереди: R={_fmt(seed_params.get('R_mm'))}, "
                      f"B={_fmt(seed_params.get('external_B'))}, n={_fmt(seed_params.get('n_density'))}, "
                      f"pump={seed_params.get('pump_speed')}, al={seed_params.get('alignment')}, mat={seed_params.get('wall_material', 'steel')}")
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

        # v46O: реэнкью чемпиона — лекарство от "забывчивости" TPE. Веса
        # наблюдений затухают с возрастом: к trial 60+ сиды были ~6%
        # below-множества, и разведка снова позволяла себе 3.4e20. Каждые 20
        # разведочных чемпион возвращается в очередь живой точкой.
        _reenq = {"expl": 0, "done": 0}
        def _reenqueue_champion(study, trial):
            if mode != "lcoe":
                return
            try:
                if (trial.user_attrs or {}).get("seed"):
                    return
                _reenq["expl"] += 1
                if _reenq["expl"] % 10 != 0:  # v46Q: 20 -> 10 (в run19 клонов было мало)
                    return
                bt = study.best_trial
                if bt is None:
                    return
                try:
                    self.study.enqueue_trial(dict(bt.params),
                                             user_attrs={"reenqueue": True})
                except TypeError:
                    self.study.enqueue_trial(dict(bt.params))
                _reenq["done"] += 1
                print(f"[REENQUEUE #{_reenq['done']}] чемпион trial #{bt.number} "
                      f"(LCOE={bt.value:.4f}) в очереди — сидовое знание не выветривается")
            except Exception as _e:
                print(f"[REENQUEUE] skip: {_e}")

        def objective_ignition(trial):
            cfg = self._build_trial_config(trial)
            pr_reason = self._prescreen_checked(trial, cfg)
            if pr_reason:
                return 1e10
            # v46Q: гейты lcoe (только не-сиды): золотая зона + жёсткий Гринвальд.
            # pl в МЕТРАХ (урок v46P: миллиметры занижали n_G в 1000 раз и убивали всех).
            if self.mode == "lcoe":
                _is_seed_q = bool((trial.user_attrs or {}).get("seed", False)) or                              ("fixed_params" in getattr(trial, "system_attrs", {}))
                if not _is_seed_q:
                    try:
                        _qk = (round(float(trial.params.get("n_density", 0) or 0), 3),
                               round(float(trial.params.get("R_mm", 0) or 0), 1),
                               round(float(trial.params.get("pl_radius_mm", 0) or 0), 1),
                               round(float(trial.params.get("pump_speed", 0) or 0), 6),
                               round(float(trial.params.get("alignment", 0) or 0), 4),
                               str(trial.params.get("wall_material", "steel")),
                               bool(trial.params.get("welded_vessel", False)),
                               bool(trial.params.get("emitter", False)))
                        _is_seed_q = _qk in getattr(self, "_seed_pkeys", set())
                    except Exception:
                        _is_seed_q = False
                if not _is_seed_q:
                    _gate = ""
                    try:
                        _Rq = float(trial.params.get("R_mm", 0) or 0)
                        _plq = float(trial.params.get("pl_radius_mm", 0) or 0)
                        _Bq = float(trial.params.get("external_B", 0) or 0)
                        _nq = float(trial.params.get("n_density", 0) or 0)
                        if _Rq > 0 and _plq > 0:
                            _share = _plq / _Rq
                            if not (0.42 - 1e-9 <= _share <= 0.52 + 1e-9):
                                _gate = "golden_zone"
                            else:
                                _nGq = self.greenwald_limit(_Bq, _plq / 1000.0)
                                if np.isfinite(_nGq) and _nq > _nGq:
                                    _gate = "greenwald"
                    except Exception:
                        _gate = "gate_error"
                    trial.set_user_attr("gate_reason", _gate)
                    if _gate:
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
            # v41: P_net_W — ПОСЛЕ бета-штрафа (в v40 писали до: value и P_net_W расходились)
            _beta41 = float(m.get("avg_beta", 0.0))
            _mult41 = 1.0 if _beta41 <= self.BETA_MAX else (self.BETA_MAX / _beta41) ** 2
            _Pnet_raw41 = float((m.get("station") or {}).get("P_net_W", 0.0))
            trial.set_user_attr("P_net_W", _Pnet_raw41 * _mult41)
            trial.set_user_attr("beta", _beta41)
            trial.set_user_attr("beta_penalty_mult", float(_mult41))
            trial.set_user_attr("recirc_raw",
                                float((m.get("station") or {}).get("recirc_raw", 0.0)))
            _B41 = float((cfg.get("plasma_monitor") or {}).get("external_B", 0.0) or 0.0)
            _a41 = float((cfg.get("plasma_monitor") or {}).get("radius", 0.0) or 0.0)
            trial.set_user_attr("greenwald_n", float(self.greenwald_limit(_B41, _a41)))
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
            # v41: телеметрия уже штрафована — переиспользуем (один расчёт, ноль дублей)
            P_net_W = _Pnet_raw41 * _mult41
            # v44: режим lcoe — objective = LCOE ($/МВт·ч) со штрафом за стенку
            if self.mode == "lcoe":
                eco = self._economics(cfg, m)
                for _k, _v in eco.items():
                    trial.set_user_attr(_k, float(_v))
                trial.set_user_attr("P_net_W_final", float(P_net_W))
                # v46D.1: reject ровно по лимиту чешуи 18 (3x множитель снят вместе со штрафом)
                _wall_lim = self.MATERIALS.get(
                    (cfg.get("vacuum_monitor") or {}).get("wall_material", "steel"),
                    self.MATERIALS["steel"])[0]  # v46H: reject по лимиту материала
                if eco["wall_load_W_m2"] > _wall_lim:
                    return 1e10
                # v46M FIX (баг v44, флаг Дипсика на run15): бета-штраф не доходил
                # до LCOE — сжатие при beta>0.15 покупалось багом. Честный LCOE:
                _lcoe46m = float(eco["lcoe"]) / _mult41
                trial.set_user_attr("lcoe_nobeta", float(eco["lcoe"]))
                trial.set_user_attr("lcoe", _lcoe46m)
                return _lcoe46m
            # v45R: жёсткий лимит стенки — reject в power-режиме тоже
            _wall = self._wall_load(cfg, m)
            trial.set_user_attr("wall_load_W_m2", float(_wall))
            if _wall > self.WALL_LIMIT_W_M2:
                return 1e10
            return -P_net_W

        def objective_gain(trial):
            cfg = self._build_trial_config(trial)
            pr_reason = self._prescreen_checked(trial, cfg)
            if pr_reason:
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
            # v41: P_net_W — ПОСЛЕ бета-штрафа + доп. поля телеметрии
            _beta41 = float(m.get("avg_beta", 0.0))
            _mult41 = 1.0 if _beta41 <= self.BETA_MAX else (self.BETA_MAX / _beta41) ** 2
            _Pnet_raw41 = float((m.get("station") or {}).get("P_net_W", 0.0))
            trial.set_user_attr("P_net_W", _Pnet_raw41 * _mult41)
            trial.set_user_attr("beta", _beta41)
            trial.set_user_attr("beta_penalty_mult", float(_mult41))
            trial.set_user_attr("recirc_raw",
                                float((m.get("station") or {}).get("recirc_raw", 0.0)))
            _B41 = float((cfg.get("plasma_monitor") or {}).get("external_B", 0.0) or 0.0)
            _a41 = float((cfg.get("plasma_monitor") or {}).get("radius", 0.0) or 0.0)
            trial.set_user_attr("greenwald_n", float(self.greenwald_limit(_B41, _a41)))
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
            # v41: телеметрия уже штрафована — переиспользуем (один расчёт, ноль дублей)
            P_net_W = _Pnet_raw41 * _mult41
            # v44: режим lcoe — objective = LCOE ($/МВт·ч) со штрафом за стенку
            if self.mode == "lcoe":
                eco = self._economics(cfg, m)
                for _k, _v in eco.items():
                    trial.set_user_attr(_k, float(_v))
                trial.set_user_attr("P_net_W_final", float(P_net_W))
                # v46D.1: reject ровно по лимиту чешуи 18 (3x множитель снят вместе со штрафом)
                _wall_lim = self.MATERIALS.get(
                    (cfg.get("vacuum_monitor") or {}).get("wall_material", "steel"),
                    self.MATERIALS["steel"])[0]  # v46H: reject по лимиту материала
                if eco["wall_load_W_m2"] > _wall_lim:
                    return 1e10
                # v46M FIX (баг v44, флаг Дипсика на run15): бета-штраф не доходил
                # до LCOE — сжатие при beta>0.15 покупалось багом. Честный LCOE:
                _lcoe46m = float(eco["lcoe"]) / _mult41
                trial.set_user_attr("lcoe_nobeta", float(eco["lcoe"]))
                trial.set_user_attr("lcoe", _lcoe46m)
                return _lcoe46m
            # v45R: жёсткий лимит стенки — reject в power-режиме тоже
            _wall = self._wall_load(cfg, m)
            trial.set_user_attr("wall_load_W_m2", float(_wall))
            if _wall > self.WALL_LIMIT_W_M2:
                return 1e10
            return -P_net_W

        def objective_pareto(trial):
            cfg = self._build_trial_config(trial)
            pr_reason = self._prescreen_checked(trial, cfg)
            if pr_reason:
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

        def _safe_obj(fn):
            # v45R.2: исключение в объективе пишется в триал (user_attrs["error"]),
            # триал завершается с reject-сентинелом — study больше не опустевает молча
            _is_pareto = (fn.__name__ == "objective_pareto")
            def _wrapped(trial):
                try:
                    return fn(trial)
                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    try:
                        trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                    except Exception:
                        pass
                    return [-1e10, -1e10] if _is_pareto else 1e10
            return _wrapped
        objective_ignition = _safe_obj(objective_ignition)
        objective_gain = _safe_obj(objective_gain)
        objective_pareto = _safe_obj(objective_pareto)

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
            elif mode == "lcoe":
                # v44: тот же прогон физики, но value = LCOE (минимизируем)
                self.study.optimize(objective_ignition, n_trials=n_trials, show_progress_bar=True,
                                   callbacks=[self._make_progress_callback(n_trials), self._make_stop_callback(),
                                   _reenqueue_champion],  # v46O: память TPE
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

        # v44: автосохранение датасета — экспорт и обучение больше не расходятся
        self._save_training_dataset()

        # v46A: сводка по выравниванию прокачки
        try:
            _al = [t for t in self.study.get_trials(deepcopy=False)
                   if t.state == optuna.trial.TrialState.COMPLETE
                   and "alignment" in t.params and t.value is not None]
            if _al:
                _best_al = min(_al, key=lambda t: t.value)
                print(f"  v46A: лучший с alignment={_best_al.params.get('alignment'):.3f}: "
                      f"LCOE={_best_al.value:.2f} $/МВт·ч")
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
            pr_reason = self._prescreen_checked(trial, cfg)
            if pr_reason:
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

        def _casc_safe(trial):
            try:
                return objective_ignition_cascade(trial)
            except optuna.TrialPruned:
                raise
            except Exception as e:
                try:
                    trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                return 1e10
        study1.optimize(_casc_safe, n_trials=n_trials_phase1,
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
            # v45R.2: пустой study — внятный диагноз вместо голого "Record does not exist"
            _all = self.study.get_trials(deepcopy=False)
            _states = {}
            for _t in _all:
                _states[_t.state.name] = _states.get(_t.state.name, 0) + 1
            if not any(_t.state == optuna.trial.TrialState.COMPLETE for _t in _all):
                _errs = [(t.number, t.user_attrs.get("error", "причина не записана"))
                         for t in _all if t.state == optuna.trial.TrialState.FAIL][-5:]
                raise RuntimeError(
                    "В study нет завершённых триалов. Состояния: {}. "
                    "Последние ошибки FAIL: {}".format(_states, _errs))
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