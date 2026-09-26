"""
GAK-WaveCAD — Optuna-оптимизатор v9.3.
Фиксы: constraint pl_radius < R*0.8, синхронизация B, честные метрики.
v9.1: D-T фиксирован, B ≤ 45, Q с P_sync, 50000 шагов, T_CAP = 1e9.
v9.2: Adaptive dt в _run_quick (Вариант B от DeepSeek) — dt = min(sim_dt, 0.05*tau_E).
v9.3: n_steps = 600000 (3*tau_E для полного схода к равновесию).
"""

import optuna
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


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

    # Суженные, физически осмысленные диапазоны
    BOUNDS = {
        "R_mm":           (50, 1500),
        "h_mm":           (0.5, 20.0),
        "T_inner":        (300.0, 400.0),
        "T_outer":        (273.0, 310.0),
        "k_thermal":      (1.0, 100.0),
        "pl_radius_mm":    (5, 600),       # верхняя граница — динамическая (R*0.8)
        "T_plasma_init":  (1e7, 1.2e9),    # 10 МК — 100 кэВ (расширен для равновесия)
        "n_density":      (1e20, 1e22),
        "fuel_type_idx":  (0, 2),
        "P_initial":      (1e-3, 1e5),
        "pump_speed":     (1e-4, 1e-1),
        "E_field":        (1e16, 3e18),    # не выше ~2× Швингера
        "gap_nm":          (1, 50),
        "external_B":     (5.0, 45.0),     # Тл — ограничено (v9.1)
    }

    T_CAP = 1e9  # 1000 МК — выше это runaway (поднят для равновесия D-T)

    # v9.1: D-T зафиксирован
    FIXED_FUEL_IDX = 1  # D-T

    def __init__(self, base_config):
        self.base_config = base_config
        self.study = None
        self.results = None
        self.mode = None

    def _build_trial_config(self, trial):
        """Конфиг из trial. Constraint: pl_radius < R_mm * 0.8."""
        cfg = dict(self.base_config)

        R_mm = trial.suggest_float("R_mm", *self.BOUNDS["R_mm"])
        h_mm = trial.suggest_float("h_mm", *self.BOUNDS["h_mm"])
        T_inner = trial.suggest_float("T_inner", *self.BOUNDS["T_inner"])
        T_outer = trial.suggest_float("T_outer", *self.BOUNDS["T_outer"])
        k_thermal = trial.suggest_float("k_thermal", *self.BOUNDS["k_thermal"])

        # Constraint: плазма внутри сферы
        pl_radius_max = R_mm * 0.4  # радиус плазмы < 40% радиуса сферы
        pl_radius_mm = trial.suggest_float("pl_radius_mm",
                                           max(5, R_mm * 0.05),  # не меньше 5% от R
                                           max(pl_radius_max, R_mm * 0.06))
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
        cfg["simulation"] = {"n_steps": 600000, "dt": 1e-5}
        return cfg

    def _run_quick(self, cfg):
        """Упрощённый прогон: plasma.step() + ручной Швингер. Без run().
        v9.2: Adaptive dt — dt_actual = min(sim_dt, 0.05 * tau_E)."""
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
            dt_actual_vals = []
            sim_time = 0.0

            runaway = False

            for step_i in range(sim_n):
                # v9.2: Adaptive dt — не больше 5% от tau_E
                tau_E_current = plasma.compute_tau_E()
                if tau_E_current > 0:
                    dt_actual = min(sim_dt, 0.05 * tau_E_current)
                else:
                    dt_actual = sim_dt

                step_result = plasma.step(dt_actual)
                sim_time += dt_actual

                T_plasma = step_result["temperature"]
                P_fusion = step_result["P_fusion"]
                P_cond = step_result["P_cond"]
                P_rad = step_result["P_rad"]
                tau_E = step_result["tau_E"]

                # P_sync и P_alpha из step_result (v0.9 plasma_monitor)
                P_sync = step_result.get("P_sync", 0.0)
                P_alpha = step_result.get("P_alpha", 0.2 * P_fusion)
                alpha_frac = step_result.get("alpha_frac", 0.2)

                # Швингеровский нагрев (вручную, без run())
                pair_rate = 0.0
                qv_cfg = cfg.get("quantum_vacuum", {})
                E_field = qv_cfg.get("E_field", 0.0)
                gap = qv_cfg.get("gap", 1e-9)

                # Швингер: pair_rate ~ E^2 * exp(-pi*E_c/E)
                E_c = 1.32e18
                if E_field > 0.01 * E_c:
                    schwinger_factor = np.exp(-np.pi * E_c / max(E_field, 1e-30))
                    pair_rate = 1e45 * (E_field / E_c) ** 2 * schwinger_factor / (gap * 1e9)
                    pair_rate = min(pair_rate, 1e20)  # cap

                if pair_rate > 0 and plasma.quantum_feedback:
                    dE_schwinger = pair_rate * pair_energy_gain * dt_actual
                    if V_plasma > 0 and plasma.density_number > 0:
                        dT_schwinger = dE_schwinger / (1.5 * plasma.density_number * 1.380649e-23 * V_plasma)
                        plasma.temperature += dT_schwinger

                # Cap температуры
                if plasma.temperature > self.T_CAP:
                    runaway = True
                    break

                if np.isnan(plasma.temperature) or plasma.temperature <= 0:
                    break

                # Диагностический ratio
                P_heat = pair_rate * pair_energy_gain
                P_heat_d = P_heat / V_plasma if V_plasma > 0 else 0.0
                ratio = P_heat_d / P_cond if P_cond > 0 and P_cond < 1e30 else 0.0

                # v9.1: Q с P_sync — честный энергобаланс
                P_loss = P_cond + P_rad + P_sync
                Q = P_fusion / P_loss if P_loss > 0 and P_loss < 1e30 else 0.0

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
                dt_actual_vals.append(dt_actual)

            if len(T_vals) < 10:
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
            }

        except Exception as e:
            return None

    def optimize(self, mode: str, n_trials: int = 50):
        self.mode = mode

        if mode == "pareto":
            self.study = optuna.create_study(
                directions=["maximize", "maximize"],
                sampler=optuna.samplers.TPESampler(seed=42),
            )
        else:
            self.study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )

        def objective_ignition(trial):
            cfg = self._build_trial_config(trial)
            m = self._run_quick(cfg)
            if m is None:
                return 1e10
            if m["runaway"]:
                return 1e10
            Q = m["Q"]
            lawson_ratio = m["lawson_product"] / m["lawson_threshold"]
            penalty = 0.0
            if lawson_ratio < 0.1:
                penalty = (0.1 - lawson_ratio) * 100
            return -Q + penalty

        def objective_gain(trial):
            cfg = self._build_trial_config(trial)
            m = self._run_quick(cfg)
            if m is None:
                return 1e10
            if m["runaway"]:
                return 1e10
            return -m["Q"]

        def objective_pareto(trial):
            cfg = self._build_trial_config(trial)
            m = self._run_quick(cfg)
            if m is None or m["runaway"]:
                return [-1e10, -1e10]
            return [m["avg_pair_rate"], m["Q"]]

        if mode == "ignition":
            self.study.optimize(objective_ignition, n_trials=n_trials, show_progress_bar=False)
        elif mode == "gain":
            self.study.optimize(objective_gain, n_trials=n_trials, show_progress_bar=False)
        elif mode == "pareto":
            self.study.optimize(objective_pareto, n_trials=n_trials, show_progress_bar=False)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        self._extract_results()

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
            ("Q-фактор (средн.)", f"{m['avg_Q']:.6f}" if m else "N/A"),
            ("Lawson n*T*tau", f"{m['lawson_product']:.3e}" if m else "N/A"),
            ("Порог Лоусона", f"{m['lawson_threshold']:.3e}" if m else "N/A"),
            ("Lawson/threshold", f"{m['lawson_product']/m['lawson_threshold']:.4f}" if m else "N/A"),
            ("Лоусон достигнут?", "ДА" if m and m["lawson_achieved"] else "НЕТ"),
            ("Runaway (T > cap)?", "ДА" if m and m["runaway"] else "НЕТ"),
            ("T плазмы (равн.), K", f"{m['avg_T']:.3e}" if m else "N/A"),
            ("T_keV", f"{m['T_keV']:.2f}" if m else "N/A"),
            ("P_fusion, Вт", f"{m['avg_P_fusion']:.4e}" if m else "N/A"),
            ("P_alpha, Вт", f"{m['avg_P_alpha']:.4e}" if m else "N/A"),
            ("P_cond, Вт", f"{m['avg_P_cond']:.4e}" if m else "N/A"),
            ("P_rad, Вт", f"{m['avg_P_rad']:.4e}" if m else "N/A"),
            ("P_sync, Вт", f"{m['avg_P_sync']:.4e}" if m else "N/A"),
            ("tau_E, с", f"{m['avg_tau_E']:.4e}" if m else "N/A"),
            ("pair_rate (средн.)", f"{m['avg_pair_rate']:.4e}" if m else "N/A"),
            ("ratio (диагн.)", f"{m['avg_ratio']:.4f}" if m else "N/A"),
            ("B внешнее (метр.), Тл", f"{m['external_B']:.1f}" if m else "N/A"),
            ("Радиус плазмы (факт.), м", f"{m['plasma_radius_actual_m']:.4f}" if m else "N/A"),
            ("V плазмы, м3", f"{m['V_plasma']:.4e}" if m else "N/A"),
            ("Шагов выполнено", f"{m['steps_done']}" if m else "N/A"),
            ("Время симуляции, с", f"{m['sim_time']:.4f}" if m else "N/A"),
            ("Ср. dt_actual, с", f"{m['avg_dt_actual']:.4e}" if m else "N/A"),
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
            ax.set_ylabel("Q-фактор (+ штраф)")
            ax.set_title("Сходимость: Q -> max (режим зажигания)")
        elif self.mode == "gain":
            ax.plot(df["number"], -df["value"], color="navy", linewidth=2)
            ax.set_ylabel("Q-фактор")
            ax.set_title("Сходимость: Q -> max (режим усиления)")

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
            "Q": m["avg_Q"] if m else 0,
            "lawson": m["lawson_product"] if m else 0,
            "lawson_threshold": m["lawson_threshold"] if m else 0,
            "lawson_achieved": m["lawson_achieved"] if m else False,
            "runaway": m["runaway"] if m else False,
            "T_plasma_eq": m["avg_T"] if m else 0,
            "P_fusion": m["avg_P_fusion"] if m else 0,
            "P_alpha": m["avg_P_alpha"] if m else 0,
            "P_cond": m["avg_P_cond"] if m else 0,
            "P_rad": m["avg_P_rad"] if m else 0,
            "P_sync": m["avg_P_sync"] if m else 0,
            "tau_E": m["avg_tau_E"] if m else 0,
            "plasma_radius_actual_m": m["plasma_radius_actual_m"] if m else 0,
            "steps_done": m["steps_done"] if m else 0,
            "sim_time": m["sim_time"] if m else 0,
            "avg_dt_actual": m["avg_dt_actual"] if m else 0,
        }
        return pd.DataFrame([row])
