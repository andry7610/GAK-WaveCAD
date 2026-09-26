#!/usr/bin/env python3
"""
GAK-WaveCAD — Coupling Monitor v0.8

Кросс-связи между всеми 10 модулями:
  1. Магнитоупругая (magnon <-> acoustic)
  2. Пьезоэлектрическая (EM <-> acoustic)
  3. Магнитоэлектрическая (magnon <-> EM)
  4. Тепло-осмос (thermal <-> osmosis)
  5. Плазма-вакуум (plasma <-> vacuum)
  6. Плазма-термальная (plasma <-> thermal, через dE_dt)
  7. Био-акустическая (bio -> acoustic)
  8. Био-плазма (bio -> plasma)
  9. Плазма-квантовый вакуум (plasma <-> quantum, через pair_rate)

v0.8 — float() на всех значениях, coupling_avg=0 без связей,
       нормировочные константы из конфига, coupling_count/coupling_list
v0.7.1 — k_plasma_quantum ограничен 0.8 (вместо 1.0)
v0.7.2 — bio_passive: при k_bio=0 статус HEALTHY вместо DEGRADED
"""

import math

try:
    from core.logger import get_logger
    logger = get_logger("coupling_monitor")
except Exception:
    import logging
    logger = logging.getLogger("coupling_monitor")


class CouplingMonitor:
    VERSION = "v0.8"

    def __init__(self, config=None):
        self.config = config or {}

        # Нормировочные константы (из конфига, с дефолтами)
        # Эти числа масштабируют произведения параметров до [0, 1]
        cfg = self.config
        self.norm_magnetoelastic = float(cfg.get("norm_magnetoelastic", 1e5))
        self.norm_piezoelectric = float(cfg.get("norm_piezoelectric", 1e6))
        self.norm_magnetoelectric = float(cfg.get("norm_magnetoelectric", 1e6))
        self.norm_thermal_osmosis = float(cfg.get("norm_thermal_osmosis", 1e6))
        self.norm_plasma_thermal = float(cfg.get("norm_plasma_thermal", 1e10))
        self.norm_bio_acoustic = float(cfg.get("norm_bio_acoustic", 1e3))
        self.norm_bio_plasma = float(cfg.get("norm_bio_plasma", 1e-3))
        self.norm_plasma_quantum = float(cfg.get("norm_plasma_quantum", 1e3))

        self.results = None
        self._modules = {}

    def init(self):
        logger.info(f"Инициализация Coupling Monitor {self.VERSION}")
        self._modules = {}
        self.results = None
        return True

    def set_results(self, osmosis=None, thermal=None, acoustic=None,
                    collapse=None, magnon=None, em=None,
                    plasma=None, vacuum=None, bio=None,
                    quantum=None):
        self._modules = {
            "osmosis": osmosis or {},
            "thermal": thermal or {},
            "acoustic": acoustic or {},
            "collapse": collapse or {},
            "magnon": magnon or {},
            "em": em or {},
            "plasma": plasma or {},
            "vacuum": vacuum or {},
            "bio": bio or {},
            "quantum": quantum or {},
        }
        count = sum(1 for v in self._modules.values() if v)
        logger.info(f"Результаты загружены от {count} модулей (10)")

    def _extract_acoustic_delta_f(self, acoustic):
        deltas = []
        for key, val in acoustic.items():
            if isinstance(val, dict) and "delta_f" in val:
                deltas.append(abs(float(val["delta_f"])))
        if deltas:
            return sum(deltas) / len(deltas)
        return 0.0

    def _extract_magnon_df(self, magnon):
        dfs = []
        for key, val in magnon.items():
            if isinstance(val, dict) and "df_stress" in val:
                dfs.append(abs(float(val["df_stress"])))
        if dfs:
            return sum(dfs) / len(dfs)
        return 0.0

    def _extract_em_df(self, em):
        dfs = []
        for key, val in em.items():
            if isinstance(val, dict) and "df_stress" in val:
                dfs.append(abs(float(val["df_stress"])))
        if dfs:
            return sum(dfs) / len(dfs)
        return 0.0

    def run(self):
        osm = self._modules.get("osmosis", {})
        th = self._modules.get("thermal", {})
        ac = self._modules.get("acoustic", {})
        col = self._modules.get("collapse", {})
        mag = self._modules.get("magnon", {})
        em = self._modules.get("em", {})
        pl = self._modules.get("plasma", {})
        vac = self._modules.get("vacuum", {})
        bio = self._modules.get("bio", {})
        quant = self._modules.get("quantum", {})

        results = {}

        # --- Геометрический фактор ---
        col_phase = col.get("phase", "STABLE") if col else "STABLE"
        col_ratio = float(col.get("ratio", 0.0)) if col else 0.0

        if col_phase in ("COLLAPSE", "CRITICAL"):
            geom_factor = 0.0
        elif col_phase == "WARNING":
            geom_factor = max(0.0, 1.0 - col_ratio)
        else:
            geom_factor = 1.0
        results["geom_factor"] = geom_factor

        # --- Lawson ---
        if pl:
            lawson_ok = pl.get("lawson_ok", False)
            lawson_triple = float(pl.get("lawson_triple", 0.0))
        else:
            lawson_ok = None
            lawson_triple = 0.0
        results["lawson_ok"] = lawson_ok
        results["lawson_triple"] = lawson_triple
        results["dE_dt"] = float(pl.get("dE_dt", 0.0)) if pl else 0.0
        results["tau_E"] = float(pl.get("tau_E", 0.0)) if pl else 0.0

        # --- 1. Магнитоупругая (magnon <-> acoustic) ---
        ac_df = self._extract_acoustic_delta_f(ac) if ac else 0.0
        mag_df = self._extract_magnon_df(mag) if mag else 0.0
        if ac_df > 0 and mag_df > 0:
            k_magnetoelastic = min(1.0, math.sqrt(ac_df * mag_df) / self.norm_magnetoelastic)
        else:
            k_magnetoelastic = 0.0
        results["k_magnetoelastic"] = k_magnetoelastic

        # --- 2. Пьезоэлектрическая (EM <-> acoustic) ---
        em_df = self._extract_em_df(em) if em else 0.0
        if ac_df > 0 and em_df > 0:
            k_piezoelectric = min(1.0, math.sqrt(ac_df * em_df) / self.norm_piezoelectric)
        else:
            k_piezoelectric = 0.0
        results["k_piezoelectric"] = k_piezoelectric

        # --- 3. Магнитоэлектрическая (magnon <-> EM) ---
        if mag_df > 0 and em_df > 0:
            k_magnetoelectric = min(1.0, math.sqrt(mag_df * em_df) / self.norm_magnetoelectric)
        else:
            k_magnetoelectric = 0.0
        results["k_magnetoelectric"] = k_magnetoelectric

        # --- 4. Тепло-осмос (thermal <-> osmosis) ---
        osm_sigma = float(osm.get("sigma_Pa", 0.0)) if osm else 0.0
        th_sigma = float(th.get("sigma_thermal_Pa", 0.0)) if th else 0.0
        if osm_sigma > 0 and th_sigma > 0:
            k_thermal_osmosis = min(1.0, math.sqrt(osm_sigma * th_sigma) / self.norm_thermal_osmosis)
        else:
            k_thermal_osmosis = 0.0
        results["k_thermal_osmosis"] = k_thermal_osmosis

        # --- 5. Плазма-вакуум ---
        P_gas = 0.0
        if vac:
            P_gas = vac.get("P_gas", vac.get("pressure_gas", vac.get("pressure", 0.0)))
        P_gas = float(P_gas)
        results["P_gas"] = P_gas

        if pl and vac and P_gas >= 0:
            k_plasma_vacuum = 1.0 / (1.0 + P_gas / 1e5)
        else:
            k_plasma_vacuum = 0.0
        results["k_plasma_vacuum"] = k_plasma_vacuum

        # --- 6. Плазма-термальная (через dE_dt) ---
        if pl and th:
            dE_dt = float(pl.get("dE_dt", 0.0))
            k_plasma_thermal = min(1.0, abs(dE_dt) / self.norm_plasma_thermal)
        else:
            k_plasma_thermal = 0.0
        results["k_plasma_thermal"] = k_plasma_thermal

        # --- 7. Био-акустическая ---
        bio_intensity = 0.0
        if bio:
            bio_intensity = bio.get("bio_intensity", bio.get("intensity", 0.0))
        bio_intensity = float(bio_intensity)
        results["bio_intensity"] = bio_intensity

        bio_acoustic_shift = float(bio.get("acoustic_freq_shift", 0.0)) if bio else 0.0
        if bio and ac:
            k_bio_acoustic = min(1.0, bio_acoustic_shift / self.norm_bio_acoustic)
        else:
            k_bio_acoustic = 0.0
        results["k_bio_acoustic"] = k_bio_acoustic

        # --- 8. Био-плазма ---
        bio_feedback = float(bio.get("feedback_current", 0.0)) if bio else 0.0
        if bio and pl:
            k_bio_plasma = min(1.0, abs(bio_feedback) / self.norm_bio_plasma)
        else:
            k_bio_plasma = 0.0
        results["k_bio_plasma"] = k_bio_plasma

        # --- 9. Плазма-квантовый вакуум (через pair_rate) ---
        if quant:
            pair_rate = float(quant.get("pair_rate", 0.0))
            k_plasma_quantum = min(pair_rate * self.norm_plasma_quantum, 0.8)
        else:
            k_plasma_quantum = 0.0
        results["k_plasma_quantum"] = k_plasma_quantum

        # --- Lawson multiplier ---
        if lawson_ok is None or lawson_ok:
            lawson_mult = 1.0
        else:
            lawson_mult = 0.7

        # --- Coupling average ---
        coupling_values = [
            ("magnetoelastic", k_magnetoelastic),
            ("piezoelectric", k_piezoelectric),
            ("magnetoelectric", k_magnetoelectric),
            ("thermal_osmosis", k_thermal_osmosis),
            ("plasma_vacuum", k_plasma_vacuum),
            ("plasma_thermal", k_plasma_thermal),
            ("bio_acoustic", k_bio_acoustic),
            ("bio_plasma", k_bio_plasma),
            ("plasma_quantum", k_plasma_quantum),
        ]
        active = [v for _, v in coupling_values if v > 0]
        active_names = [name for name, v in coupling_values if v > 0]
        if active:
            coupling_avg = sum(active) / len(active)
        else:
            coupling_avg = 0.0

        results["coupling_avg"] = coupling_avg
        results["coupling_count"] = len(active)
        results["coupling_list"] = active_names
        results["no_coupling_data"] = (len(active) == 0)

        # --- Stability index ---
        n_modules = sum(1 for v in self._modules.values() if v)
        if n_modules == 0:
            stability = 1.0  # нет модулей = нет угроз = стабильно
        else:
            stability = geom_factor * lawson_mult
            if active:
                # связи уточняют оценку
                stability = stability * (0.7 + 0.3 * coupling_avg)
        stability = max(0.0, min(1.0, stability))
        results["stability_index"] = stability

        # --- Bio passive detection (v0.7.2) ---
        bio_passive = (k_bio_acoustic == 0.0 and k_bio_plasma == 0.0)
        results["bio_passive"] = bio_passive

        # --- System status ---
        pl_phase = pl.get("phase", "STABLE") if pl else "STABLE"

        if col_phase in ("COLLAPSE", "CRITICAL"):
            system_status = "CRITICAL"
        elif pl_phase == "BREAKDOWN":
            system_status = "CRITICAL"
        elif pl_phase == "WARNING" or col_phase == "WARNING":
            system_status = "DEGRADED"
        elif stability < 0.3:
            system_status = "CRITICAL"
        elif stability < 0.7:
            if bio_passive and len(active) == 0:
                system_status = "HEALTHY"
            else:
                system_status = "DEGRADED"
        else:
            system_status = "HEALTHY"
        results["system_status"] = system_status

        results["lawson_ratio"] = float(lawson_triple) if pl else 0.0

        log_msg = f"Stability={stability:.3f}, status={system_status}, "
        log_msg += f"k_plasma_quantum={k_plasma_quantum:.3f}"
        if bio_passive:
            log_msg += " BIO PASSIVE"
        if len(active) == 0:
            log_msg += " NO_COUPLING"
        logger.info(log_msg)

        self.results = results
        return results

    def get_results(self):
        return self.results

    def print_report(self):
        if not self.results:
            return
        r = self.results
        print(f"\n  Кросс-связи ({self.VERSION}, 10 модулей):")
        print(f"    Stability index : {r['stability_index']:.3f}")
        print(f"    Status          : {r['system_status']}")
        print(f"    Coupling count  : {r.get('coupling_count', 0)}")
        print(f"    Coupling list   : {r.get('coupling_list', [])}")
        if r.get('bio_passive', False):
            print(f"    Bio mode        : PASSIVE")
        print(f"    k_magnetoelastic: {r['k_magnetoelastic']:.3f}")
        print(f"    k_piezoelectric : {r['k_piezoelectric']:.3f}")
        print(f"    k_magnetoelectr.: {r['k_magnetoelectric']:.3f}")
        print(f"    k_thermal_osmos.: {r['k_thermal_osmosis']:.3f}")
        print(f"    k_plasma_vacuum : {r['k_plasma_vacuum']:.3f}")
        print(f"    P_gas           : {r['P_gas']:.3e} Па")
        print(f"    k_bio_acoustic  : {r['k_bio_acoustic']:.3f}")
        print(f"    k_bio_plasma    : {r['k_bio_plasma']:.3f}")
        print(f"    k_plasma_quantum: {r['k_plasma_quantum']:.3f}")
        print(f"    bio_intensity   : {r['bio_intensity']:.3f}")
        print(f"    Lawson ratio    : {r.get('lawson_ratio', 0.0):.3e}")
