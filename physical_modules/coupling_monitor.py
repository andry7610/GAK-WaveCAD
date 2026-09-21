"""
Coupling Monitor — кросс-связи между всеми модулями системы.

Модель:
  Принимает результаты восьми модулей и считает:
    - магнитоупругую связь (акустика ↔ магноны)
    - пьезоэлектрическую связь (акустика ↔ EM)
    - магнитоэлектрическую связь (магноны ↔ EM)
    - тепловой сдвиг осмоса (термалка ↔ осмос)
    - влияние коллапса на геометрию (коллапс → все)
    - плазменные кросс-связи (плазма ↔ все)
    - вакуумные кросс-связи (плазма ↔ вакуум)
  Выдаёт общий индекс стабильности (0..1).

История:
  v0.1 — базовая реализация
  v0.2 — добавлен плазменный модуль
  v0.3 — подхват Лоусона, dE_dt, tau_E из плазмы v0.2
  v0.4 — вакуум: k_plasma_vacuum, P_gas → плазма
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("coupling_monitor")
class CouplingMonitor(BaseModule):
    """Анализ кросс-связей между модулями системы."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}
        self.B_me_scale = cfg.get("B_me_scale", kwargs.get("B_me_scale", 1.0))
        self.alpha_me_scale = cfg.get("alpha_me_scale", kwargs.get("alpha_me_scale", 1.0))

        # Результаты от других модулей
        self.osmosis_results = None
        self.thermal_results = None
        self.acoustic_results = None
        self.collapse_results = None
        self.magnon_results = None
        self.em_results = None
        self.plasma_results = None
        self.vacuum_results = None

        self.logger = get_logger("coupling_monitor")

    def init(self) -> bool:
        self._set_initialized(True)
        self.logger.info("Инициализация Coupling Monitor")
        return True

    def set_results(self, osmosis=None, thermal=None, acoustic=None,
                    collapse=None, magnon=None, em=None, plasma=None,
                    vacuum=None):
        """Загрузить результаты от всех модулей."""
        self.osmosis_results = osmosis
        self.thermal_results = thermal
        self.acoustic_results = acoustic
        self.collapse_results = collapse
        self.magnon_results = magnon
        self.em_results = em
        self.plasma_results = plasma
        self.vacuum_results = vacuum
        self.logger.info("Результаты загружены от всех модулей")

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        results = {}

        # --- 1. Магнитоупругая связь: акустика ↔ магноны ---
        if self.acoustic_results and self.magnon_results:
            ac_shifts = [r.get('delta_f', 0) for r in self.acoustic_results.values()]
            mag_shifts = [r.get('df_stress', 0) for r in self.magnon_results.values()]
            ac_norm = np.sqrt(np.mean(np.square(ac_shifts)))
            mag_norm = np.sqrt(np.mean(np.square(mag_shifts)))
            k_magnetoelastic = ac_norm * mag_norm / (ac_norm + mag_norm + 1e-30)
            results['k_magnetoelastic'] = k_magnetoelastic
        else:
            results['k_magnetoelastic'] = 0.0

        # --- 2. Пьезоэлектрическая связь: акустика ↔ EM ---
        if self.acoustic_results and self.em_results:
            ac_shifts = [r.get('delta_f', 0) for r in self.acoustic_results.values()]
            em_shifts = [r.get('df_stress', 0) for r in self.em_results.values()]
            ac_norm = np.sqrt(np.mean(np.square(ac_shifts)))
            em_norm = np.sqrt(np.mean(np.square(em_shifts)))
            k_piezo = ac_norm * em_norm / (ac_norm + em_norm + 1e-30)
            results['k_piezoelectric'] = k_piezo
        else:
            results['k_piezoelectric'] = 0.0

        # --- 3. Магнитоэлектрическая связь: магноны ↔ EM ---
        if self.magnon_results and self.em_results:
            mag_freqs = [r.get('f', 0) for r in self.magnon_results.values()]
            em_freqs = [r.get('f_shifted', 0) for r in self.em_results.values()]
            mag_mean = np.mean(mag_freqs)
            em_mean = np.mean(em_freqs)
            k_me = abs(mag_mean - em_mean) / (mag_mean + em_mean + 1e-30)
            results['k_magnetoelectric'] = k_me * self.alpha_me_scale
        else:
            results['k_magnetoelectric'] = 0.0

        # --- 4. Тепловой сдвиг осмоса: термалка ↔ осмос ---
        if self.osmosis_results and self.thermal_results:
            sigma_osm = self.osmosis_results.get('sigma_Pa', 0)
            sigma_th = self.thermal_results.get('sigma_thermal_Pa', 0)
            total = sigma_osm + sigma_th + 1e-30
            k_thermal_osmosis = sigma_th / total
            results['k_thermal_osmosis'] = k_thermal_osmosis
        else:
            results['k_thermal_osmosis'] = 0.0

        # --- 5. Влияние коллапса на геометрию ---
        if self.collapse_results:
            phase = self.collapse_results.get('phase', 'STABLE')
            ratio = self.collapse_results.get('ratio', 0.0)
            if phase == 'COLLAPSE':
                geom_factor = 0.0
            elif phase == 'WARNING':
                geom_factor = 1.0 - ratio
            elif phase == 'INFLATION':
                geom_factor = 1.0
            else:
                geom_factor = 1.0 - 0.5 * ratio
            results['geom_factor'] = max(0.0, min(1.0, geom_factor))
            results['collapse_phase'] = phase
            results['collapse_ratio'] = ratio
        else:
            results['geom_factor'] = 1.0
            results['collapse_phase'] = 'UNKNOWN'
            results['collapse_ratio'] = 0.0

        # --- 6. Плазменные кросс-связи ---
        if self.plasma_results:
            # Плазма ↔ Акустика: вязкое напряжение сдвигает резонансные частоты
            sigma_visc = self.plasma_results.get("sigma_viscous", 0.0)
            acoustic = self.acoustic_results or {}
            max_ac_stress = max(
                (abs(r.get('delta_f', 0)) for r in acoustic.values()),
                default=1.0
            )
            k_plasma_acoustic = sigma_visc / max(max_ac_stress, 1.0)
            results['k_plasma_acoustic'] = k_plasma_acoustic

            # Плазма ↔ Магноны: магнитное поле плазмы сдвигает Kittel-моды
            B_plasma = self.plasma_results.get("B_field", 0.0)
            magnon = self.magnon_results or {}
            B_sat = magnon.get("saturation_field", 1.0) if isinstance(magnon, dict) else 1.0
            k_plasma_magnon = B_plasma / max(B_sat, 1.0)
            results['k_plasma_magnon'] = k_plasma_magnon

            # Плазма ↔ EM: диффузия влияет на добротность резонатора
            D = self.plasma_results.get("diffusion_coeff", 0.0)
            k_plasma_em = min(D * 1e4, 1.0)
            results['k_plasma_em'] = k_plasma_em

            # Плазма ↔ Термалка: температура плазмы + энергобаланс
            T_plasma = self.plasma_results.get("temperature_plasma", 0.0)
            thermal = self.thermal_results or {}
            T_thermal = thermal.get("temperature", 300.0) if isinstance(thermal, dict) else 300.0
            k_plasma_thermal = (T_plasma - T_thermal) / max(T_thermal, 1.0) if T_plasma > 0 else 0.0

            # v0.3: энергобаланс dE_dt усиливает thermal coupling
            dE_dt = self.plasma_results.get("dE_dt", 0.0)
            if dE_dt > 0:
                k_plasma_thermal = min(k_plasma_thermal + dE_dt * 1e-6, 1.0)
            results['k_plasma_thermal'] = k_plasma_thermal

            # Плазма ↔ Коллапс: барьерный индекс влияет на устойчивость оболочки
            barrier = self.plasma_results.get("barrier_index", 0.0)
            k_plasma_collapse = 1.0 - barrier
            results['k_plasma_collapse'] = k_plasma_collapse
        else:
            results['k_plasma_acoustic'] = 0.0
            results['k_plasma_magnon'] = 0.0
            results['k_plasma_em'] = 0.0
            results['k_plasma_thermal'] = 0.0
            results['k_plasma_collapse'] = 0.0

        # --- 7. Вакуумные кросс-связи: плазма ↔ вакуум ---
        if self.plasma_results and self.vacuum_results:
            # Газовое давление из вакуума охлаждает плазму
            P_gas = self.vacuum_results.get("pressure", 0.0)
            T_plasma = self.plasma_results.get("temperature_plasma", 0.0)
            # Нормировка: 1 Па — умеренное давление для плазменной камеры
            k_plasma_vacuum = min(P_gas / 1.0, 1.0) if P_gas > 0 else 0.0
            results['k_plasma_vacuum'] = k_plasma_vacuum
            results['P_gas'] = P_gas

            # Вакуум ↔ Термалка: отвод тепла через газ
            pump_speed = self.vacuum_results.get("pump_speed", 0.0)
            results['k_vacuum_thermal'] = min(pump_speed * 1e3, 1.0)
        else:
            results['k_plasma_vacuum'] = 0.0
            results['P_gas'] = 0.0
            results['k_vacuum_thermal'] = 0.0

        # --- 8. Общий индекс стабильности ---
        stability = (
            results['geom_factor'] * 0.25 +
            (1.0 - min(results['k_magnetoelastic'], 1.0)) * 0.15 +
            (1.0 - min(results['k_piezoelectric'], 1.0)) * 0.15 +
            (1.0 - min(results['k_magnetoelectric'], 1.0)) * 0.10 +
            (1.0 - min(results['k_thermal_osmosis'], 1.0)) * 0.10 +
            (1.0 - min(results['k_plasma_acoustic'], 1.0)) * 0.05 +
            (1.0 - min(results['k_plasma_magnon'], 1.0)) * 0.05 +
            (1.0 - min(results['k_plasma_collapse'], 1.0)) * 0.05 +
            (1.0 - min(results['k_plasma_em'], 1.0)) * 0.05 +
            (1.0 - min(results['k_plasma_thermal'], 1.0)) * 0.05 +
            (1.0 - min(results['k_plasma_vacuum'], 1.0)) * 0.05
        )
        stability = max(0.0, min(1.0, stability))

        # v0.3: штраф за незажигание Лоусона
        if self.plasma_results:
            lawson_ok = self.plasma_results.get("lawson_ok", False)
            if not lawson_ok:
                stability *= 0.7
            results['lawson_ok'] = lawson_ok
            results['lawson_triple'] = self.plasma_results.get("lawson_triple", 0.0)
            results['tau_E'] = self.plasma_results.get("tau_E", 0.0)
            results['dE_dt'] = self.plasma_results.get("dE_dt", 0.0)
        else:
            results['lawson_ok'] = None
            results['lawson_triple'] = 0.0
            results['tau_E'] = 0.0
            results['dE_dt'] = 0.0

        # Фаза плазмы влияет на системный статус
        if self.plasma_results:
            plasma_phase = self.plasma_results.get("phase", "STABLE")
            if plasma_phase == "BREAKDOWN":
                system_status = "CRITICAL"
            elif plasma_phase == "CONTACT":
                system_status = "CRITICAL"
            elif plasma_phase == "WARNING":
                system_status = "DEGRADED"
            elif stability > 0.7:
                system_status = "HEALTHY"
            elif stability > 0.4:
                system_status = "DEGRADED"
            else:
                system_status = "CRITICAL"
        else:
            if stability > 0.7:
                system_status = "HEALTHY"
            elif stability > 0.4:
                system_status = "DEGRADED"
            else:
                system_status = "CRITICAL"

        results['stability_index'] = stability
        results['system_status'] = system_status

        self.logger.info(f"Stability={stability:.3f}, status={system_status}")
        self.results = results
        return True

    def get_results(self) -> dict:
        return self.results

    def print_report(self):
        print("\n  Coupling Monitor — отчёт:")
        print(f"    k_magnetoelastic  : {self.results['k_magnetoelastic']:.4e}")
        print(f"    k_piezoelectric   : {self.results['k_piezoelectric']:.4e}")
        print(f"    k_magnetoelectric : {self.results['k_magnetoelectric']:.4e}")
        print(f"    k_thermal_osmosis : {self.results['k_thermal_osmosis']:.4f}")
        print(f"    k_plasma_acoustic : {self.results['k_plasma_acoustic']:.4e}")
        print(f"    k_plasma_magnon   : {self.results['k_plasma_magnon']:.4e}")
        print(f"    k_plasma_em       : {self.results['k_plasma_em']:.4e}")
        print(f"    k_plasma_thermal  : {self.results['k_plasma_thermal']:.4e}")
        print(f"    k_plasma_collapse : {self.results['k_plasma_collapse']:.4e}")
        print(f"    k_plasma_vacuum   : {self.results['k_plasma_vacuum']:.4e}")
        print(f"    P_gas             : {self.results['P_gas']:.4e} Па")
        print(f"    geom_factor       : {self.results['geom_factor']:.3f}")
        print(f"    collapse_phase    : {self.results['collapse_phase']}")
        print(f"    stability_index   : {self.results['stability_index']:.3f}")
        print(f"    system_status     : {self.results['system_status']}")
        # v0.3
        print(f"    lawson_ok         : {self.results.get('lawson_ok', None)}")
        print(f"    lawson_triple     : {self.results.get('lawson_triple', 0):.3e}")
        print(f"    tau_E             : {self.results.get('tau_E', 0):.3e} с")
        print(f"    dE_dt             : {self.results.get('dE_dt', 0):.3e} Вт/м³")
