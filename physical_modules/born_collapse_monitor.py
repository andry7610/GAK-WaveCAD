"""
Born Collapse Monitor — анализ устойчивости сферической оболочки.

Модель:
  Критическое давление коллапса по формуле Зоэлли:
    P_cr = 2*E / sqrt(3*(1-nu^2)) * (h/R)^2
  Сравнение внешнего давления с критическим:
    P < P_cr  -> устойчива
    P >= P_cr -> коллапс
    P < 0     -> надувание

  v0.2 — imperfection_factor (реальное P_cr с учётом несовершенств),
         float() на параметрах, fallback ключей R/R_shell и h/h_wall

История:
  v0.1 — базовая реализация
  v0.2 — imperfection_factor, float(), fallback
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("born_collapse_monitor")
class BornCollapseMonitor(BaseModule):
    """Анализ коллапса сферической оболочки под давлением."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}

        # float() на всех параметрах + fallback R <- R_shell, h <- h_wall
        R_val = cfg.get("R", kwargs.get("R", None))
        if R_val is None:
            R_val = cfg.get("R_shell", kwargs.get("R_shell", 0.05))
        self.R = float(R_val)

        h_val = cfg.get("h", kwargs.get("h", None))
        if h_val is None:
            h_val = cfg.get("h_wall", kwargs.get("h_wall", self.R * 0.04))
        self.h = float(h_val)

        self.E = float(cfg.get("E", kwargs.get("E", 2.0e9)))
        self.nu = float(cfg.get("nu", kwargs.get("nu", 0.33)))
        self.external_pressure = float(cfg.get("external_pressure",
                                     kwargs.get("external_pressure", 0.0)))
        self.safety_factor = float(cfg.get("safety_factor",
                                  kwargs.get("safety_factor", 1.5)))
        # Imperfection factor: реальная оболочка теряет устойчивость
        # при давлении в 3-5 раз ниже идеальной (0.2-0.3)
        self.imperfection_factor = float(cfg.get("imperfection_factor",
                                        kwargs.get("imperfection_factor", 0.25)))
        if self.imperfection_factor <= 0 or self.imperfection_factor > 1:
            raise ValueError("imperfection_factor must be in (0, 1]")

        self.logger = get_logger("born_collapse_monitor")

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.E <= 0:
            raise ValueError("E must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, h={self.h*1e3:.1f} мм, "
                         f"imperfection={self.imperfection_factor:.2f}")
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        # Критическое давление коллапса (Зоэлли) — для идеальной сферы
        P_cr_theory = (2 * self.E / np.sqrt(3 * (1 - self.nu**2))) * (self.h / self.R)**2

        # Реальное критическое давление с учётом несовершенств
        P_cr = P_cr_theory * self.imperfection_factor

        # Допустимое давление с запасом прочности
        P_allowable = P_cr / self.safety_factor

        P_ext = self.external_pressure
        ratio = P_ext / P_cr if P_cr > 0 else 0.0
        ratio_theory = P_ext / P_cr_theory if P_cr_theory > 0 else 0.0

        if P_ext < 0:
            phase = "INFLATION"
        elif P_ext >= P_cr:
            phase = "COLLAPSE"
        elif P_ext >= P_allowable:
            phase = "WARNING"
        else:
            phase = "STABLE"

        margin = P_cr - P_ext

        self.logger.info(f"P_cr_theory={P_cr_theory:.1f} Па, P_cr_real={P_cr:.1f} Па, "
                         f"P_ext={P_ext:.1f} Па, phase={phase}")

        self.results = {
            'P_cr': P_cr,
            'P_cr_MPa': P_cr / 1e6,
            'P_cr_theory': P_cr_theory,
            'P_cr_theory_MPa': P_cr_theory / 1e6,
            'P_ext': P_ext,
            'P_allowable': P_allowable,
            'safety_factor': self.safety_factor,
            'imperfection_factor': self.imperfection_factor,
            'ratio': ratio,
            'ratio_theory': ratio_theory,
            'margin': margin,
            'phase': phase,
        }

        return True

    def get_results(self) -> dict:
        return self.results

    def get_critical_pressure_Pa(self) -> float:
        """Возвращает реальное критическое давление (с imperfection)."""
        return self.results.get('P_cr', 0.0)

    def set_external_pressure(self, pressure_Pa: float):
        self.external_pressure = pressure_Pa
        self.logger.info(f"Установлено внешнее давление: {pressure_Pa:.1f} Па")

    def print_report(self):
        r = self.results
        print("\n  Born Collapse Monitor — отчёт:")
        print(f"    P_cr_theory : {r['P_cr_theory']:.1f} Па ({r['P_cr_theory_MPa']:.3f} МПа)")
        print(f"    P_cr_real   : {r['P_cr']:.1f} Па ({r['P_cr_MPa']:.3f} МПа) "
              f"[imperfection={r['imperfection_factor']:.2f}]")
        print(f"    P_ext       : {r['P_ext']:.1f} Па")
        print(f"    P_allowable : {r['P_allowable']:.1f} Па (k={r['safety_factor']:.1f})")
        print(f"    ratio       : {r['ratio']:.4f} (theory: {r['ratio_theory']:.4f})")
        print(f"    margin      : {r['margin']:.1f} Па")
        print(f"    phase       : {r['phase']}")
