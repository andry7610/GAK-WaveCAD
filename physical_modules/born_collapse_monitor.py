"""
Born Collapse Monitor — анализ устойчивости сферической оболочки.

Модель:
  Критическое давление коллапса по формуле Зоэлли:
    P_cr = 2*E / sqrt(3*(1-nu^2)) * (h/R)^2
  Сравнение внешнего давления с критическим:
    P < P_cr  → устойчива
    P >= P_cr → коллапс
    P < 0     → надувание

История:
  v0.1 — базовая реализация
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
        self.R = cfg.get("R", kwargs.get("R", 0.05))
        self.h = cfg.get("h", kwargs.get("h", self.R * 0.04))
        self.E = cfg.get("E", kwargs.get("E", 2.0e9))
        self.nu = cfg.get("nu", kwargs.get("nu", 0.33))
        self.external_pressure = cfg.get("external_pressure", kwargs.get("external_pressure", 0.0))
        self.safety_factor = cfg.get("safety_factor", kwargs.get("safety_factor", 1.5))

        self.logger = get_logger("born_collapse_monitor")

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.E <= 0:
            raise ValueError("E must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, h={self.h*1e3:.1f} мм")
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        # Критическое давление коллапса (Зоэлли)
        P_cr = (2 * self.E / np.sqrt(3 * (1 - self.nu**2))) * (self.h / self.R)**2

        # Допустимое давление с запасом прочности
        P_allowable = P_cr / self.safety_factor

        # Сравнение
        P_ext = self.external_pressure
        ratio = P_ext / P_cr if P_cr > 0 else 0.0

        if P_ext < 0:
            phase = "INFLATION"
        elif P_ext >= P_cr:
            phase = "COLLAPSE"
        elif P_ext >= P_allowable:
            phase = "WARNING"
        else:
            phase = "STABLE"

        margin = P_cr - P_ext

        self.logger.info(f"P_cr={P_cr:.1f} Па, P_ext={P_ext:.1f} Па, phase={phase}")

        self.results = {
            'P_cr': P_cr,
            'P_cr_MPa': P_cr / 1e6,
            'P_ext': P_ext,
            'P_allowable': P_allowable,
            'safety_factor': self.safety_factor,
            'ratio': ratio,
            'margin': margin,
            'phase': phase,
        }

        return True

    def get_results(self) -> dict:
        return self.results

    def set_external_pressure(self, pressure_Pa: float):
        self.external_pressure = pressure_Pa
        self.logger.info(f"Установлено внешнее давление: {pressure_Pa:.1f} Па")

    def print_report(self):
        print("\n  Born Collapse Monitor — отчёт:")
        print(f"    P_cr        : {self.results['P_cr']:.1f} Па "
              f"({self.results['P_cr_MPa']:.3f} МПа)")
        print(f"    P_ext       : {self.results['P_ext']:.1f} Па")
        print(f"    P_allowable : {self.results['P_allowable']:.1f} Па "
              f"(k={self.results['safety_factor']:.1f})")
        print(f"    ratio       : {self.results['ratio']:.4f}")
        print(f"    margin      : {self.results['margin']:.1f} Па")
        print(f"    phase       : {self.results['phase']}")
