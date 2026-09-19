"""
Thermal Monitor — анализ теплового состояния сферической оболочки.

Модель:
  Стационарное распределение температуры по толщине оболочки.
  Тепловой поток от внутренней стенки к внешней.
  Тепловое напряжение передаётся в акустический модуль.

История:
  v0.1 — базовая реализация
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("thermal_monitor")
class ThermalMonitor(BaseModule):
    """Анализ тепловых напряжений в сферической оболочке."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}
        self.R = cfg.get("R", kwargs.get("R", 0.05))
        self.h = cfg.get("h", kwargs.get("h", self.R * 0.04))
        self.k = cfg.get("k", kwargs.get("k", 0.6))
        self.alpha = cfg.get("alpha", kwargs.get("alpha", 70e-6))
        self.E = cfg.get("E", kwargs.get("E", 2.0e9))
        self.nu = cfg.get("nu", kwargs.get("nu", 0.33))
        self.T_inner = cfg.get("T_inner", kwargs.get("T_inner", 310.0))
        self.T_outer = cfg.get("T_outer", kwargs.get("T_outer", 293.0))
        self.n_layers = cfg.get("n_layers", kwargs.get("n_layers", 20))

        self.logger = get_logger("thermal_monitor")

        self.R_inner = self.R - self.h / 2
        self.R_outer = self.R + self.h / 2

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, h={self.h*1e3:.1f} мм")
        self.logger.info(f"T_inner={self.T_inner:.1f} K, T_outer={self.T_outer:.1f} K")
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        r = np.linspace(self.R_inner, self.R_outer, self.n_layers)
        dT = self.T_inner - self.T_outer
        inv_factor = (1 / self.R_inner - 1 / self.R_outer)
        T_profile = self.T_inner - dT * (1 / self.R_inner - 1 / r) / inv_factor

        Q = 4 * np.pi * self.k * dT / inv_factor
        T_avg = np.mean(T_profile)
        sigma_thermal = self.E * self.alpha * dT / (1 - self.nu)
        grad_T = dT / self.h

        self.logger.info(f"Тепловой поток: Q={Q:.3f} Вт")
        self.logger.info(f"Тепловое напряжение: sigma={sigma_thermal:.1f} Па")

        self.results = {
            'T_inner': self.T_inner,
            'T_outer': self.T_outer,
            'dT': dT,
            'T_avg': T_avg,
            'grad_T': grad_T,
            'Q': Q,
            'sigma_thermal_Pa': sigma_thermal,
            'sigma_thermal_MPa': sigma_thermal / 1e6,
            'r_profile': r,
            'T_profile': T_profile,
        }

        return True

    def get_results(self) -> dict:
        return self.results

    def get_thermal_stress_Pa(self) -> float:
        return self.results['sigma_thermal_Pa']

    def print_report(self):
        print("\n  Thermal Monitor — отчёт:")
        print(f"    T_inner  : {self.results['T_inner']:.1f} K")
        print(f"    T_outer  : {self.results['T_outer']:.1f} K")
        print(f"    dT       : {self.results['dT']:.1f} K")
        print(f"    T_avg    : {self.results['T_avg']:.1f} K")
        print(f"    grad_T   : {self.results['grad_T']:.1f} K/m")
        print(f"    Q        : {self.results['Q']:.3f} Вт")
        print(f"    sigma_th : {self.results['sigma_thermal_Pa']:.1f} Па "
              f"({self.results['sigma_thermal_MPa']:.3f} МПа)")
