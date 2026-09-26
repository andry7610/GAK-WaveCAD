"""
Acoustic Monitor — анализ акустических мод сферической оболочки.

Модель:
  Сферическая оболочка радиуса R с упругими свойствами (E, nu, rho).
  Собственные частоты мод (l, type) рассчитываются по теории оболочек.
  Внутреннее напряжение сдвигает частоты (акустопластический эффект).

История:
  v0.3 — наследование от BaseModule, регистрация
  v0.4 — логирование
  v0.5 — float() на параметрах, h из конфига, формула дыхательной моды (2 вместо 1+nu),
         guard от отрицательного корня, n_modes из конфига, FFT убран
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("acoustic_monitor")
class AcousticMonitor(BaseModule):
    """Анализ акустических мод сферической оболочки."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}
        self.R = float(cfg.get("R", kwargs.get("R", 0.05)))
        self.rho = float(cfg.get("rho", kwargs.get("rho", 1000.0)))
        self.E = float(cfg.get("E", kwargs.get("E", 2.0e9)))
        self.nu = float(cfg.get("nu", kwargs.get("nu", 0.33)))
        self.h = float(cfg.get("h", kwargs.get("h", self.R * 0.04)))
        self.n_modes = int(cfg.get("n_modes", kwargs.get("n_modes", 4)))
        self.duration = float(cfg.get("duration", kwargs.get("duration", 8.0)))
        self.fs = float(cfg.get("fs", kwargs.get("fs", 20000)))

        self.internal_stress = 0.0

        self.logger = get_logger("acoustic_monitor")

        # Динамическая генерация мод: l=0 (breathing) + l=1..n_modes-1 (flexural)
        self.modes = [(0, "breathing")]
        for l in range(1, self.n_modes):
            self.modes.append((l, "flexural"))

        self._compute_reference_frequencies()

    def _compute_reference_frequencies(self):
        """Расчёт опорных частот для каждой моды."""
        self.f0 = {}
        c_l = np.sqrt(self.E / (self.rho * (1 - self.nu**2)))

        for (l, mtype) in self.modes:
            if mtype == "breathing":
                # Дыхательная мода сферической оболочки:
                # f = (1 / 2πR) * sqrt(2E / (ρ(1-ν)))
                # (множитель 2, не (1+ν) — для сферической оболочки)
                f = (1 / (2 * np.pi * self.R)) * np.sqrt(
                    2 * self.E / (self.rho * (1 - self.nu))
                )
            else:
                lambda_l = l * (l + 1)
                f = (self.h / (2 * np.pi * self.R**2)) * c_l * np.sqrt(lambda_l)

            self.f0[(l, mtype)] = f

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.E <= 0:
            raise ValueError("E must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, h={self.h*1e3:.1f} мм, n_modes={self.n_modes}")
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        self.logger.info(f"Анализ мод: stress={self.internal_stress:.1f} Па")

        self.results = {}
        for (l, mtype) in self.modes:
            key = f"{mtype}_l{l}"
            f_ref = self.f0[(l, mtype)]

            stress_ratio = self.internal_stress / self.E
            arg = 1 + stress_ratio
            if arg <= 0:
                f_shifted = 0.0
                status = "INVALID_STRESS"
            else:
                f_shifted = f_ref * np.sqrt(arg)
                if abs(stress_ratio) > 1e-4:
                    status = "CRITICAL_STRESS"
                else:
                    status = "FREE_OR_DAMPED"

            f_measured = f_shifted
            delta_f = f_measured - f_ref
            sigma_MPa = self.internal_stress / 1e6

            self.results[key] = {
                'l': l,
                'mtype': mtype,
                'f_reference': float(f_ref),
                'f_measured': float(f_measured),
                'f_shifted': float(f_shifted),
                'delta_f': float(delta_f),
                'sigma_MPa': float(sigma_MPa),
                'n_peaks': 1,
                'status': status,
            }

        self.logger.info(f"Анализ завершён: {len(self.results)} мод")
        return True

    def get_results(self) -> dict:
        return self.results

    def set_internal_stress(self, stress_Pa: float):
        self.internal_stress = float(stress_Pa)
        self.logger.info(f"Установлено напряжение: {stress_Pa:.1f} Па")

    def print_report(self):
        print("\n  Acoustic Monitor — отчёт:")
        for key, r in self.results.items():
            print(f"    {key:16s} : f_ref={r['f_reference']:.3e} Гц, "
                  f"f_shift={r['f_shifted']:.3e} Гц, "
                  f"df={r['delta_f']:+.3e} Гц, "
                  f"status={r['status']}")
