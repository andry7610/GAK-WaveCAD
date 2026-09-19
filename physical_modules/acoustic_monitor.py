"""
Acoustic Monitor — анализ акустических мод сферической оболочки.

Модель:
  Сферическая оболочка радиуса R с упругими свойствами (E, nu, rho).
  Собственные частоты мод (l, type) рассчитываются по теории оболочек.
  Внутреннее напряжение сдвигает частоты (акустопластический эффект).

История:
  v0.3 — наследование от BaseModule, регистрация
  v0.4 — логирование
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
        self.R = cfg.get("R", kwargs.get("R", 0.05))
        self.rho = cfg.get("rho", kwargs.get("rho", 1000.0))
        self.E = cfg.get("E", kwargs.get("E", 2.0e9))
        self.nu = cfg.get("nu", kwargs.get("nu", 0.33))
        self.n_modes = cfg.get("n_modes", kwargs.get("n_modes", 4))
        self.duration = cfg.get("duration", kwargs.get("duration", 8.0))
        self.fs = cfg.get("fs", kwargs.get("fs", 20000))

        self.internal_stress = 0.0

        self.logger = get_logger("acoustic_monitor")

        self.modes = [
            (0, "breathing"),
            (1, "flexural"),
            (2, "flexural"),
            (3, "flexural"),
        ]

        self._compute_reference_frequencies()

    def _compute_reference_frequencies(self):
        """Расчёт опорных частот для каждой моды."""
        self.f0 = {}
        h = self.R * 0.04
        c_l = np.sqrt(self.E / (self.rho * (1 - self.nu**2)))

        for (l, mtype) in self.modes:
            if mtype == "breathing":
                f = (1 / (2 * np.pi * self.R)) * np.sqrt(
                    self.E * (1 + self.nu) / (self.rho * (1 - self.nu))
                )
            else:
                lambda_l = l * (l + 1)
                f = (h / (2 * np.pi * self.R**2)) * c_l * np.sqrt(lambda_l)

            self.f0[(l, mtype)] = f

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.E <= 0:
            raise ValueError("E must be positive")
        self._set_initialized(True)
        self.logger.info("Инициализация завершена")
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
            f_shifted = f_ref * np.sqrt(1 + stress_ratio)

            t = np.linspace(0, self.duration, int(self.fs * self.duration))
            signal = 0.1 * np.sin(2 * np.pi * f_shifted * t)
            noise = 0.01 * np.random.randn(len(t))
            signal += noise

            spectrum = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), 1 / self.fs)

            peak_idx = np.argmax(spectrum)
            f_measured = freqs[peak_idx]

            threshold = 0.1 * np.max(spectrum)
            n_peaks = np.sum(spectrum > threshold)

            delta_f = f_measured - f_ref
            sigma_MPa = self.internal_stress / 1e6

            if abs(stress_ratio) > 1e-4:
                status = "CRITICAL_STRESS"
            else:
                status = "FREE_OR_DAMPED"

            self.results[key] = {
                'l': l,
                'mtype': mtype,
                'f_reference': f_ref,
                'f_measured': f_measured,
                'delta_f': delta_f,
                'sigma_MPa': sigma_MPa,
                'n_peaks': int(n_peaks),
                'status': status,
            }

        self.logger.info(f"Анализ завершён: {len(self.results)} мод")
        return True

    def get_results(self) -> dict:
        return self.results

    def set_internal_stress(self, stress_Pa: float):
        self.internal_stress = stress_Pa
        self.logger.info(f"Установлено напряжение: {stress_Pa:.1f} Па")
