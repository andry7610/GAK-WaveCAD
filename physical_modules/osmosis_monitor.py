"""
Osmosis Monitor — расчёт осмотического давления в сферической оболочке.

Модель:
  Внутри оболочки — раствор с концентрацией c_solute.
  Снаружи — чистый растворитель (c = 0).
  Осмотическое давление: Π = c * R_gas * T  (уравнение Вант-Гоффа).

  Внутреннее напряжение в оболочке:
    σ = Π * R / (2 * h)
  где R — радиус, h — толщина стенки.

История:
  v0.1 — базовый расчёт по Вант-Гоффу
  v0.2 — наследование от BaseModule, регистрация
"""

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry


@ModuleRegistry.register("osmosis_monitor")
class OsmosisMonitor(BaseModule):
    """Расчёт осмотического давления и напряжения в оболочке.

    Параметры:
      c_solute — концентрация растворённого вещества (моль/м³)
      T        — температура (К)
      R_gas    — универсальная газовая постоянная (Дж/(моль·К))
      R_shell  — радиус оболочки (м)
      h_wall   — толщина стенки оболочки (м)
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}
        self.c_solute = cfg.get("c_solute", kwargs.get("c_solute", 300.0))
        self.T = cfg.get("T", kwargs.get("T", 310.0))
        self.R_gas = cfg.get("R_gas", kwargs.get("R_gas", 8.314))
        self.R_shell = cfg.get("R_shell", kwargs.get("R_shell", 0.05))
        self.h_wall = cfg.get("h_wall", kwargs.get("h_wall", 0.002))

        # Результаты
        self.results = {}

    # --- Реализация интерфейса BaseModule ---

    def init(self) -> bool:
        if self.T <= 0:
            raise ValueError("Temperature must be positive")
        if self.R_shell <= 0:
            raise ValueError("R_shell must be positive")
        if self.h_wall <= 0:
            raise ValueError("h_wall must be positive")
        if self.c_solute < 0:
            raise ValueError("c_solute cannot be negative")
        self._set_initialized(True)
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        # Осмотическое давление (Вант-Гофф)
        pi = self.c_solute * self.R_gas * self.T

        # Напряжение в стенке сферической оболочки
        sigma = pi * self.R_shell / (2 * self.h_wall)

        self.results = {
            'pi_Pa': pi,
            'pi_kPa': pi / 1e3,
            'sigma_Pa': sigma,
            'sigma_MPa': sigma / 1e6,
            'c_solute': self.c_solute,
            'T': self.T,
        }
        return True

    def get_results(self) -> dict:
        return self.results

    # --- Дополнительно ---

    def get_stress_Pa(self) -> float:
        """Возвращает напряжение в Па — для передачи в AcousticMonitor."""
        return self.results.get('sigma_Pa', 0.0)

    def print_report(self):
        r = self.results
        print(f"  Осмотическое давление: {r['pi_kPa']:.2f} кПа ({r['pi_Pa']:.1f} Па)")
        print(f"  Напряжение в стенке:   {r['sigma_MPa']:.4f} МПа ({r['sigma_Pa']:.1f} Па)")
        print(f"  Концентрация: {r['c_solute']:.1f} моль/м³  |  T = {r['T']:.1f} К")
