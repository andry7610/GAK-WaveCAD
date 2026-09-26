"""
Osmosis Monitor — расчёт осмотического давления в сферической оболочке.

Модель:
  Внутри оболочки — раствор с концентрацией c_solute.
  Снаружи — чистый растворитель (c = 0).
  Осмотическое давление: Пи = c * R_gas * T  (уравнение Вант-Гоффа).

  Внутреннее напряжение в оболочке:
    sigma = Пи * R / (2 * h)
  где R — радиус, h — толщина стенки.

История:
  v0.1 — базовый расчёт по Вант-Гоффу
  v0.2 — наследование от BaseModule, регистрация
  v0.3 — логирование
  v0.4 — float() на всех параметрах, fallback R_shell/R и h_wall/h
"""

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("osmosis_monitor")
class OsmosisMonitor(BaseModule):
    """Расчёт осмотического давления и напряжения в оболочке."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}

        self.c_solute = float(cfg.get("c_solute", kwargs.get("c_solute", 300.0)))
        self.T = float(cfg.get("T", kwargs.get("T", 310.0)))
        self.R_gas = float(cfg.get("R_gas", kwargs.get("R_gas", 8.314)))

        # fallback: R_shell -> R -> дефолт 0.05
        R_val = cfg.get("R_shell", kwargs.get("R_shell", None))
        if R_val is None:
            R_val = cfg.get("R", kwargs.get("R", 0.05))
        self.R_shell = float(R_val)

        # fallback: h_wall -> h -> дефолт 0.002
        h_val = cfg.get("h_wall", kwargs.get("h_wall", None))
        if h_val is None:
            h_val = cfg.get("h", kwargs.get("h", 0.002))
        self.h_wall = float(h_val)

        self.logger = get_logger("osmosis_monitor")
        self.results = {}

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
        self.logger.info("Инициализация завершена")
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        pi = self.c_solute * self.R_gas * self.T
        sigma = pi * self.R_shell / (2 * self.h_wall)

        self.logger.info(f"Расчёт: c={self.c_solute}, T={self.T}, "
                         f"Pi={pi:.1f} Па, sigma={sigma:.1f} Па")

        self.results = {
            'pi_Pa': pi,
            'pi_kPa': pi / 1e3,
            'sigma_Pa': sigma,
            'sigma_MPa': sigma / 1e6,
            'c_solute': self.c_solute,
            'T': self.T,
            'R_shell': self.R_shell,
            'h_wall': self.h_wall,
        }
        return True

    def get_results(self) -> dict:
        return self.results

    def get_stress_Pa(self) -> float:
        """Возвращает напряжение в Па — для передачи в AcousticMonitor."""
        return self.results.get('sigma_Pa', 0.0)

    def print_report(self):
        r = self.results
        print(f"  Осмотическое давление: {r['pi_kPa']:.2f} кПа ({r['pi_Pa']:.1f} Па)")
        print(f"  Напряжение в стенке:   {r['sigma_MPa']:.4f} МПа ({r['sigma_Pa']:.1f} Па)")
        print(f"  Концентрация: {r['c_solute']:.1f} моль/м^3  |  T = {r['T']:.1f} К")
        print(f"  R={r['R_shell']:.3f} м, h={r['h_wall']*1e3:.1f} мм")
