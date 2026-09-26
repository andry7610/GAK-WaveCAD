"""
EM Resonance Monitor — электромагнитные моды в сферической полости.

Модель:
  Собственные частоты TM- и TE-мод сферического резонатора.
  Связь с акустикой через пьезоэффект: механическое напряжение
  генерирует электрическое поле, сдвигающее EM-частоты.

История:
  v0.1 — базовая реализация
  v0.2 — TM/TE корни исправлены, float() на параметрах, Q=1/tan_delta,
         n_modes из конфига, динамический поиск корней через spherical_jn
"""

import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("em_resonance_monitor")
class EMResonanceMonitor(BaseModule):
    """Анализ электромагнитных мод сферического резонатора."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}

        # float() на всех параметрах + fallback R <- R_shell
        self.R = float(cfg.get("R", kwargs.get("R",
                   cfg.get("R_shell", kwargs.get("R_shell", 0.05)))))
        self.eps_r = float(cfg.get("eps_r", kwargs.get("eps_r", 10.0)))
        self.mu_r = float(cfg.get("mu_r", kwargs.get("mu_r", 1.0)))
        self.d_piezo = float(cfg.get("d_piezo", kwargs.get("d_piezo", 220e-12)))
        self.eps_33 = float(cfg.get("eps_33", kwargs.get("eps_33", 8.85e-12 * 10)))
        # Тангенс угла потерь диэлектрика (для добротности)
        self.tan_delta = float(cfg.get("tan_delta", kwargs.get("tan_delta", 1e-4)))
        # Количество мод (l = 1 .. n_modes), по 2 на каждое l (TM + TE)
        self.n_modes = int(cfg.get("n_modes", kwargs.get("n_modes", 2)))

        self.mechanical_stress = 0.0

        self.logger = get_logger("em_resonance_monitor")

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.eps_r <= 0:
            raise ValueError("eps_r must be positive")
        if self.tan_delta <= 0:
            raise ValueError("tan_delta must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, eps_r={self.eps_r:.1f}, "
                         f"tan_delta={self.tan_delta:.1e}, n_modes={self.n_modes}")
        return True

    def _find_tm_root(self, l, n):
        """Найти n-й корень j_l(x) = 0 (TM-моды).
        Граничное условие для TM: тангенциальное E = 0 → j_l(kR) = 0.
        """
        # Корни сферической функции Бесселя j_l(x) = 0
        # Табличные значения для проверки:
        table = {
            (1, 1): 4.4934, (1, 2): 7.7253,
            (2, 1): 5.7635, (2, 2): 9.0950,
            (3, 1): 6.9879, (3, 2): 10.4174,
        }
        key = (l, n)
        if key in table:
            return table[key]
        # Динамический поиск через brentq
        return self._brentq_root(lambda x: spherical_jn(l, x), l + n, l + n + 10)

    def _find_te_root(self, l, n):
        """Найти n-й корень [x * j_l(x)]' = 0 (TE-моды).
        Граничное условие для TE: тангенциальное H = 0 → производная от x*j_l(x) = 0.
        Эквивалентно: x*j_l'(x) + j_l(x) = 0.
        """
        # Корни производной x*j_l(x) = 0
        # Табличные значения для проверки:
        table = {
            (1, 1): 2.7437, (1, 2): 6.1168,
            (2, 1): 3.8702, (2, 2): 7.4435,
            (3, 1): 4.9735, (3, 2): 8.7225,
        }
        key = (l, n)
        if key in table:
            return table[key]
        # Динамический поиск
        def func(x):
            return x * spherical_jn(l, x, derivative=True) + spherical_jn(l, x)
        return self._brentq_root(func, l + n, l + n + 10)

    def _brentq_root(self, func, lo, hi, n_points=200):
        """Найти первый корень func в (lo, hi) методом brentq."""
        xs = np.linspace(lo, hi, n_points)
        vals = [func(x) for x in xs]
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                return brentq(func, xs[i], xs[i + 1])
        return float(lo)

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        c = 3.0e8  # скорость света, м/с
        v_em = c / np.sqrt(self.eps_r * self.mu_r)

        # Пьезоэлектрический сдвиг
        E_piezo = self.d_piezo * self.mechanical_stress / self.eps_33
        k_piezo = self.d_piezo * self.mechanical_stress / (self.eps_33 * self.eps_r)

        # Добротность: Q = 1 / tan(delta) — стандартная формула для диэлектрика
        Q = 1.0 / self.tan_delta

        self.logger.info(f"E_piezo={E_piezo:.3e} В/м, k_piezo={k_piezo:.3e}, Q={Q:.0f}")

        self.results = {}
        for l in range(1, self.n_modes + 1):
            for mtype in ("TM", "TE"):
                n = 1  # первая радиальная мода
                key = f"{mtype}_l{l}_n{n}"

                if mtype == "TM":
                    x_root = self._find_tm_root(l, n)
                else:
                    x_root = self._find_te_root(l, n)

                f_base = v_em * x_root / (2 * np.pi * self.R)
                f_shifted = f_base * (1 + k_piezo)
                df_stress = f_shifted - f_base

                self.results[key] = {
                    'mtype': mtype,
                    'l': l,
                    'n': n,
                    'x_root': x_root,
                    'f_base': f_base,
                    'f_shifted': f_shifted,
                    'df_stress': df_stress,
                    'E_piezo': E_piezo,
                    'k_piezo': k_piezo,
                    'Q': Q,
                }

        self.logger.info(f"Анализ завершён: {len(self.results)} мод")
        return True

    def get_results(self) -> dict:
        return self.results

    def set_mechanical_stress(self, stress_Pa: float):
        self.mechanical_stress = stress_Pa
        self.logger.info(f"Установлено напряжение: {stress_Pa:.1f} Па")

    def print_report(self):
        print("\n  EM Resonance Monitor — отчёт:")
        for key, r in self.results.items():
            print(f"    {key:12s} : f={r['f_shifted']:.3e} Гц, "
                  f"df_stress={r['df_stress']:+.3e} Гц, "
                  f"Q={r['Q']:.0f}, E_piezo={r['E_piezo']:.2e} В/м")
