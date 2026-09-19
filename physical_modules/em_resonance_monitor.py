"""
EM Resonance Monitor — электромагнитные моды в сферической полости.

Модель:
  Собственные частоты TM- и TE-мод сферического резонатора.
  Связь с акустикой через пьезоэффект: механическое напряжение
  генерирует электрическое поле, сдвигающее EM-частоты.

История:
  v0.1 — базовая реализация
"""

import numpy as np
from scipy.special import spherical_jn

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("em_resonance_monitor")
class EMResonanceMonitor(BaseModule):
    """Анализ электромагнитных мод сферического резонатора."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}

        self.R = cfg.get("R", kwargs.get("R", 0.05))
        self.eps_r = cfg.get("eps_r", kwargs.get("eps_r", 10.0))      # диэлектрическая проницаемость
        self.mu_r = cfg.get("mu_r", kwargs.get("mu_r", 1.0))          # магнитная проницаемость
        self.d_piezo = cfg.get("d_piezo", kwargs.get("d_piezo", 220e-12))  # пьезоэлектрический коэффициент, м/В
        self.eps_33 = cfg.get("eps_33", kwargs.get("eps_33", 8.85e-12 * 10))  # пьезо-диэлектрик, Ф/м

        self.mechanical_stress = 0.0

        self.logger = get_logger("em_resonance_monitor")

        # Корни производной сферической функции Бесселя для TM-мод
        # и корни самой функции для TE-мод (первые 4 моды)
        self.modes = [
            ("TM", 1, 1),
            ("TE", 1, 1),
            ("TM", 2, 1),
            ("TE", 2, 1),
        ]

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.eps_r <= 0:
            raise ValueError("eps_r must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, eps_r={self.eps_r:.1f}")
        return True

    def _find_root(self, mtype, l, n):
        """Найти n-й корень для TM или TE моды."""
        # Приближённые корни (табличные значения для сферических функций Бесселя)
        # TM: корни производной j_l'(x) = 0
        # TE: корни j_l(x) = 0
        roots_tm = {
            (1, 1): 2.7437, (1, 2): 6.1168,
            (2, 1): 3.8702, (2, 2): 7.4435,
            (3, 1): 4.9735, (3, 2): 8.7225,
        }
        roots_te = {
            (1, 1): 4.4934, (1, 2): 7.7253,
            (2, 1): 5.7635, (2, 2): 9.0950,
            (3, 1): 6.9879, (3, 2): 10.4174,
        }

        if mtype == "TM":
            return roots_tm.get((l, n), float(l + n))
        else:
            return roots_te.get((l, n), float(l + n))

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        c = 3.0e8  # скорость света, м/с
        v_em = c / np.sqrt(self.eps_r * self.mu_r)

        # Пьезоэлектрический сдвиг: напряжение → электрическое поле → сдвиг частоты
        # E_piezo = d_piezo * sigma / eps_33
        E_piezo = self.d_piezo * self.mechanical_stress / self.eps_33

        # Относительный сдвиг частоты: df/f ~ E_piezo / E_scale
        # E_scale — характерное поле внутри резонатора
        # Используем безразмерный параметр: k_piezo = d_piezo * sigma / (eps_33 * eps_r)
        k_piezo = self.d_piezo * self.mechanical_stress / (self.eps_33 * self.eps_r)

        self.logger.info(f"E_piezo={E_piezo:.3e} В/м, k_piezo={k_piezo:.3e}")

        self.results = {}
        for (mtype, l, n) in self.modes:
            key = f"{mtype}_l{l}_n{n}"

            x_root = self._find_root(mtype, l, n)

            # Базовая частота: f = v * x_root / (2*pi*R)
            f_base = v_em * x_root / (2 * np.pi * self.R)

            # Сдвиг от пьезоэффекта: f = f_base * (1 + k_piezo)
            f_shifted = f_base * (1 + k_piezo)
            df_stress = f_shifted - f_base

            # Добротность (приближённо, для диэлектрического резонатора)
            Q = 1.0 / (2 * self.eps_r * 1e-4) if self.eps_r > 0 else float('inf')

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
