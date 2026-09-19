"""
Magnon Monitor — анализ спиновых мод в сферической оболочке (YIG).

Модель:
  Уравнение Ландау-Лифшица-Гилберта для прецессии намагниченности.
  Собственные частоты: Kittel (l=0) + обменные моды (l=1,2,3).
  Магнитоупругая связь: механическое напряжение сдвигает частоты
    через магнитострикцию λ_s.

История:
  v0.1 — базовая реализация
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("magnon_monitor")
class MagnonMonitor(BaseModule):
    """Анализ спиновых (магнонных) мод сферической оболочки."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}

        # Геометрия
        self.R = cfg.get("R", kwargs.get("R", 0.05))

        # Параметры YIG
        self.M_s = cfg.get("M_s", kwargs.get("M_s", 140e3))       # намагниченность, А/м
        self.A_ex = cfg.get("A_ex", kwargs.get("A_ex", 3.5e-12))  # обмен, Дж/м
        self.K_anis = cfg.get("K_anis", kwargs.get("K_anis", 610.0))  # анизотропия, Дж/м³
        self.lambda_s = cfg.get("lambda_s", kwargs.get("lambda_s", -5.9e-6))  # магнитострикция
        self.gamma_bar = cfg.get("gamma_bar", kwargs.get("gamma_bar", 28e9))   # гиромагн., Гц/Т
        self.B_ext = cfg.get("B_ext", kwargs.get("B_ext", 0.1))   # внешнее поле, Тл
        self.alpha_damp = cfg.get("alpha_damp", kwargs.get("alpha_damp", 3e-4))  # затухание Гилберта

        self.mechanical_stress = 0.0
        self.n_modes = 4

        self.logger = get_logger("magnon_monitor")

        self.modes = [
            (0, "kittel"),
            (1, "exchange"),
            (2, "exchange"),
            (3, "exchange"),
        ]

    def init(self) -> bool:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.M_s <= 0:
            raise ValueError("M_s must be positive")
        if self.gamma_bar <= 0:
            raise ValueError("gamma_bar must be positive")
        self._set_initialized(True)
        self.logger.info(f"Инициализация: R={self.R:.3f} м, B_ext={self.B_ext:.3f} Тл")
        self.logger.info(f"M_s={self.M_s:.0f} А/м, λ_s={self.lambda_s:.2e}")
        return True

    def run(self) -> bool:
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        mu_0 = 4 * np.pi * 1e-7

        # Поле анизотропии
        B_anis = mu_0 * 2 * self.K_anis / self.M_s
        B_eff = self.B_ext + B_anis

        # Магнитоупругий сдвиг поля от механического напряжения
        # ΔB_me = 3*λ_s*σ / M_s
        B_me = 3 * self.lambda_s * self.mechanical_stress / self.M_s

        B_total = B_eff + B_me

        self.logger.info(f"B_eff={B_eff:.6f} Тл, B_me={B_me:.2e} Тл, B_total={B_total:.6f} Тл")

        self.results = {}
        for (l, mtype) in self.modes:
            key = f"{mtype}_l{l}"

            # Kittel: f = gamma_bar * B_total
            # Exchange: f = gamma_bar * B_total + gamma_bar * mu_0 * A_ex * l*(l+1) / (M_s * R^2)
            f_base = self.gamma_bar * B_total

            if l > 0:
                f_exchange = self.gamma_bar * mu_0 * self.A_ex * l * (l + 1) / (self.M_s * self.R**2)
                f = f_base + f_exchange
            else:
                f = f_base

            # Затухание Гилберта: ширина линии
                        delta_f = self.alpha_damp * 2 * f  # полная ширина (приближённо)

            # Добротность
            Q = f / delta_f if delta_f > 0 else float('inf')

            # Сдвиг от напряжения
            f_no_stress = self.gamma_bar * B_eff
            if l > 0:
                f_no_stress += self.gamma_bar * mu_0 * self.A_ex * l * (l + 1) / (self.M_s * self.R**2)
            df_stress = f - f_no_stress

            self.results[key] = {
                'l': l,
                'mtype': mtype,
                'f': f,
                'f_no_stress': f_no_stress,
                'df_stress': df_stress,
                'delta_f': delta_f,
                'Q': Q,
                'B_me': B_me,
            }

        self.logger.info(f"Анализ завершён: {len(self.results)} мод")
        return True

    def get_results(self) -> dict:
        return self.results

    def set_mechanical_stress(self, stress_Pa: float):
        self.mechanical_stress = stress_Pa
        self.logger.info(f"Установлено напряжение: {stress_Pa:.1f} Па")

    def print_report(self):
        print("\n  Magnon Monitor — отчёт:")
        for key, r in self.results.items():
            print(f"    {key:14s} : f={r['f']:.3e} Гц, "
                  f"df_stress={r['df_stress']:+.3e} Гц, "
                  f"Q={r['Q']:.0f}, dF={r['delta_f']:.2e} Гц")
