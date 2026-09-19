"""
Acoustic Monitor — анализатор акустических мод сферической оболочки.

Моды:
  radial_l0 — радиальная (l=0)
  shear_l2  — сдвиговая квадрупольная (l=2)
  shear_l3  — сдвиговая (l=3)
  shear_l4  — сдвиговая (l=4)

История:
  v0.1 — FFT-анализ, 4 моды, расщепление
  v0.2 — генерация временного сигнала s(t)
  v0.3 — корни Бесселя исправлены, разрешение FFT улучшено
  v0.4 — set_internal_stress() + analyze_all_modes() для связи с осмосом
  v0.5 — наследование от BaseModule, регистрация в ModuleRegistry
"""

import numpy as np
from core.base_module import BaseModule
from core.module_registry import ModuleRegistry


@ModuleRegistry.register("acoustic_monitor")
class AcousticMonitor(BaseModule):
    """Анализатор акустических мод сферической оболочки.

    Параметры:
      R     — радиус оболочки (м)
      rho   — плотность (кг/м³)
      E     — модуль Юнга (Па)
      nu    — коэффициент Пуассона
      n_modes — количество мод (до 4)
      duration — длительность сигнала (с)
      fs    — частота дискретизации (Гц)
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        # Параметры из конфига, с откатом на значения по умолчанию
        cfg = config or {}
        self.R = cfg.get("R", kwargs.get("R", 0.05))
        self.rho = cfg.get("rho", kwargs.get("rho", 1000.0))
        self.E = cfg.get("E", kwargs.get("E", 2.0e9))
        self.nu = cfg.get("nu", kwargs.get("nu", 0.33))
        self.n_modes = cfg.get("n_modes", kwargs.get("n_modes", 4))
        self.duration = cfg.get("duration", kwargs.get("duration", 8.0))
        self.fs = cfg.get("fs", kwargs.get("fs", 20000))

        self.N = int(self.duration * self.fs)
        self.t = np.linspace(0, self.duration, self.N, endpoint=False)
        self.df = 1.0 / self.duration

        # Упругие константы (Ламе)
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

        # Внутреннее напряжение (от осмоса)
        self.internal_stress = 0.0

        # Моды: (l, тип)
        self.modes = [
            (0, 'radial'),
            (2, 'shear'),
            (3, 'shear'),
            (4, 'shear'),
        ][:self.n_modes]

        # Корни сферических функций Бесселя
        self.bessel_roots = {
            (0, 'radial'): 3.14159,   # pi
            (2, 'shear'):   2.0816,   # первый корень j_2
            (3, 'shear'):   3.3420,   # первый корень j_3
            (4, 'shear'):   4.4934,   # первый корень j_4
        }

        # Базовые частоты
        self.f0 = {}
        for (l, mtype) in self.modes:
            k = self.bessel_roots[(l, mtype)]
            if mtype == 'radial':
                v = np.sqrt((self.lam + 2 * self.mu) / self.rho)
            else:
                v = np.sqrt(self.mu / self.rho)
            self.f0[(l, mtype)] = v * k / (2 * np.pi * self.R)

        # Результаты последнего анализа
        self.results = {}

    # --- Реализация интерфейса BaseModule ---

    def init(self) -> bool:
        """Проверка целостности параметров перед запуском."""
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.E <= 0:
            raise ValueError("E must be positive")
        if self.n_modes < 1 or self.n_modes > 4:
            raise ValueError("n_modes must be between 1 and 4")
        self._set_initialized(True)
        return True

    def run(self) -> bool:
        """Основной расчёт — анализ всех мод."""
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")
        self.analyze_all_modes()
        return True

    def get_results(self) -> dict:
        """Возвращает словарь с результатами анализа."""
        return self.results

    # --- Существующая физика (без изменений) ---

    def set_internal_stress(self, sigma_Pa):
        """Установить внутреннее напряжение (Па) — от осмотического давления."""
        self.internal_stress = sigma_Pa

    def _shift_factor(self, l, mtype):
        """Относительный сдвиг частоты от внутреннего напряжения."""
        if abs(self.internal_stress) < 1e-20:
            return 0.0
        sigma = self.internal_stress
        if mtype == 'radial':
            factor = sigma / (self.lam + 2 * self.mu)
        else:
            factor = sigma / (2 * self.mu)
        sensitivity = 1.0 + 0.1 * (l - 1)
        return -sensitivity * factor

    def generate_signal(self, f_base, shift=0.0):
        """Сгенерировать временной сигнал для моды с частотой f_base*(1+shift)."""
        f = f_base * (1.0 + shift)
        tau = self.duration * 0.7
        signal = np.sin(2 * np.pi * f * self.t) * np.exp(-self.t / tau)
        signal += 0.01 * np.random.randn(self.N)
        return signal

    def analyze_mode(self, l, mtype):
        """Анализ одной моды: FFT, поиск пиков, оценка расщепления."""
        f_base = self.f0[(l, mtype)]
        shift = self._shift_factor(l, mtype)
        signal = self.generate_signal(f_base, shift)

        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(self.N, 1.0 / self.fs)
        mag = np.abs(spectrum)

        threshold = 0.1 * np.max(mag)
        peaks = []
        for i in range(1, len(mag) - 1):
            if mag[i] > threshold and mag[i] > mag[i-1] and mag[i] > mag[i+1]:
                peaks.append((freqs[i], mag[i]))

        peaks.sort(key=lambda x: -x[1])
        n_peaks = len(peaks)

        if n_peaks > 0:
            f_main = peaks[0][0]
        else:
            f_main = f_base

        if n_peaks >= 2:
            f_sorted = sorted([p[0] for p in peaks[:4]])
            delta_f = f_sorted[-1] - f_sorted[0]
        else:
            delta_f = 0.0

        sigma_MPa = self.internal_stress / 1e6

        return {
            'l': l,
            'type': mtype,
            'f_base': f_base,
            'f_measured': f_main,
            'shift': shift,
            'delta_f': delta_f,
            'sigma_MPa': sigma_MPa,
            'n_peaks': n_peaks,
            'status': 'CRITICAL_STRESS' if abs(shift) > 1e-5 else 'FREE_OR_DAMPED'
        }

    def analyze_all_modes(self):
        """Анализ всех мод."""
        self.results = {}
        for (l, mtype) in self.modes:
            key = f"{mtype}_l{l}"
            self.results[key] = self.analyze_mode(l, mtype)
        return self.results

    def print_report(self):
        """Вывод отчёта."""
        for key, r in self.results.items():
            print(f"  {key:12s} | {r['status']:16s} | "
                  f"f = {r['f_measured']:.6f} Гц | "
                  f"Δf = {r['delta_f']:.6f} Гц | "
                  f"σ = {r['sigma_MPa']:.4f} МПа | "
                  f"{r['n_peaks']} пик(ов)")
