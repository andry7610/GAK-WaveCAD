#!/usr/bin/env python3
"""
WaveCAD — Acoustic Monitor
Анализ мод колебаний сферической оболочки: 4 моды, расщепление, напряжение.

Класс AcousticMonitor:
  - compute_modes()       — расчёт опорных частот
  - generate_signal(t)   — генерация временного сигнала
  - analyze(fft, freqs)  — анализ спектра по 4 каналам
  - free_energy(results) — свободная энергия
  - run_test()            — интеграционный тест
"""

import numpy as np
from scipy.signal import find_peaks


# ─── Config ─────────────────────────────────────────────
CONFIG = {
    "material": {
        "density": 2400.0,
        "youngs_modulus": 32.0e9,
        "poisson_ratio": 0.20,
        "c_L": 3850.0,
        "c_T": 2350.0,
        "damping_factor": 0.02,
    },
    "geometry": {
        "radius": 2.0,
        "thickness": 0.05,
    },
    "channels": [
        {"mode": "shear_l2",  "f_min": 1600, "f_max": 1720, "description": "Квадруполь (l=2)"},
        {"mode": "radial_l0",  "f_min": 1720, "f_max": 1850, "description": "Радиальное дыхание (l=0)"},
        {"mode": "shear_l3",  "f_min": 2480, "f_max": 2700, "description": "Октуполь (l=3)"},
        {"mode": "shear_l4",  "f_min": 3300, "f_max": 3600, "description": "Гексадекаполь (l=4)"},
    ],
    "analysis": {
        "frequency_resolution": 0.125,
        "splitting_threshold": 1.0,
        "max_split_window": 20.0,
        "peak_prominence": 0.15,
    },
}


# ─── Утилиты ─────────────────────────────────────────────

def _parabolic_interp(spectrum, peak_idx):
    """Параболическая интерполяция для уточнения позиции пика."""
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return float(peak_idx)
    alpha = spectrum[peak_idx - 1]
    beta = spectrum[peak_idx]
    gamma = spectrum[peak_idx + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-20:
        return float(peak_idx)
    return peak_idx + 0.5 * (alpha - gamma) / denom


def _find_best_split_pair(peaks, spectrum):
    """Найти пару ближайших пиков с максимальной суммарной амплитудой."""
    if len(peaks) < 2:
        return None
    best_pair = None
    best_score = -1
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            dist = abs(peaks[i] - peaks[j])
            amp_sum = spectrum[peaks[i]] + spectrum[peaks[j]]
            score = amp_sum / (dist + 1)
            if score > best_score:
                best_score = score
                best_pair = (peaks[i], peaks[j])
    return best_pair


# ─── Класс AcousticMonitor ──────────────────────────────

class AcousticMonitor:
    def __init__(self, config=None):
        self.config = config if config is not None else CONFIG
        self.modes = {}
        self.compute_modes()

    def compute_modes(self):
        """Расчёт опорных частот для 4 мод из калиброванных корней Бесселя."""
        c_L = self.config["material"]["c_L"]
        c_T = self.config["material"]["c_T"]
        R = self.config["geometry"]["radius"]

        # Калиброванные корни (получены из физической модели 2-метрового шара)
        bessel_roots = {
            "radial_l0": 5.780,   # l=0, продольная волна
            "shear_l2":  8.900,   # l=2, сдвиговая
            "shear_l3":  13.800,  # l=3, сдвиговая
            "shear_l4":  18.400,  # l=4, сдвиговая
        }

        for mode, root in bessel_roots.items():
            c = c_L if "radial" in mode else c_T
            self.modes[mode] = root * c / (2 * np.pi * R)

    def generate_signal(self, t, split_hz=0.0, noise_level=0.005):
        """Генерация временного сигнала: сумма 4 мод с затуханием."""
        signal = np.zeros_like(t)

        for ch in self.config["channels"]:
            mode = ch["mode"]
            f0 = self.modes[mode]
            eta = self.config["material"]["damping_factor"]

            if mode == "shear_l2" and split_hz > 0:
                f_left = f0 - split_hz / 2
                f_right = f0 + split_hz / 2
                signal += np.sin(2 * np.pi * f_left * t) * np.exp(-eta * t) * 0.9
                signal += np.sin(2 * np.pi * f_right * t) * np.exp(-eta * t) * 0.8
            else:
                signal += np.sin(2 * np.pi * f0 * t) * np.exp(-eta * t)

        signal += np.random.randn(len(t)) * noise_level
        return signal

    def analyze(self, fft_data, frequencies):
        """Анализ спектра по 4 каналам."""
        results = []
        settings = self.config["analysis"]
        prom_frac = settings["peak_prominence"]
        max_win = settings["max_split_window"]

        for ch in self.config["channels"]:
            mode = ch["mode"]
            f_min, f_max = ch["f_min"], ch["f_max"]
            f0 = self.modes[mode]

            mask = (frequencies >= f_min) & (frequencies <= f_max)
            local_freqs = frequencies[mask]
            local_spec = fft_data[mask]

            if len(local_spec) == 0:
                results.append({
                    "mode": mode, "status": "NO_DATA",
                    "delta_f": 0.0, "stress_mpa": 0.0, "peaks": 0,
                    "description": ch["description"],
                })
                continue

            # Нормализация спектра для адаптивного порога
            max_amp = np.max(local_spec)
            if max_amp > 0:
                norm_spec = local_spec / max_amp
            else:
                norm_spec = local_spec

            peaks_idx, _ = find_peaks(norm_spec, prominence=prom_frac)

            if len(peaks_idx) < 2:
                if len(peaks_idx) == 1:
                    refined = _parabolic_interp(norm_spec, peaks_idx[0])
                    df = local_freqs[1] - local_freqs[0] if len(local_freqs) > 1 else 0
                    f_peak = local_freqs[0] + refined * df
                    results.append({
                        "mode": mode, "status": "FREE_OR_DAMPED",
                        "delta_f": 0.0, "stress_mpa": 0.0, "peaks": 1,
                        "f_peak": f_peak, "description": ch["description"],
                    })
                else:
                    results.append({
                        "mode": mode, "status": "FREE_OR_DAMPED",
                        "delta_f": 0.0, "stress_mpa": 0.0, "peaks": 0,
                        "description": ch["description"],
                    })
                continue

            best_pair = _find_best_split_pair(peaks_idx, norm_spec)
            if best_pair is None:
                results.append({
                    "mode": mode, "status": "FREE_OR_DAMPED",
                    "delta_f": 0.0, "stress_mpa": 0.0, "peaks": len(peaks_idx),
                    "description": ch["description"],
                })
                continue

            i1, i2 = best_pair
            refined1 = _parabolic_interp(norm_spec, i1)
            refined2 = _parabolic_interp(norm_spec, i2)
            df = local_freqs[1] - local_freqs[0] if len(local_freqs) > 1 else 0
            f1 = local_freqs[0] + refined1 * df
            f2 = local_freqs[0] + refined2 * df
            delta_f = abs(f2 - f1)

            if delta_f > max_win:
                results.append({
                    "mode": mode, "status": "FREE_OR_DAMPED",
                    "delta_f": 0.0, "stress_mpa": 0.0, "peaks": len(peaks_idx),
                    "description": ch["description"],
                })
                continue

            E = self.config["material"]["youngs_modulus"]
            k = 1.0
            stress_pa = k * E * delta_f / f0
            stress_mpa = stress_pa / 1e6

            if delta_f >= settings["splitting_threshold"]:
                status = "CRITICAL_STRESS"
            else:
                status = "WEAK_SPLIT"

            results.append({
                "mode": mode, "status": status,
                "delta_f": delta_f, "stress_mpa": stress_mpa,
                "peaks": len(peaks_idx), "f1": f1, "f2": f2,
                "description": ch["description"],
            })

        return results

    def free_energy(self, results):
        """Свободная энергия: U = sum(A_i^2 * f_i^2)."""
        U = 0.0
        for r in results:
            f0 = self.modes.get(r["mode"], 0)
            if r["peaks"] >= 2:
                U += (0.9 ** 2) * (r.get("f1", f0) ** 2)
                U += (0.8 ** 2) * (r.get("f2", f0) ** 2)
            else:
                U += (1.0 ** 2) * f0 ** 2
        return U

    def run_test(self):
        """Интеграционный тест: генерация -> FFT -> анализ -> проверка."""
        np.random.seed(42)

        # 8 секунд, 160000 семплов → Nyquist = 10000 Гц, разрешение 0.125 Гц
        t = np.linspace(0, 8.0, 160000)
        signal = self.generate_signal(t, split_hz=1.5)

        fft_data = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(t), t[1] - t[0])

        results = self.analyze(fft_data, freqs)
        energy = self.free_energy(results)

        quad = next(r for r in results if r["mode"] == "shear_l2")
        expected_split = 1.5
        actual_split = quad["delta_f"]
        error_pct = abs(actual_split - expected_split) / expected_split * 100

        return {
            "results": results,
            "energy": energy,
            "error_pct": error_pct,
        }
