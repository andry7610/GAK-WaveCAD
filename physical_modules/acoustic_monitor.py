import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════════════════════════════
#  acoustic_monitor.py
#  GAK-Tornado Project — Acoustic Spectrum Monitor v1.0
#  4-mode acoustic resonance monitor for 2-meter sphere
#  Material: heavy_compound (polymer-cement composite)
#
#  Authored by: Андрей Н. (chief engineer), Alice AI (metrology & validation)
#  Date: 2026-09-19
#  License: MIT
# ═══════════════════════════════════════════════════════════════════════════

# ─── CONFIG ─────────────────────────────────────────────────────────────────

CONFIG = {
    "acoustic_core": {
        "geometry": {
            "shape": "sphere",
            "radius": 1.0
        },
        "material_presets": {
            "heavy_compound": {
                "density": 2400.0,
                "youngs_modulus": 32.0e9,
                "poisson_ratio": 0.20,
                "c_L": 3850.0,
                "c_T": 2350.0,
                "damping_factor": 0.02
            }
        },
        "monitor_scan_matrix": {
            "channels": [
                {"mode": "shear_l2",        "f_min": 1600, "f_max": 1720,
                 "description": "Квадруполь (l=2) — основной индикатор напряжений"},
                {"mode": "radial_l0",      "f_min": 1720, "f_max": 1850,
                 "description": "Радиальное дыхание (l=0)"},
                {"mode": "shear_l3",       "f_min": 2480, "f_max": 2700,
                 "description": "Октуполь (l=3)"},
                {"mode": "shear_l4",       "f_min": 3300, "f_max": 3600,
                 "description": "Гексадекаполь (l=4)"}
            ]
        },
        "analysis_settings": {
            "frequency_resolution": 0.5,
            "splitting_threshold": 1.0,
            "max_split_window": 20.0,
            "peak_prominence": 0.15
        },
        "reference_modes": [
            {"n": 1, "l": 0, "type": "radial", "root": 2.89, "c": 3850, "f_hz": 1771},
            {"n": 1, "l": 2, "type": "shear",  "root": 4.45, "c": 2350, "f_hz": 1664},
            {"n": 1, "l": 3, "type": "shear",  "root": 6.90, "c": 2350, "f_hz": 2581},
            {"n": 1, "l": 4, "type": "shear",  "root": 9.20, "c": 2350, "f_hz": 3441}
        ]
    }
}


# ─── CORE FUNCTIONS ──────────────────────────────────────────────────────────

def calculate_tension(delta_f, f0, youngs_modulus, k=1.0):
    """
    Рассчитывает внутреннее напряжение через относительное расщепление моды.

    sigma ≈ k * E * (Δf / f0)

    Parameters
    ----------
    delta_f : float   — расщепление частоты (Гц)
    f0 : float         — центральная частота моды (Гц)
    youngs_modulus : float — модуль Юнга (Па)
    k : float         — калибровочный коэффициент (безразмерный)

    Returns
    -------
    float — напряжение в Па
    """
    if f0 is None or f0 == 0:
        return 0.0
    return k * youngs_modulus * (delta_f / f0)


def free_energy(results):
    """
    Жёсткостная часть свободной энергии по зарегистрированным пикам:

    U = Σ (A_i² × f_i²)

    Parameters
    ----------
    results : dict — вывод analyze_acoustic_spectrum()

    Returns
    -------
    float — суммарная энергия (усл. ед.)
    """
    total = 0.0
    for mode, data in results.items():
        pf = data.get("peaks_found")
        if pf is None:
            continue
        for amp, freq in zip(pf["amplitudes"], pf["frequencies"]):
            total += (amp ** 2) * (freq ** 2)
    return total


def _parabolic_interp(amps, peak_idx):
    """
    Параболическая интерполяция для уточнения позиции пика между узлами FFT.

    Возвращает смещение (в долях шага сетки) от пикового узла.
    """
    if peak_idx <= 0 or peak_idx >= len(amps) - 1:
        return 0.0
    a0 = amps[peak_idx - 1]
    a1 = amps[peak_idx]
    a2 = amps[peak_idx + 1]
    denom = (a0 - 2 * a1 + a2)
    if denom == 0:
        return 0.0
    return 0.5 * (a0 - a2) / denom


def _find_best_split_pair(peaks, amps, freqs, max_window):
    """
    Ищет пару ближайших пиков (в пределах max_window) с максимальной
    суммарной амплитудой. Возвращает (idx_left, idx_right) или None.
    """
    if len(peaks) < 2:
        return None

    best_pair = None
    best_score = -1.0

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            df = abs(freqs[peaks[j]] - freqs[peaks[i]])
            if df > max_window or df < 0.01:
                continue
            score = amps[peaks[i]] + amps[peaks[j]]
            if score > best_score:
                best_score = score
                best_pair = (peaks[i], peaks[j])

    return best_pair


def analyze_acoustic_spectrum(fft_data, frequencies, config):
    """
    Анализирует спектр по каналам (модам), ищет расщепление пиков,
    считает внутреннее напряжение и свободную энергию.

    Parameters
    ----------
    fft_data : np.ndarray — амплитуды FFT
    frequencies : np.ndarray — частотная сетка (Гц)
    config : dict — конфигурация (см. CONFIG выше)

    Returns
    -------
    dict — результаты по каждому каналу
    """
    ac = config['acoustic_core']
    channels = ac['monitor_scan_matrix']['channels']
    settings = ac['analysis_settings']
    E = ac['material_presets']['heavy_compound']['youngs_modulus']

    threshold = settings['splitting_threshold']
    max_window = settings['max_split_window']
    prominence = settings['peak_prominence']

    results = {}

    for ch in channels:
        f_min = ch['f_min']
        f_max = ch['f_max']
        mode_name = ch['mode']

        mask = (frequencies >= f_min) & (frequencies <= f_max)
        target_freqs = frequencies[mask]
        target_amps = fft_data[mask]

        if len(target_amps) == 0:
            results[mode_name] = {
                "status": "NO_DATA",
                "delta_f": 0.0,
                "f0": None,
                "tension_Pa": 0.0,
                "peaks_found": None
            }
            continue

        peaks, props = find_peaks(target_amps, prominence=prominence)

        # Параболическая интерполяция для уточнения частот пиков
        refined_freqs = []
        refined_amps = []
        for p in peaks:
            offset = _parabolic_interp(target_amps, p)
            refined_f = target_freqs[p] + offset * (target_freqs[1] - target_freqs[0])
            refined_freqs.append(refined_f)
            refined_amps.append(target_amps[p])

        refined_freqs = np.array(refined_freqs)
        refined_amps = np.array(refined_amps)

        peaks_found = {
            "frequencies": refined_freqs.tolist(),
            "amplitudes": refined_amps.tolist()
        }

        if len(peaks) >= 2:
            pair = _find_best_split_pair(
                np.arange(len(refined_freqs)),
                refined_amps,
                refined_freqs,
                max_window
            )

            if pair is not None:
                idx_l, idx_r = pair
                f_left = refined_freqs[idx_l]
                f_right = refined_freqs[idx_r]
                delta_f = abs(f_right - f_left)
                f0 = (f_left + f_right) / 2.0

                if delta_f > threshold:
                    status = "CRITICAL_STRESS"
                    tension = calculate_tension(delta_f, f0, E, k=1.0)
                else:
                    status = "STABLE"
                    tension = 0.0
            else:
                status = "FREE_OR_DAMPED"
                delta_f = 0.0
                tension = 0.0
                f0 = float(refined_freqs[np.argmax(refined_amps)]) if len(refined_freqs) > 0 else None
        else:
            status = "FREE_OR_DAMPED"
            delta_f = 0.0
            tension = 0.0
            f0 = float(refined_freqs[0]) if len(refined_freqs) == 1 else None

        results[mode_name] = {
            "status": status,
            "delta_f": delta_f,
            "f0": f0,
            "tension_Pa": tension,
            "peaks_found": peaks_found
        }

    return results


# ─── VALIDATOR (Synthetic test) ─────────────────────────────────────────────

def generate_synthetic_spectrum(config, splitting_hz=1.5, noise_level=0.02):
    """
    Создаёт искусственный FFT-спектр с 4 модами и заданным расщеплением
    в квадрупольной моде (shear_l2).
    """
    ac = config['acoustic_core']
    df = ac['analysis_settings']['frequency_resolution']

    frequencies = np.arange(0, 5000, df)
    fft_data = np.random.normal(0, noise_level, len(frequencies))

    ref = {m["type"] + f"_l{m['l']}": m["f_hz"] for m in ac['reference_modes']}

    # Канал 1: квадруполь с расщеплением
    f_center = ref["shear_l2"]
    f_left = f_center - splitting_hz / 2.0
    f_right = f_center + splitting_hz / 2.0
    sigma = 0.3
    fft_data += 0.8 * np.exp(-0.5 * ((frequencies - f_left) / sigma) ** 2)
    fft_data += 0.6 * np.exp(-0.5 * ((frequencies - f_right) / sigma) ** 2)

    # Канал 2: радиальная — чистый пик
    f_rad = ref["radial_l0"]
    fft_data += 0.9 * np.exp(-0.5 * ((frequencies - f_rad) / 0.3) ** 2)

    # Канал 3: октуполь — чистый пик, слабее (затухание)
    f_oct = ref["shear_l3"]
    fft_data += 0.5 * np.exp(-0.5 * ((frequencies - f_oct) / 0.5) ** 2)

    # Канал 4: гексадекаполь — ещё слабее
    f_hex = ref["shear_l4"]
    fft_data += 0.35 * np.exp(-0.5 * ((frequencies - f_hex) / 0.7) ** 2)

    return frequencies, fft_data, ref


# ─── PLOTTING ────────────────────────────────────────────────────────────────

def plot_spectrum(frequencies, fft_data, results, ref_freqs, save_path=None):
    """
    Рисует 4-панельный график спектра с отметкой пиков и расщеплений.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    channel_info = [
        ("shear_l2",        1600, 1720, "red",   "Квадруполь (l=2)"),
        ("radial_l0",       1720, 1850, "blue",  "Радиальное дыхание (l=0)"),
        ("shear_l3",        2480, 2700, "green", "Октуполь (l=3)"),
        ("shear_l4",        3300, 3600, "purple","Гексадекаполь (l=4)")
    ]

    for i, (mode, f_lo, f_hi, color, title) in enumerate(channel_info):
        ax = axes[i]
        mask = (frequencies >= f_lo) & (frequencies <= f_hi)
        ax.plot(frequencies[mask], fft_data[mask],
                linewidth=0.8, color='black', label='Спектр')

        data = results.get(mode, {})
        pf = data.get("peaks_found")

        if pf is not None:
            f_arr = np.array(pf["frequencies"])
            a_arr = np.array(pf["amplitudes"])
            ax.scatter(f_arr, a_arr, color=color, s=60, zorder=5,
                       label=f"Пики: {data['status']}")

            if data.get("delta_f", 0) > 0:
                f0 = data["f0"]
                sigma_mpa = data["tension_Pa"] / 1e6
                ax.annotate(
                    f"Δf = {data['delta_f']:.3f} Гц\nσ = {sigma_mpa:.2f} МПа",
                    xy=(f0, max(a_arr)),
                    xytext=(f0 + (f_hi - f_lo) * 0.15, max(a_arr) * 0.9),
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
                )

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Частота (Гц)', fontsize=9)
        ax.set_ylabel('Амплитуда', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('GAK-Tornado — Акустический спектр 2-м сферы (4 моды)\n'
                 'Синтетический тест: расщепление 1.5 Гц в квадрупольной моде',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")

    plt.show()


# ─── INTEGRATION TEST ────────────────────────────────────────────────────────

def run_integration_test(config, save_plot_path=None):
    """
    Полный интеграционный тест: генерация → анализ → проверка → график.
    """
    SPLITTING = 1.5
    E = config['acoustic_core']['material_presets']['heavy_compound']['youngs_modulus']

    print("=" * 65)
    print("  GAK-TORNADO — ИНТЕГРАЦИОННЫЙ ТЕСТ ACOUSTIC_MONITOR v1.0")
    print("=" * 65)

    # 1. Генерация
    freqs, fft, ref = generate_synthetic_spectrum(config, splitting_hz=SPLITTING)
    df_actual = freqs[1] - freqs[0]
    print(f"\n[1] Синтетический спектр: {len(freqs)} точек, Δf = {df_actual:.1f} Гц")
    print(f"    Заложено расщепление: {SPLITTING} Гц в моде shear_l2")
    print(f"    Опорные частоты: l2={ref['shear_l2']:.0f}, "
          f"l0={ref['radial_l0']:.0f}, "
          f"l3={ref['shear_l3']:.0f}, "
          f"l4={ref['shear_l4']:.0f} Гц")

    # 2. Анализ
    results = analyze_acoustic_spectrum(fft, freqs, config)
    print(f"\n[2] Анализ завершён. Результаты по 4 каналам:")
    print(f"    {'Мода':<14s} | {'Статус':<18s} | {'Δf':>7s} | {'σ (МПа)':>10s} | Пики")
    print(f"    {'-'*14}-+-{'-'*18}-+-{'-'*7}-+-{'-'*10}-+-{'-'*5}")

    for mode, data in results.items():
        n_peaks = len(data["peaks_found"]["frequencies"]) if data["peaks_found"] else 0
        sigma_str = f"{data['tension_Pa']/1e6:.2f}" if data["tension_Pa"] > 0 else "0.00"
        print(f"    {mode:<14s} | {data['status']:<18s} | "
              f"{data['delta_f']:>7.3f} | {sigma_str:>10s} | {n_peaks}")

    # 3. Проверка квадруполя
    quad = results.get('shear_l2', {})
    print(f"\n[3] Проверка квадрупольной моды (shear_l2):")

    if quad['status'] == 'CRITICAL_STRESS':
        measured_df = quad['delta_f']
        f0 = quad['f0']
        theoretical_sigma = 1.0 * E * (SPLITTING / f0) if f0 else 0
        measured_sigma = quad['tension_Pa']
        error_pct = (abs(measured_sigma - theoretical_sigma)
                     / theoretical_sigma * 100 if theoretical_sigma > 0 else 0)

        print(f"    Заложенное расщепление:  {SPLITTING:.3f} Гц")
        print(f"    Измеренное расщепление:  {measured_df:.3f} Гц")
        print(f"    Центральная частота f0:  {f0:.3f} Гц")
        print(f"    Теоретическое σ:         {theoretical_sigma/1e6:.2f} МПа")
        print(f"    Измеренное σ:            {measured_sigma/1e6:.2f} МПа")
        print(f"    Погрешность:             {error_pct:.2f}%")

        if error_pct < 5.0:
            print(f"\n    ✅ ТЕСТ ПРОЙДЕН — погрешность < 5%")
        else:
            print(f"\n    ❌ ТЕСТ ПРОВАЛЕН — погрешность > 5%")
    else:
        print(f"    Статус: {quad['status']}")
        print(f"    ❌ ТЕСТ ПРОВАЛЕН — расщепление {SPLITTING} Гц не поймано")

    # 4. Свободная энергия
    U = free_energy(results)
    print(f"\n[4] Свободная энергия (жёсткостная часть): U = {U:.2f} усл. ед.")

    # 5. График
    print(f"\n[5] Отрисовка спектра...")
    plot_spectrum(freqs, fft, results, ref, save_path=save_plot_path)

    print(f"\n{'='*65}")
    print(f"  ТЕСТ ЗАВЕРШЁН")
    print(f"{'='*65}")

    return results


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    results = run_integration_test(
        CONFIG,
        save_plot_path="gak_acoustic_spectrum_4modes.png"
    )
