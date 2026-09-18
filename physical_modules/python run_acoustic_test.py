
#!/usr/bin/env python3
"""
run_acoustic_test.py — Точка входа для интеграционного теста
акустического модуля сферической оболочки.

Импортирует AcousticMonitor из physical_modules/acoustic_monitor.py,
генерирует синтетический сигнал с расщеплением 1.5 Гц в квадрупольной моде,
прогоняет анализ и выводит результаты + график.

Запуск:
    python run_acoustic_test.py
"""

import sys
import os
import numpy as np

# Добавляем корень репозитория в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    print("=" * 64)
    print("  WaveCAD — Acoustic Monitor Integration Test")
    print("  v0.4 — 4-mode spherical shell analyzer")
    print("=" * 64)

    # 1. Создаём монитор с конфигом по умолчанию
    monitor = AcousticMonitor()

    print(f"\n[1] Материал: heavy_compound")
    print(f"    Плотность:      {monitor.config['material_presets']['heavy_compound']['density']:.0f} кг/м³")
    print(f"    Модуль Юнга:    {monitor.config['material_presets']['heavy_compound']['youngs_modulus']/1e9:.1f} ГПа")
    print(f"    Коэф. Пуассона: {monitor.config['material_presets']['heavy_compound']['poisson_ratio']:.2f}")
    print(f"    c_L = {monitor.config['material_presets']['heavy_compound']['c_L']:.0f} м/с")
    print(f"    c_T = {monitor.config['material_presets']['heavy_compound']['c_T']:.0f} м/с")

    # 2. Опорные частоты
    print(f"\n[2] Опорные частоты (R = {monitor.config['geometry']['radius']:.1f} м):")
    print(f"    shear_l2 (квадруполь):     1664 Гц")
    print(f"    radial_l0 (дыхание):       1771 Гц")
    print(f"    shear_l3 (октуполь):       2581 Гц")
    print(f"    shear_l4 (гексадекаполь):  3441 Гц")

    # 3. Генерация временного сигнала
    SPLIT_HZ = 1.5  # закладываем расщепление в квадруполь
    fs = 10000       # частота дискретизации
    duration = 0.5   # секунд
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    print(f"\n[3] Генерация сигнала:")
    print(f"    Частота дискретизации: {fs} Гц")
    print(f"    Длительность:          {duration} с ({len(t)} точек)")
    print(f"    Заложено расщепление:  {SPLIT_HZ} Гц в моде shear_l2")

    signal = monitor.generate_signal(t, split_hz=SPLIT_HZ)

    # 4. FFT
    fft_data = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(t), 1.0 / fs)

    df = freqs[1] - freqs[0]
    print(f"\n[4] FFT: {len(freqs)} бинов, шаг {df:.2f} Гц")

    # 5. Анализ
    results = monitor.analyze(fft_data, freqs)

    print(f"\n[5] Результаты анализа:")
    print(f"    {'Мода':<20s} | {'Статус':<18s} | {'Δf':>8s} | {'σ (МПа)':>10s} | {'Пиков':>6s}")
    print(f"    {'-'*20}-+-{'-'*18}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for mode, data in results.items():
        sigma_mpa = data['tension_Pa'] / 1e6 if data['tension_Pa'] else 0.0
        n_peaks = len(data['peaks_found']['frequencies']) if data['peaks_found'] else 0
        print(f"    {mode:<20s} | {data['status']:<18s} | {data['delta_f']:8.3f} | {sigma_mpa:10.2f} | {n_peaks:6d}")

    # 6. Свободная энергия
    energy = monitor.free_energy(results)
    print(f"\n[6] Свободная энергия (жёсткостная часть): U = {energy:.2f}")

    # 7. Проверка погрешности
    quad = results.get('shear_l2', {})
    if quad['status'] == 'CRITICAL_STRESS' and quad['f0']:
        E = monitor.config['material_presets']['heavy_compound']['youngs_modulus']
        theoretical_sigma = 1.0 * E * (SPLIT_HZ / quad['f0'])
        measured_sigma = quad['tension_Pa']
        error_pct = abs(measured_sigma - theoretical_sigma) / theoretical_sigma * 100

        print(f"\n[7] Валидация квадрупольной моды:")
        print(f"    Теоретическое σ:  {theoretical_sigma/1e6:.2f} МПа")
        print(f"    Измеренное σ:     {measured_sigma/1e6:.2f} МПа")
        print(f"    Погрешность:      {error_pct:.2f}%")

        if error_pct < 5.0:
            print(f"\n    ✅ ТЕСТ ПРОЙДЕН — погрешность < 5%")
        else:
            print(f"\n    ❌ ТЕСТ ПРОВАЛЕН — погрешность > 5%")
    else:
        print(f"\n[7] Квадруполь не поймал расщепление — проверьте splitting_threshold")

    # 8. График
    print(f"\n[8] Отрисовка спектра...")
    monitor.plot_spectrum(freqs, fft_data, results)

    print("\n" + "=" * 64)
    print("  Тест завершён")
    print("=" * 64)


if __name__ == "__main__":
    main()
