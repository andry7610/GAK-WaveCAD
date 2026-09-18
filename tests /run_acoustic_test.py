#!/usr/bin/env python3
"""
WaveCAD — Acoustic Monitor Test
v0.4.1 — Bessel roots + frequency accuracy + stress response

Запуск из корня репозитория:
    python tests/run_acoustic_test.py
"""

import sys
import os
import numpy as np

# Добавляем корень репозитория в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    print("=" * 64)
    print("  WaveCAD — Acoustic Monitor Test")
    print("  v0.4.1 — Bessel roots + frequency accuracy")
    print("=" * 64)

    # --- Создаём монитор ---
    monitor = AcousticMonitor(R=1.0, rho=2200.0, E=1.0e6, nu=0.30,
                              l_max=4, n_max=1)

    print("\n  [1] Параметры оболочки:")
    print(f"  R = {monitor.R} м")
    print(f"  ρ = {monitor.rho} кг/м³")
    print(f"  E = {monitor.E:.1e} Па")
    print(f"  ν = {monitor.nu}")
    print(f"  c_l = {monitor.c_l:.2f} м/с (продольная)")
    print(f"  c_t = {monitor.c_t:.2f} м/с (сдвиговая)")

    # --- Проверка корней Бесселя ---
    print("\n  [2] Проверка корней Бесселя:")

    # j_0(x) = sin(x)/x, нули → π, 2π, 3π
    j0_roots = monitor._bessel_zeros(0, 3)
    expected_j0 = np.array([np.pi, 2 * np.pi, 3 * np.pi])
    err_j0 = np.max(np.abs(j0_roots - expected_j0))
    print(f"  j_0 нули:    {j0_roots}")
    print(f"  Ожидаемо:    [π, 2π, 3π] = [{np.pi:.6f}, {2*np.pi:.6f}, {3*np.pi:.6f}]")
    print(f"  Погрешность: {err_j0:.2e} {'✅' if err_j0 < 1e-8 else '❌'}")

    # j_1(x), нули → 4.4934..., 7.7253..., 10.9041...
    j1_roots = monitor._bessel_zeros(1, 3)
    expected_j1 = np.array([4.493409458, 7.725251837, 10.904121659])
    err_j1 = np.max(np.abs(j1_roots - expected_j1))
    print(f"\n  j_1 нули:    {j1_roots}")
    print(f"  Ожидаемо:    [4.493409, 7.725252, 10.904122]")
    print(f"  Погрешность: {err_j1:.2e} {'✅' if err_j1 < 1e-5 else '❌'}")

    # --- Базовые частоты ---
    print("\n  [3] Базовые частоты (без напряжения):")
    modes_base = monitor.analyze_all_modes()
    for m in modes_base:
        sigma_mpa = m["stress"] / 1e6
        print(f"  {m['name']:14s} | f = {m['frequency']:12.3f} Гц | σ = {sigma_mpa:.4f} МПа | {m['status']}")

    # --- Частоты с напряжением ---
    sigma_test = 0.5e6  # 0.5 МПа
    print(f"\n  [4] Частоты с напряжением σ = {sigma_test/1e6:.1f} МПа:")
    monitor.set_internal_stress([sigma_test, 0, 0])
    modes_stressed = monitor.analyze_all_modes()
    for m in modes_stressed:
        sigma_mpa = m["stress"] / 1e6
        print(f"  {m['name']:14s} | f = {m['frequency']:12.3f} Гц | σ = {sigma_mpa:.4f} МПа | {m['status']}")

    # --- Сдвиг частот ---
    print("\n  [5] Сдвиг частот:")
    for m0, m1 in zip(modes_base, modes_stressed):
        df = m1["frequency"] - m0["frequency"]
        df_rel = df / m0["frequency"] * 100 if m0["frequency"] > 0 else 0
        print(f"  {m0['name']:14s} | Δf = {df:+.3f} Гц | Δf/f0 = {df_rel:+.4f}%")

    # --- Проверка физики ---
    print("\n  [6] Проверка симметрии:")
    radial_df = modes_stressed[0]["frequency"] - modes_base[0]["frequency"]
    shear_df = modes_stressed[1]["frequency"] - modes_base[1]["frequency"]
    print(f"  Радиальная (l=0): Δf = {radial_df:+.3f} Гц (ожидаем > 0) {'✅' if radial_df > 0 else '❌'}")
    print(f"  Сдвиговая  (l=1): Δf = {shear_df:+.3f} Гц (ожидаем < 0) {'✅' if shear_df < 0 else '❌'}")

    # --- Итог ---
    all_ok = (err_j0 < 1e-8) and (err_j1 < 1e-5) and (radial_df > 0) and (shear_df < 0)
    print(f"\n  Тест пройден {'✅' if all_ok else '❌'}")
    print("=" * 64)
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
