#!/usr/bin/env python3
"""
WaveCAD — Osmosis ↔ Acoustic Coupling Test (v0.4.1)

Связывает bio_diffusion (ионный транспорт) с acoustic_monitor
(акустика сферической оболочки) через осмотическое давление.

Цепочка:
  концентрации ионов → осмотическое давление Π
  → механическое напряжение σ (формула Лапласа)
  → сдвиг частоты акустической моды

Запуск:
  python tests/run_coupling_test.py
"""

import sys
import os
import numpy as np

# Добавляем корень репозитория в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.bio_diffusion import SyntheticBioDiffusion
from physical_modules.acoustic_monitor import AcousticMonitor


class SimpleGraph:
    """Минимальный граф для теста: цепочка из N узлов."""
    def __init__(self, N=10):
        self.N = N
        self.edges = []
        for i in range(N - 1):
            self.edges.append((i, i + 1, 1.0))
        L = np.zeros((N, N))
        for (i, j, w) in self.edges:
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
        self.laplacian = L


def osmosis_to_stress(osmotic_pressure_dimless, c_ref=100.0,
                      R=8.314, T=300.0):
    """Безразмерное осмотическое давление → механическое напряжение [Па].

    Pi_dimless = sum_k c_k / c_ref
    Pi_dimensional = R * T * c_ref * Pi_dimless  [Па]

    Для тонкой сферической оболочки: sigma ~ Pi (упрощённая формула Лапласа).
    """
    return R * T * c_ref * osmotic_pressure_dimless


def main():
    print("=" * 64)
    print("  WaveCAD — Osmosis <-> Acoustic Coupling Test")
    print("  v0.4.1 — bio_diffusion -> acoustic_monitor bridge")
    print("=" * 64)

    # 1. Граф
    graph = SimpleGraph(N=10)

    # 2. Ионы: Na+, K+, Cl-
    ions = [
        {"name": "Na+", "D": 1.0, "z": +1, "c0": 3.0},
        {"name": "K+",  "D": 0.8, "z": +1, "c0": 2.0},
        {"name": "Cl-", "D": 1.0, "z": -1, "c0": 5.0},
    ]

    # 3. Диффузия
    diff = SyntheticBioDiffusion(graph, ions, eps=1.0, F_RT=1.0, dt=1e-3)

    print("\n  [1] Начальное состояние:")
    print(f"  Концентрации (узел 5): "
          f"Na={diff.c[0,5]:.3f}  K={diff.c[1,5]:.3f}  Cl={diff.c[2,5]:.3f}")
    pi0 = diff.osmotic_pressure()
    print(f"  Осмотическое давление (безразм.): "
          f"min={pi0.min():.3f}  max={pi0.max():.3f}  mean={pi0.mean():.3f}")

    # 4. Акустический монитор — базовая линия
    acoustic = AcousticMonitor(R=0.01, rho=2700.0, E=70e9, nu=0.33)
    baseline = acoustic.analyze_all_modes()

    print("\n  [2] Акустика — базовая линия (без осмоса):")
    for mode, info in baseline.items():
        print(f"  {mode:12s} | f = {info['frequency']:.3f} Гц | "
              f"sigma = {info['sigma_total_MPa']:.4f} МПа")

    # 5. Прогон диффузии
    print("\n  [3] Прогон диффузии: 100 шагов...")
    for i in range(100):
        diff.step()

    pi1 = diff.osmotic_pressure()
    print(f"  Концентрации (узел 5): "
          f"Na={diff.c[0,5]:.3f}  K={diff.c[1,5]:.3f}  Cl={diff.c[2,5]:.3f}")
    print(f"  Осмотическое давление (безразм.): "
          f"min={pi1.min():.3f}  max={pi1.max():.3f}  mean={pi1.mean():.3f}")

    # 6. Мост: осмос → напряжение
    c_ref = 100.0
    R_gas = 8.314
    T = 300.0

    sigma_osmotic = osmosis_to_stress(pi1.mean(), c_ref=c_ref,
                                       R=R_gas, T=T)
    print(f"\n  [4] Мост: осмотическое давление -> напряжение")
    print(f"  Pi (размерное) = {R_gas * T * c_ref * pi1.mean():.2f} Па")
    print(f"  sigma_osmotic = {sigma_osmotic:.2f} Па "
          f"= {sigma_osmotic / 1e6:.6f} МПа")

    # 7. Связка с акустикой
    acoustic.set_internal_stress(sigma_osmotic)
    coupled = acoustic.analyze_all_modes()

    print(f"\n  [5] Акустика — с осмотическим напряжением:")
    for mode, info in coupled.items():
        print(f"  {mode:12s} | f = {info['frequency']:.6f} Гц | "
              f"df = {info['shift']:.6f} Гц | "
              f"sigma = {info['sigma_total_MPa']:.6f} МПа")

    # 8. Сравнение
    print(f"\n  [6] Сдвиг частот (базовая -> с осмосом):")
    for mode in baseline:
        f0 = baseline[mode]['frequency']
        f1 = coupled[mode]['frequency']
        print(f"  {mode:12s} | df = {f1 - f0:.6f} Гц | "
              f"df/f0 = {(f1 - f0)/f0 * 100:.4f}%")

    print("\n  Тест пройден ✅")
    print("=" * 64)


if __name__ == "__main__":
    main()
