"""
tests/run_magnon_test.py — Тест магнонной динамики (LLG + RK4).
Запуск из корня репозитория:
    python tests/run_magnon_test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.magnon import MagnonState


class SimpleGraph:
    """Минимальный граф для теста."""
    def __init__(self, N, edges_list):
        self.N = N
        self.edges = edges_list
        L = np.zeros((N, N))
        for (i, j, w) in edges:
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
        self.laplacian = L


def main():
    print("=" * 64)
    print("  WaveCAD — Magnon Dynamics Test (LLG + RK4)")
    print("  v0.4.2 — magnon.py")
    print("=" * 64)

    # Граф: цепочка из 8 узлов
    N = 8
    edges = [(i, i + 1, 1.0) for i in range(N - 1)]
    graph = SimpleGraph(N, edges)

    magnon = MagnonState(
        graph, M_s=1750.0, alpha=0.01, gamma=2.8e6,
        A_ex=1.3e-11, B_me=2.0e6, H_ext=1000.0
    )

    print(f"\n  [1] Начальное состояние:")
    print(f"  N узлов: {magnon.N}")
    print(f"  M_s = {magnon.M_s} Гс")
    print(f"  alpha = {magnon.alpha}")
    print(f"  H_ext = {magnon.H_ext[0, 2]} Э")

    # Возмущение
    magnon.M[0, 0] = 50.0
    magnon._normalize_M()
    print(f"\n  [2] После возмущения (узел 0):")
    print(f"  |M[0]| = {np.linalg.norm(magnon.M[0]):.2f}")

    # Прогон
    dt = 1e-11
    n_steps = 5000
    print(f"\n  [3] Прогон LLG: {n_steps} шагов, dt = {dt:.1e} с")

    history_Mx = []
    for step in range(n_steps):
        magnon.step(dt)
        if step % 100 == 0:
            history_Mx.append(magnon.M[0, 0])

    # Результаты
    print(f"\n  [4] Амплитуда прецессии:")
    amps = magnon.precession_amplitude
    for i in range(N):
        print(f"    узел {i}: |M_perp| = {amps[i]:.4f} Гс ({amps[i]/magnon.M_s*100:.3f}%)")

    print(f"\n  [5] Проверка |M|:")
    norms = np.linalg.norm(magnon.M, axis=1)
    max_dev = np.max(np.abs(norms - magnon.M_s))
    print(f"  max отклонение = {max_dev:.2e}")
    print(f"  {'✓ Нормировка сохранена' if max_dev < 1e-6 else '⚠ Отклонение нормы'}")

    print(f"\n  [6] Магнитоупругое напряжение:")
    me = magnon.magnetoelastic_stress()
    print(f"  mean σ_me = {me.mean():.4e}")

    # Частота
    signal = np.array(history_Mx)
    if np.std(signal) > 1e-10:
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        total_time = len(history_Mx) * 100 * dt
        freq_est = zero_crossings / (2 * total_time)
        f_fmr = magnon.gamma * magnon.H_ext[0, 2] / (2 * np.pi)
        print(f"\n  [7] Частота прецессии: {freq_est:.4e} Гц (ФМР: {f_fmr:.4e} Гц)")

    print(f"\n  Тест пройден ✅")
    print("=" * 64)


if __name__ == "__main__":
    main()
