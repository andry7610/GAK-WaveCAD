"""
tests/run_born_collapse_test.py — Тест коллапса Борна.
v0.5 — magnoнная энергия → порог → материализация → обновление графа.

Запуск:
    python tests/run_born_collapse_test.py
"""
import sys
import os
import numpy as np

# Добавляем корень репозитория в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.magnon import MagnonState
from network.born_collapse import BornCollapse


class Graph:
    """Простой граф для тестирования."""
    def __init__(self, N, edges=None):
        self.N = N
        if edges is None:
            edges = [(i, i+1, 1.0) for i in range(N-1)]
        self._edges = edges
        self.edge_weights = {}
        W = np.zeros((N, N))
        for i, j, w in edges:
            W[i, j] = w
            W[j, i] = w
            self.edge_weights[(min(i,j), max(i,j))] = w
        degrees = np.sum(W, axis=1)
        self.laplacian = np.diag(degrees) - W
        self.W = W
        self.degrees = degrees
    
    @property
    def edges(self):
        return [(i, j, self.edge_weights.get((min(i,j), max(i,j)), 0.0))
                for (i, j) in sorted(self.edge_weights.keys())]
    
    def update_edge_weight(self, i, j, w):
        key = (min(i, j), max(i, j))
        if key in self.edge_weights:
            self.edge_weights[key] = w
            W = np.zeros((self.N, self.N))
            for (a, b), wt in self.edge_weights.items():
                W[a, b] = wt
                W[b, a] = wt
            self.W = W
            self.degrees = np.sum(W, axis=1)
            self.laplacian = np.diag(self.degrees) - W


def main():
    print("=" * 64)
    print("  WaveCAD — Born Collapse Test")
    print("  v0.5 — magnon energy -> materialization -> graph update")
    print("=" * 64)
    
    # 1. Граф
    N = 30
    graph = Graph(N)
    print(f"\n  [1] Граф: {N} узлов, цепочка, веса = 1.0")
    print(f"  Лапласиан: {graph.laplacian.shape}, сумма весов = {np.sum(graph.W)/2:.1f}")
    
    # 2. Магноны
    magnon = MagnonState(graph, M_s=1750.0, alpha=0.01,
                         A_ex=1.3e-11, B_me=2.0e6,
                         H_ext=np.full(N, 1000.0))
    print(f"\n  [2] Магноны: M_s={magnon.M_s}, alpha={magnon.alpha}, H_ext=1000 Э")
    print(f"  |M| = {np.linalg.norm(magnon.M[0]):.1f}")
    
    # 3. Коллапс Борна
    E_threshold = 5e4
    born = BornCollapse(graph, E_threshold=E_threshold,
                        material_weight=10.0, vacuum_weight=1.0)
    print(f"\n  [3] Коллапс Борна: E_threshold = {E_threshold:.0e}")
    print(f"  material_weight = 10.0, vacuum_weight = 1.0")
    
    # 4. Возмущение
    center = N // 2
    angle = 0.5
    magnon.M[center, 0] = magnon.M_s * np.sin(angle)
    magnon.M[center, 2] = magnon.M_s * np.cos(angle)
    print(f"\n  [4] Возмущение: узел {center}, угол = {angle:.2f} рад")
    print(f"  Энергия в узле {center}: E = {magnon.precession_energy[center]:.2e}")
    print(f"  Порог: E_thr = {E_threshold:.2e}")
    over = "ДА" if magnon.precession_energy[center] > E_threshold else "НЕТ"
    print(f"  Превышает порог: {over}")
    
    # 5. Прогон
    dt = 1e-12
    steps = 3000
    check_interval = 50
    
    L_before = graph.laplacian.copy()
    W_before = graph.W.copy()
    
    print(f"\n  [5] Прогон: {steps} шагов, dt = {dt:.0e} с")
    print(f"  Проверка коллапса каждые {check_interval} шагов\n")
    
    for step in range(1, steps + 1):
        magnon.step(dt)
        if step % check_interval == 0:
            newly = born.check_collapse(magnon, current_step=step,
                                        current_time=step * dt)
            if newly:
                born.apply_collapse_to_graph(newly)
                for node in newly:
                    print(f"    Шаг {step:4d}: УЗЕЛ {node} КОЛЛАПСИРОВАН "
                          f"(E = {magnon.precession_energy[node]:.2e})")
            magnon.graph = graph
    
    # 6. Результаты
    print(f"\n  [6] Результаты:")
    print(f"  {born.summary()}")
    
    W_after = graph.W.copy()
    changed_edges = np.sum(np.abs(W_after - W_before) > 1e-10)
    print(f"  Изменено рёбер: {changed_edges // 2}")
    print(f"  Сумма весов: до={np.sum(W_before)/2:.1f}, "
          f"после={np.sum(W_after)/2:.1f}")
    
    norms = np.linalg.norm(magnon.M, axis=1)
    max_dev = np.max(np.abs(norms - magnon.M_s))
    print(f"  |M| после прогона: {np.mean(norms):.6f} "
          f"(отклонение: {max_dev:.2e})")
    
    # 7. Влияние на граф
    L_after = graph.laplacian.copy()
    L_diff = np.max(np.abs(L_after - L_before))
    print(f"\n  [7] Влияние на граф:")
    print(f"  max(|L_after - L_before|) = {L_diff:.4f}")
    graph_changed = "изменился" if L_diff > 0.01 else "не изменился"
    print(f"  Граф {graph_changed}")
    
    # 8. Энергии
    print(f"\n  [8] Энергии прецессии (первые 15 узлов):")
    energies = magnon.precession_energy
    for i in range(min(15, N)):
        status = "МАТ" if born.collapsed[i] else "ВАК"
        marker = " <<<" if born.collapsed[i] else ""
        print(f"    Узел {i:2d}: E = {energies[i]:.2e}  [{status}]{marker}")
    
    # Итог
    n_collapsed = int(np.sum(born.collapsed))
    test_passed = (n_collapsed > 0 and max_dev < 1e-3 and L_diff > 0.01)
    print(f"\n  Коллапсировано: {n_collapsed} узлов")
    print(f"  |M| сохранён: {'✅' if max_dev < 1e-3 else '❌'}")
    print(f"  Граф обновлён: {'✅' if L_diff > 0.01 else '❌'}")
    print(f"\n  Тест {'пройден ✅' if test_passed else 'провален ❌'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
