"""
network/born_collapse.py — Коллапс Борна: переход узлов из вакуума в материальное состояние.
v0.5 — magnoнная энергия → порог → материализация → обновление графа.

Когда энергия прецессии магнонов в узле превышает порог E_threshold,
узел "материализуется": веса его рёбер увеличиваются,
что влияет на лапласиан графа, используемый в био-диффузии и акустике.
"""
import numpy as np


class BornCollapse:
    """
    Управляет коллапсом Борна на графе.
    
    Параметры:
        graph         — объект графа с методами update_edge_weight, edge_weights
        E_threshold   — порог энергии прецессии для коллапса (эрг/см³)
        material_weight — вес ребра между двумя материальными узлами
        vacuum_weight   — вес ребра между двумя вакуумными узлами
    """
    
    def __init__(self, graph, E_threshold=1e5,
                 material_weight=10.0, vacuum_weight=1.0):
        self.graph = graph
        self.E_threshold = E_threshold
        self.material_weight = material_weight
        self.vacuum_weight = vacuum_weight
        
        # Состояние узлов: False = вакуум, True = материал
        self.collapsed = np.zeros(graph.N, dtype=bool)
        self.collapse_time = np.full(graph.N, -1.0)
        self.collapse_history = []  # [(step, node, energy)]
    
    def check_collapse(self, magnon_state, current_step=0, current_time=0.0):
        """
        Проверяет, достигли ли узлы порога коллапса.
        Возвращает список узлов, коллапсировавших на этом шаге.
        """
        energies = magnon_state.precession_energy
        newly_collapsed = []
        
        for node in range(self.graph.N):
            if not self.collapsed[node] and energies[node] > self.E_threshold:
                self.collapsed[node] = True
                self.collapse_time[node] = current_time
                self.collapse_history.append(
                    (current_step, node, energies[node]))
                newly_collapsed.append(node)
        
        return newly_collapsed
    
    def apply_collapse_to_graph(self, newly_collapsed):
        """
        Обновляет веса рёбер для коллапсировавших узлов.
        Ребро между двумя материальными узлами → material_weight.
        Ребро между материальным и вакуумным → среднее.
        """
        for node in newly_collapsed:
            for (i, j), old_w in list(self.graph.edge_weights.items()):
                if i == node or j == node:
                    other = j if i == node else i
                    if self.collapsed[other]:
                        new_w = self.material_weight
                    else:
                        new_w = 0.5 * (self.material_weight + self.vacuum_weight)
                    self.graph.update_edge_weight(i, j, new_w)
    
    def material_fraction(self):
        """Доля материализованных узлов."""
        return np.mean(self.collapsed)
    
    def summary(self):
        """Сводка состояния коллапса."""
        n_collapsed = int(np.sum(self.collapsed))
        if self.collapse_history:
            last = self.collapse_history[-1]
            return (f"Коллапсировано узлов: {n_collapsed}/{self.graph.N} "
                    f"({100*self.material_fraction():.1f}%) | "
                    f"Последний: узел {last[1]}, шаг {last[0]}, "
                    f"E = {last[2]:.2e}")
        return f"Коллапсировано узлов: 0/{self.graph.N}"
