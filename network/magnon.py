"""
network/magnon.py — Магнонная динамика на графе.
v0.4: Решатель LLG (RK4) + магнитоупругий обмен B_me.

История:
  v0.3 — каркас от GigaChat (явный Эйлер, нестабилен)
  v0.4 — RK4, правильный M_s, магнитоупругая связь, совместимость с графом
"""

import numpy as np
from scipy.sparse import csr_matrix


class MagnonState:
    """
    Магнонная динамика на графе.
    Решатель LLG (Ландау-Лифшица-Гилберта) методом RK4.
    """

    def __init__(self, graph, M_s=1750.0, alpha=0.01, gamma=2.8e6,
                 A_ex=1.3e-11, B_me=2.0e6, H_ext=None):
        """
        Параметры:
          M_s    — намагниченность насыщения (Гс), для YIG ~1750
          alpha  — демпфирование Гилберта (безразм.)
          gamma  — гиромагнитное отношение (Гц/Э)
          A_ex   — константа обмена (эрг/см)
          B_me   — константа магнитоупругой связи (эрг/см³)
          H_ext  — внешнее поле (Э), вдоль Z по умолчанию
        """
        self.graph = graph
        self.N = graph.N
        self.edges = graph.edges

        self.M_s = float(M_s)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.A_ex = float(A_ex)
        self.B_me = float(B_me)

        # Намагниченность: (N, 3), начально вдоль Z
        self.M = np.zeros((self.N, 3))
        self.M[:, 2] = self.M_s

        # Внешнее поле
        self.H_ext = np.zeros((self.N, 3))
        if H_ext is not None:
            self.H_ext[:, 2] = H_ext

        # Лапласиан графа (разреженный)
        L = np.asarray(graph.laplacian, dtype=float)
        self.L = csr_matrix(L)

        # Масштаб обменного поля
        self.J_ex = 2.0 * self.A_ex / (self.M_s * 1e-7)

        # История прецессии (для коллапса Борна)
        self.precession_amplitude = np.zeros(self.N)

    def _normalize_M(self):
        """Нормировка |M_i| = M_s для каждого узла."""
        norms = np.linalg.norm(self.M, axis=1).reshape(-1, 1)
        norms[norms == 0] = 1.0
        self.M = self.M / norms * self.M_s

    @staticmethod
    def _normalize_vector(M):
        """Нормировка вектора намагниченности (in-place)."""
        norms = np.linalg.norm(M, axis=1).reshape(-1, 1)
        norms[norms == 0] = 1.0
        M /= norms
        M *= np.linalg.norm(M[0]) if np.linalg.norm(M[0]) > 0 else 1.0

    def compute_effective_field(self, M, external_stress=None):
        """
        Эффективное поле: H_eff = H_ex + H_an + H_ext + H_me
        """
        H = np.zeros_like(M)

        # Обменное поле
        H += -self.J_ex * (self.L @ M)

        # Поле анизотропии (одноосная, вдоль Z)
        K_u = 1.0e3
        H[:, 2] += (2.0 * K_u / self.M_s) * M[:, 2]

        # Внешнее поле
        H += self.H_ext

        # Магнитоупругое поле от напряжения
        if external_stress is not None:
            stress_factor = (3.0 * self.B_me / self.M_s**2) * external_stress
            H[:, 2] += stress_factor * M[:, 2]

        return H

    def _llg_rhs(self, M, H_eff):
        """
        Правая часть LLG:
        dM/dt = -gamma [M x H_eff] - alpha*gamma/M_s [M x (M x H_eff)]
        """
        gamma = self.gamma
        alpha = self.alpha
        Ms = self.M_s

        precession = gamma * np.cross(M, H_eff)
        cross_MH = np.cross(M, H_eff)
        damping = (alpha * gamma / Ms) * np.cross(M, cross_MH)

        return -(precession + damping)

    def _llg_rk4_step(self, dt, external_stress=None):
        """Один шаг RK4 для LLG."""
        M0 = self.M.copy()
        Ms = self.M_s

        # k1
        H1 = self.compute_effective_field(M0, external_stress)
        k1 = dt * self._llg_rhs(M0, H1)

        # k2
        M1 = M0 + 0.5 * k1
        norms1 = np.linalg.norm(M1, axis=1).reshape(-1, 1)
        norms1[norms1 == 0] = 1.0
        M1 = M1 / norms1 * Ms
        H2 = self.compute_effective_field(M1, external_stress)
        k2 = dt * self._llg_rhs(M1, H2)

        # k3
        M2 = M0 + 0.5 * k2
        norms2 = np.linalg.norm(M2, axis=1).reshape(-1, 1)
        norms2[norms2 == 0] = 1.0
        M2 = M2 / norms2 * Ms
        H3 = self.compute_effective_field(M2, external_stress)
        k3 = dt * self._llg_rhs(M2, H3)

        # k4
        M3 = M0 + k3
        norms3 = np.linalg.norm(M3, axis=1).reshape(-1, 1)
        norms3[norms3 == 0] = 1.0
        M3 = M3 / norms3 * Ms
        H4 = self.compute_effective_field(M3, external_stress)
        k4 = dt * self._llg_rhs(M3, H4)

        self.M = M0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        self._normalize_M()

        # Амплитуда прецессии
        self.precession_amplitude = np.sqrt(
            self.M[:, 0]**2 + self.M[:, 1]**2
        )

    def step(self, dt, external_stress=None):
        """
        Шаг интегрирования LLG методом RK4.

        Параметры:
          dt              — шаг по времени (с)
          external_stress — скаляр (Па) или массив (N,) — магнитоупругая связь
        """
        self._llg_rk4_step(dt, external_stress)

    def magnetoelastic_stress(self):
        """
        Магнитоупругое напряжение: sigma_me = B_me * (m_x² - m_y²)
        Возвращает массив (N,) — эффективное напряжение на узел.
        """
        mx = self.M[:, 0] / self.M_s
        my = self.M[:, 1] / self.M_s
        return self.B_me * (mx**2 - my**2)

    def precession_energy(self):
        """Энергия прецессии (для коллапса Борна)."""
        return 0.5 * np.sum(self.precession_amplitude**2) * self.M_s**2

    def nodes_above_threshold(self, threshold):
        """Узлы с амплитудой прецессии выше порога (для коллапса Борна)."""
        return np.where(self.precession_amplitude > threshold)[0]
