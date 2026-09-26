"""
GAK Quantum Vacuum Module
Физический вакуум: эффект Казимира + пробой Швингера.

Внешний модуль — подключается через run.py и config.yaml.
Не зависит от ядра КАД, использует только core/constants.py.

v0.2 — динамический E_field: обратная связь от T_plasma
"""

import math
from core.constants import (
    HBAR, C_LIGHT, E_CHARGE, M_ELECTRON, PI,
    SCHWINGER_FIELD, COMPTON_WAVELENGTH, ELECTRON_REST_ENERGY,
)


class GAKQuantumVacuum:
    """
    Модуль физического вакуума для платформы GAK.

    Считает:
    - Энергию и силу Казимира между параллельными пластинами
    - Критическое поле Швингера и вероятность рождения e+e- пар
    - Амплитуду нулевых колебаний поля
    - Скорость рождения пар (пар/м^3/с)
    - Динамический отклик E_field на температуру плазмы (v0.2)

    Параметры конфигурации (quantum_vacuum секция в config.yaml):
        gap:             зазор между пластинами, м (default 3.5e-9)
        area:            площадь пластин, м^2 (default 1e-6)
        E_field:         базовое электрическое поле, В/м (default 0.0)
        volume:          объём области, м^3 (default 1e-18)
        dt:              шаг по времени, с (default 1e-9)
        plasma_coupling:  безразмерный коэффициент связи T_plasma -> E_field (default 0.0)
        T_ref:           опорная температура для логарифмической связи, К (default 1e6)
    """

    def __init__(self, config=None):
        config = config or {}
        self.gap = float(config.get("gap", 3.5e-9))
        self.area = float(config.get("area", 1e-6))
        self.E_field = float(config.get("E_field", 0.0))
        self.E_field_base = self.E_field
        self.volume = float(config.get("volume", 1e-18))
        self.dt = float(config.get("dt", 1e-9))
        self.plasma_coupling = float(config.get("plasma_coupling", 0.0))
        self.T_ref = float(config.get("T_ref", 1e6))

        # Накопители
        self._total_casimir_energy = 0.0
        self._total_pairs_created = 0.0
        self._step_count = 0
        self._T_plasma_history = []

    # ── Динамический E_field (v0.2) ──

    def update_E_field(self, T_plasma):
        """
        Обновляет E_field на основе температуры плазмы.
        E = E_base * (1 + plasma_coupling * ln(T / T_ref))

        При plasma_coupling = 0: E_field не меняется (статический режим).
        """
        if self.plasma_coupling <= 0 or T_plasma <= 0:
            return
        ratio = T_plasma / self.T_ref
        if ratio <= 0:
            return
        factor = 1.0 + self.plasma_coupling * math.log(max(ratio, 1e-10))
        factor = max(factor, 0.01)
        self.E_field = self.E_field_base * factor

    # ── Эффект Казимира ──

    def casimir_energy(self, gap=None):
        d = gap if gap is not None else self.gap
        if d <= 0:
            raise ValueError("Зазор должен быть положительным")
        return -(PI ** 2) * HBAR * C_LIGHT * self.area / (720.0 * d ** 3)

    def casimir_force(self, gap=None):
        d = gap if gap is not None else self.gap
        if d <= 0:
            raise ValueError("Зазор должен быть положительным")
        return -(PI ** 2) * HBAR * C_LIGHT * self.area / (240.0 * d ** 4)

    def casimir_pressure(self, gap=None):
        d = gap if gap is not None else self.gap
        if d <= 0:
            raise ValueError("Зазор должен быть положительным")
        return -(PI ** 2) * HBAR * C_LIGHT / (240.0 * d ** 4)

    # ── Пробой Швингера ──

    @staticmethod
    def schwinger_critical_field():
        return SCHWINGER_FIELD

    def pair_probability(self, E_field=None):
        E = E_field if E_field is not None else self.E_field
        if E <= 0:
            return 0.0
        return math.exp(-PI * SCHWINGER_FIELD / E)

    def pair_rate(self, E_field=None):
        E = E_field if E_field is not None else self.E_field
        if E <= 0:
            return 0.0
        omega_c = ELECTRON_REST_ENERGY / HBAR
        alpha = E_CHARGE ** 2 / (4 * PI * HBAR * C_LIGHT)
        x = E / SCHWINGER_FIELD
        if x > 0:
            gamma = (alpha / PI) * (x ** 2) * omega_c * math.exp(-PI / x)
        else:
            gamma = 0.0
        return gamma

    # ── Нулевые колебания ──

    def zero_point_amplitude(self, freq=1e9):
        epsilon_0 = 8.854187817e-12
        return math.sqrt(HBAR * 2 * PI * freq / (2 * epsilon_0 * self.volume))

    # ── Шаг и запуск ──

    def step(self, dt=None, T_plasma=None):
        """
        Один шаг по времени.
        Если T_plasma передан — обновляет E_field перед расчётом.
        """
        dt = dt if dt is not None else self.dt

        if T_plasma is not None:
            self.update_E_field(T_plasma)
            self._T_plasma_history.append(T_plasma)

        E = self.casimir_energy()
        self._total_casimir_energy += E * dt
        rate = self.pair_rate()
        self._total_pairs_created += rate * self.volume * dt
        self._step_count += 1

    def run(self):
        """
        Возвращает словарь со всеми параметрами вакуума.
        Готово для coupling_monitor и run.py.
        """
        return {
            "casimir_energy": self.casimir_energy(),
            "casimir_force": self.casimir_force(),
            "casimir_pressure": self.casimir_pressure(),
            "schwinger_field": SCHWINGER_FIELD,
            "E_field": self.E_field,
            "E_field_base": self.E_field_base,
            "pair_probability": self.pair_probability(),
            "pair_rate": self.pair_rate(),
            "zero_point_amplitude": self.zero_point_amplitude(),
            "total_casimir_energy": self._total_casimir_energy,
            "total_pairs_created": self._total_pairs_created,
            "step_count": self._step_count,
        }

    def get_results(self):
        return self.run()

    def __repr__(self):
        return (f"GAKQuantumVacuum(gap={self.gap:.2e}m, "
                f"area={self.area:.2e}m^2, "
                f"E_field={self.E_field:.2e}V/m)")
