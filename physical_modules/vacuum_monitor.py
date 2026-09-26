"""vacuum_monitor.py — Технический вакуум: откачка, проводимость, давление.

Версия: v0.1
  - Уравнение состояния: P = n * k_B * T
  - Проводимость канала: вязкий (Пуазейль) и молекулярный (Кнудсен)
  - Откачка: dP/dt = -S*P/V + Q_in/V
  - Средний пробег: lambda = k_B*T / (sqrt(2)*pi*d^2*P)
  - Связь с плазмой: принимает T_plasma, отдаёт P_gas
"""

import numpy as np

# Физические константы (СИ)
K_B = 1.380649e-23      # Дж/К
PI = np.pi


class VacuumMonitor:
    """Монитор технического вакуума.

    Parameters
    ----------
    config : dict
        volume : float — объём камеры (м³)
        pump_speed : float — скорость откачки (м³/с)
        gas_temp : float — температура газа (К)
        channel_length : float — длина канала (м)
        channel_diameter : float — диаметр канала (м)
        gas_viscosity : float — вязкость газа (Па·с)
        molecule_mass : float — масса молекулы (кг)
        molecule_diameter : float — диаметр молекулы (м)
        initial_pressure : float — начальное давление (Па)
    """

    MODULE_NAME = "vacuum_monitor"

    def __init__(self, config=None):
        config = config or {}

        # Параметры камеры
        self.chamber_volume = config.get("volume", 1.0)          # м³
        self.pump_speed = config.get("pump_speed", 0.1)          # м³/с
        self.gas_temp = config.get("gas_temp", 300.0)            # К

        # Параметры канала
        self.channel_length = config.get("channel_length", 1.0)    # м
        self.channel_diameter = config.get("channel_diameter", 0.01)  # м
        self.gas_viscosity = config.get("gas_viscosity", 1.8e-5)    # Па·с (воздух 300 К)

        # Свойства газа (по умолчанию N₂)
        self.molecule_mass = config.get("molecule_mass", 4.65e-26)     # кг
        self.molecule_diameter = config.get("molecule_diameter", 3.7e-10)  # м

        # Состояние
        self.pressure = config.get("initial_pressure", 1.0e5)    # Па
        self.time = 0.0
        self.results = None

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def init(self):
        self.results = None
        self.time = 0.0
        return True

    # ------------------------------------------------------------------
    # Физика
    # ------------------------------------------------------------------

    def compute_pressure(self, n_gas):
        """P = n * k_B * T — уравнение состояния идеального газа.

        Parameters
        ----------
        n_gas : float или np.ndarray
            Плотность газа (м⁻³).

        Returns
        -------
        float или np.ndarray
            Давление (Па).
        """
        return n_gas * K_B * self.gas_temp

    def compute_mean_free_path(self, P):
        """lambda = k_B*T / (sqrt(2) * pi * d² * P).

        Returns
        -------
        float
            Средний свободный пробег (м). inf при P <= 0.
        """
        if P <= 0:
            return float("inf")
        return (K_B * self.gas_temp /
                (np.sqrt(2.0) * PI * self.molecule_diameter**2 * P))

    def compute_knudsen(self, P):
        """Kn = lambda / d_channel — число Кнудсена."""
        lam = self.compute_mean_free_path(P)
        return lam / self.channel_diameter

    def compute_conductance(self, P):
        """Проводимость канала (м³/с).

        Вязкий режим (Kn < 0.01): C = (pi*d⁴ / 128*mu*L) * P_avg
        Молекулярный режим (Kn > 1): C = (pi*d³ / 12*L) * v_bar
        Переходный: сумма обоих вкладов.
        """
        if P <= 0:
            return 0.0

        Kn = self.compute_knudsen(P)
        d = self.channel_diameter
        L = self.channel_length

        # Вязкий вклад (Пуазейль)
        C_visc = (PI * d**4 / (128.0 * self.gas_viscosity * L)) * P

        # Молекулярный вклад (Кнудсен)
        v_bar = np.sqrt(8.0 * K_B * self.gas_temp /
                        (PI * self.molecule_mass))
        C_mol = (PI * d**3 / (12.0 * L)) * v_bar

        if Kn < 0.01:
            return C_visc
        elif Kn > 1.0:
            return C_mol
        else:
            # Переходный режим — простая аддитивная аппроксимация
            return C_visc + C_mol

    def compute_flow(self, P1, P2):
        """Q = C * Delta_P — поток газа (Па·м³/с)."""
        P_avg = (abs(P1) + abs(P2)) / 2.0
        C = self.compute_conductance(P_avg)
        return C * (P1 - P2)

    def step(self, dt, Q_in=0.0, S_pump=None):
        """Один шаг откачки: dP/dt = -S*P/V + Q_in/V.

        Parameters
        ----------
        dt : float
            Шаг по времени (с).
        Q_in : float
            Поступление газа (Па·м³/с).
        S_pump : float или None
            Эффективная скорость откачки (м³/с). По умолчанию self.pump_speed.
        """
        if S_pump is None:
            S_pump = self.pump_speed

        dP = (-S_pump * self.pressure + Q_in) / self.chamber_volume
        self.pressure += dP * dt
        self.pressure = max(self.pressure, 0.0)
        self.time += dt

    # ------------------------------------------------------------------
    # Полный прогон
    # ------------------------------------------------------------------

    def run(self, external=None):
        """Полный расчёт вакуумного состояния.

        Parameters
        ----------
        external : dict или None
            Если передан, может содержать:
            - 'temperature_plasma' — температура плазмы (К)
            - 'pressure_plasma' — давление плазмы (Па)
            - 'n_plasma' — плотность плазмы (м⁻³)
        """
        mean_free_path = self.compute_mean_free_path(self.pressure)
        Kn = self.compute_knudsen(self.pressure)
        conductance = self.compute_conductance(self.pressure)

        # Газ втекает из камеры в канал (P_chamber → 0, насос)
        flow = self.compute_flow(self.pressure, 0.0)

        # Связь с плазмой
        plasma_temp = None
        plasma_pressure = None
        if external is not None:
            plasma_temp = external.get("temperature_plasma")
            plasma_pressure = external.get("pressure_plasma")

        self.results = {
            "pressure_gas": self.pressure,
            "gas_temp": self.gas_temp,
            "mean_free_path": mean_free_path,
            "knudsen_number": Kn,
            "conductance": conductance,
            "flow_rate": flow,
            "pump_speed": self.pump_speed,
            "chamber_volume": self.chamber_volume,
            "time": self.time,
            "plasma_temp": plasma_temp,
            "plasma_pressure": plasma_pressure,
        }
        return self.results

    def get_results(self):
        return self.results

    def print_report(self):
        if self.results is None:
            print("  Vacuum: run() не вызывался")
            return
        r = self.results
        print(f"\n  Вакуумный анализ:")
        print(f"    Давление газа     : {r['pressure_gas']:.3e} Па")
        print(f"    Температура газа  : {r['gas_temp']:.1f} К")
        print(f"    Средний пробег    : {r['mean_free_path']:.3e} м")
        print(f"    Число Кнудсена    : {r['knudsen_number']:.3e}")
        print(f"    Проводимость      : {r['conductance']:.3e} м³/с")
        print(f"    Поток газа        : {r['flow_rate']:.3e} Па·м³/с")
