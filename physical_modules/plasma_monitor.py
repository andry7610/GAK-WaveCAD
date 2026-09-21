"""
Plasma Monitor — модуль плазменной оболочки.

Модель:
  Плазма внутри сферической оболочки (R=1 см).
  Считает:
    - зону Экмана (пристеночный слой)
    - магнитное поле (внешнее + равновесное + Z-пинч)
    - вязкое напряжение (с подавлением магнитным полем)
    - барьерный индекс (устойчивость плазма-стенка)
    - фазовое состояние (STABLE / WARNING / CONTACT / BREAKDOWN)
  Принимает внешние напряжения от осмоса, термалки, акустики.

История:
  v0.2 — критерий Лоусона, энергобаланс, временная динамика (step)
  v0.1 — базовая реализация
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


# --- Физические константы ---
MU_0 = 4 * np.pi * 1e-7       # магнитная проницаемость вакуума, Гн/м
K_B = 1.380649e-23            # постоянная Больцмана, Дж/К
E_CHARGE = 1.602176634e-19    # элементарный заряд, Кл

# --- Топливные пресеты (массы ионов, кг) ---
FUEL_MASSES = {
    "D-He3": [3.34e-27, 5.01e-27],   # дейтерий, гелий-3
    "D-T":   [3.34e-27, 5.01e-27],   # дейтерий, тритий
    "D-D":   [3.34e-27, 3.34e-27],   # дейтерий, дейтерий
}

# --- Пороги Лоусона: n * T_keV * tau_E [кэВ·с/м³] ---
LAWSON_THRESHOLDS = {
    "D-T":   3.0e21,
    "D-D":   1.0e24,
    "D-He3": 1.0e22,
    "custom": 3.0e21,
}

# --- Энергия на реакцию [Дж] ---
FUEL_ENERGY = {
    "D-T":   17.6e6 * E_CHARGE,   # 17.6 МэВ
    "D-D":   3.65e6 * E_CHARGE,    # 3.65 МэВ (среднее по ветвям)
    "D-He3": 18.3e6 * E_CHARGE,   # 18.3 МэВ
    "custom": 17.6e6 * E_CHARGE,
}

# --- Коэффициент gyro-Bohm для tau_E ---
C_GYRO_BOHM = 0.14


@ModuleRegistry.register("plasma_monitor")
class PlasmaMonitor(BaseModule):
    """Монитор плазменной оболочки внутри сферы."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = dict(config or {})
        cfg.update(kwargs)

        # Геометрия
        self.radius = cfg.get("radius", 1.0e-2)
        self.shell_thickness = cfg.get("shell_thickness", 1.0e-4)
        self.gap = cfg.get("plasma_gap", 5.0e-4)

        # Параметры плазмы
        self.density_number = cfg.get("density_number", 1.0e20)
        self.temperature = cfg.get("temperature", 5.0e6)
        self.viscosity = cfg.get("viscosity", 1.0e-5)
        self.mass_density = cfg.get("mass_density", 1.0e-3)

        # Динамика
        self.rotation_freq = cfg.get("rotation_freq", 1.0e3)
        self.flow_velocity = cfg.get("flow_velocity", 1.0e4)
        self.z_pinch_freq = cfg.get("z_pinch_freq", 1.253e7)
        self.auto_feed = cfg.get("auto_feed", True)
        self.ekman_layer = cfg.get("ekman_layer", True)

        # Частицы
        self.ion_charge = cfg.get("ion_charge", 1.0)
        self.ion_mass = cfg.get("ion_mass", 1.673e-27)
        self.collision_freq = cfg.get("collision_freq", 1.0e9)

        # Магнитное поле
        self.external_B = cfg.get("external_B", 2.0)
        self.breakdown_B = cfg.get("breakdown_B", 15.0)

        # Пороги
        self.barrier_threshold = cfg.get("barrier_threshold", 0.3)

        # Топливо
        self.fuel_type = cfg.get("fuel_type", "custom")

        # Пересчёт массы иона по топливному пресету
        self._apply_fuel_preset()

        # Временная динамика (v0.2)
        self.sim_time = 0.0
        self.last_balance = None

        self.logger = get_logger("plasma_monitor")
        self.results = None

    def _apply_fuel_preset(self):
        """Пересчитать массу иона по топливному пресету.

        Берём массу более тяжёлого иона (основной реагент
        для термоядерного синтеза).
        """
        if self.fuel_type in FUEL_MASSES:
            masses = FUEL_MASSES[self.fuel_type]
            self.ion_mass = max(masses)
        # При "custom" — оставляем ion_mass из конфига

    def init(self) -> bool:
        self._set_initialized(True)
        self.logger.info(
            f"Инициализация Plasma Monitor: "
            f"R={self.radius:.2e} м, T={self.temperature:.2e} К, "
            f"B={self.external_B:.1f} Тл, fuel={self.fuel_type}"
        )
        return True

    def compute_ekman_layer(self) -> float:
        """Толщина пристеночного слоя Экмана.

        delta = sqrt(2 * nu / Omega) * R / (R - gap)
        """
        if not self.ekman_layer:
            return 0.0

        nu = self.viscosity
        Omega = self.rotation_freq
        R = self.radius
        gap = self.gap

        delta_base = np.sqrt(2.0 * nu / Omega)
        geom = R / (R - gap)
        return delta_base * geom

    def compute_magnetic_field(self) -> float:
        """Базовое магнитное поле: внешнее + равновесное.

        B = sqrt(B_ext^2 + B_eq^2)

        Вклад Z-пинча добавляется в run(), не здесь.
        """
        B_ext = self.external_B

        # Равновесное (тепловое) поле плазмы
        n = self.density_number
        T = self.temperature
        B_eq = np.sqrt(2.0 * MU_0 * n * K_B * T)

        return np.sqrt(B_ext**2 + B_eq**2)

    def compute_viscous_stress(self, delta_E: float, B: float) -> float:
        """Вязкое напряжение в слое Экмана.

        sigma = mu * v / delta_E * suppression(B)
        Магнитное поле подавляет напряжение.
        """
        if delta_E <= 0:
            return 0.0

        sigma_0 = self.viscosity * self.flow_velocity / delta_E

        # Подавление магнитным полем
        suppression = 1.0 / (1.0 + (B / self.breakdown_B) ** 4)

        return sigma_0 * suppression

    def compute_barrier_index(self, delta_E: float, B: float) -> float:
        """Барьерный индекс устойчивости плазма-стенка (0..1).

        0 — нет барьера (плазма касается стенки),
        1 — максимальный барьер.
        """
        if self.gap <= 0:
            return 0.0

        geom = self.gap / (self.gap + delta_E)
        mag = B / (B + self.breakdown_B)

        idx = geom * mag

        return max(0.0, min(1.0, idx))

    def _compute_diffusion_coeff(self, B: float) -> float:
        """Коэффициент диффузии (бомовская диффузия)."""
        if B <= 0:
            B = 1e-6
        D_bohm = (K_B * self.temperature) / (16.0 * E_CHARGE * B)
        return D_bohm

    def _compute_z_pinch_field(self) -> float:
        """Вклад Z-пинча (бегущее поле) в магнитное поле."""
        if self.z_pinch_freq <= 0:
            return 0.0

        omega_p = self.z_pinch_freq * 2.0 * np.pi
        B_z = MU_0 * self.flow_velocity * self.density_number * E_CHARGE / omega_p
        # Нормируем вклад
        B_z = min(B_z * 1e6, 0.5)

        return B_z

    def run(self, external_stress=None) -> dict:
        """Полный прогон модуля.

        Args:
            external_stress: dict с ключами sigma_thermal, sigma_osmotic,
                            acoustic_freq_shift (от других модулей).

        Returns:
            dict с результатами для Coupling Monitor.
        """
        ext = external_stress or {}

        # --- Базовое магнитное поле ---
        B = self.compute_magnetic_field()

        # --- Вклад Z-пинча (только в run, не в compute_magnetic_field) ---
        B_z = self._compute_z_pinch_field()
        if B_z > 0:
            B = np.sqrt(B**2 + B_z**2)

        # --- Слой Экмана ---
        delta_E = self.compute_ekman_layer()

        # --- Вязкое напряжение ---
        sigma_viscous = self.compute_viscous_stress(delta_E, B)

        # --- Барьерный индекс ---
        barrier = self.compute_barrier_index(delta_E, B)

        # --- Диффузия ---
        diffusion = self._compute_diffusion_coeff(B)

        # --- Температура плазмы ---
        T_plasma = self.temperature

        # Внешнее тепловое напряжение разогревает плазму
        sigma_thermal = ext.get("sigma_thermal", 0.0)
        if sigma_thermal > 0:
            volume = (4.0 / 3.0) * np.pi * self.radius**3
            dT = sigma_thermal * volume / (self.density_number * K_B * self.radius**2)
            T_plasma += dT

        # Осмотическое напряжение — слабый вклад
        sigma_osmotic = ext.get("sigma_osmotic", 0.0)
        if sigma_osmotic > 0:
            T_plasma += sigma_osmotic * 1e-3 / (self.density_number * K_B)

        # Автозапитка: плазменное динамо поддерживает температуру
        if self.auto_feed:
            T_plasma = max(T_plasma, self.temperature)
        else:
            # Без автозапитки — радиационные потери
            T_plasma *= 0.95

        # --- Угловая частота вращения ---
        Omega = self.rotation_freq

        # Акустический сдвиг меняет скорость вращения
        acoustic_shift = ext.get("acoustic_freq_shift", 0.0)
        if acoustic_shift != 0:
            Omega += acoustic_shift

        # --- Плазменное напряжение на стенку ---
        plasma_stress = sigma_viscous + self.mass_density * self.flow_velocity**2

        # --- Магнитная связь ---
        magnetic_coupling = B / self.breakdown_B

        # --- Фазовое состояние ---
        if B >= self.breakdown_B:
            phase = "BREAKDOWN"
        elif self.gap <= delta_E * 0.5:
            phase = "CONTACT"
        elif barrier < self.barrier_threshold:
            phase = "WARNING"
        else:
            phase = "STABLE"

        results = {
            "module": "plasma_monitor",
            "ekman_thickness": delta_E,
            "B_field": B,
            "sigma_viscous": sigma_viscous,
            "diffusion_coeff": diffusion,
            "barrier_index": barrier,
            "phase": phase,
            "plasma_stress": plasma_stress,
            "magnetic_coupling": magnetic_coupling,
            "temperature_plasma": T_plasma,
            "rotation_omega": Omega,
        }

        self.logger.info(
            f"Phase={phase}, B={B:.3e} T, delta_E={delta_E:.3e} м, "
            f"barrier={barrier:.3f}, T={T_plasma:.3e} К"
        )

        self.results = results
        return results

    def get_results(self) -> dict:
        return self.results

    # === Плазма v0.2: Лоусон, энергобаланс, динамика ===

    def _T_to_keV(self) -> float:
        """Перевод температуры из Кельвина в кэВ."""
        return self.temperature * K_B / (E_CHARGE * 1e3)

    def compute_tau_E(self) -> float:
        """Время удержания энергии (gyro-Bohm).

        tau_E = C_gB * a² * B² / (sqrt(T_keV) * n_20)

        где n_20 = n / 1e20 (в единицах 10^20 м⁻³),
        T_keV — температура в кэВ,
        B — полное магнитное поле (Тл),
        a — радиус плазмы (м).

        Возвращает время в секундах.
        """
        T_keV = self._T_to_keV()
        if T_keV <= 0:
            return 0.0

        n_20 = self.density_number / 1e20
        if n_20 <= 0:
            return 0.0

        # Полное магнитное поле
        B = self.compute_magnetic_field()
        B_z = self._compute_z_pinch_field()
        if B_z > 0:
            B = np.sqrt(B**2 + B_z**2)

        if B <= 0:
            return 0.0

        tau_E = C_GYRO_BOHM * self.radius**2 * B**2 / (np.sqrt(T_keV) * n_20)
        return float(tau_E)

    def compute_reactivity(self) -> float:
        """Реактивность <σv>(T) — Bosch-Hale / NRL аппроксимация.

        Возвращает <σv> в м³/с для заданного топлива и температуры.

        D-T:   3.68e-18 * T^(-2/3) * exp(-19.94 * T^(-1/3))
        D-D:   3.70e-18 * T^(-2/3) * exp(-46.10 * T^(-1/3))
        D-He3: 5.50e-18 * T^(-2/3) * exp(-38.40 * T^(-1/3))
        """
        T_keV = self._T_to_keV()
        if T_keV <= 0:
            return 0.0

        T_pow = T_keV ** (-2.0 / 3.0)

        if self.fuel_type == "D-T":
            sv = 3.68e-18 * T_pow * np.exp(-19.94 * T_keV ** (-1.0 / 3.0))
        elif self.fuel_type == "D-D":
            sv = 3.70e-18 * T_pow * np.exp(-46.10 * T_keV ** (-1.0 / 3.0))
        elif self.fuel_type == "D-He3":
            sv = 5.50e-18 * T_pow * np.exp(-38.40 * T_keV ** (-1.0 / 3.0))
        else:
            # custom — по умолчанию D-T
            sv = 3.68e-18 * T_pow * np.exp(-19.94 * T_keV ** (-1.0 / 3.0))

        return float(sv)

    def compute_energy_balance(self) -> dict:
        """Энергобаланс плазмы: P_fusion vs P_rad + P_cond.

        P_fusion — термоядерная мощность (Вт/м³)
        P_rad    — тормозное излучение Bremsstrahlung (Вт/м³)
        P_cond   — потери на теплопроводность (Вт/м³)
        dE_dt    — чистый приток энергии (Вт/м³)

        P_fusion = (n²/4) * <σv> * E_fus    (D-T, D-He3)
        P_fusion = (n²/2) * <σv> * E_fus    (D-D, одинаковые частицы)
        P_rad    = 1.69e-38 * Z_eff * n² * sqrt(T_keV)
        P_cond   = 3 * n * k_B * T / tau_E
        """
        T_keV = self._T_to_keV()
        n = self.density_number
        sv = self.compute_reactivity()
        E_fus = FUEL_ENERGY.get(self.fuel_type, 17.6e6 * E_CHARGE)

        # --- Термоядерная мощность ---
        if self.fuel_type == "D-D":
            # Одинаковые частицы: n_D = n, фактор 1/2
            P_fusion = 0.5 * n**2 * sv * E_fus
        else:
            # D-T, D-He3: n_D = n_T = n/2
            P_fusion = 0.25 * n**2 * sv * E_fus

        # --- Тормозное излучение (Bremsstrahlung) ---
        Z_eff = 1.0  # Для чистого топлива Z_eff ≈ 1
        P_rad = 1.69e-38 * Z_eff * n**2 * np.sqrt(max(T_keV, 0.0))

        # --- Потери на теплопроводность ---
        tau_E = self.compute_tau_E()
        if tau_E > 0:
            P_cond = 3.0 * n * K_B * self.temperature / tau_E
        else:
            P_cond = float('inf')

        # --- Чистый баланс ---
        dE_dt = P_fusion - P_rad - P_cond

        result = {
            "P_fusion": float(P_fusion),
            "P_rad": float(P_rad),
            "P_cond": float(P_cond),
            "dE_dt": float(dE_dt),
            "tau_E": float(tau_E),
            "T_keV": float(T_keV),
            "reactivity": float(sv),
        }

        self.last_balance = result
        return result

    def check_lawson(self) -> bool:
        """Критерий Лоусона: n * T_keV * tau_E >= threshold?

        Пороги:
            D-T:   3 × 10²¹ кэВ·с/м³
            D-D:   1 × 10²⁴ кэВ·с/м³
            D-He3: 1 × 10²² кэВ·с/м³

        Возвращает True, если зажигание возможно.
        """
        threshold = LAWSON_THRESHOLDS.get(self.fuel_type, 3.0e21)

        T_keV = self._T_to_keV()
        tau_E = self.compute_tau_E()

        product = self.density_number * T_keV * tau_E
        return product >= threshold

    def step(self, dt: float) -> dict:
        """Один шаг временной динамики.

        dT/dt = (P_fusion - P_rad - P_cond) / (3/2 * n * k_B)

        Обновляет self.temperature и self.sim_time.
        Возвращает словарь с состоянием после шага.
        """
        balance = self.compute_energy_balance()
        n = self.density_number
        dE_dt = balance["dE_dt"]

        # Изменение температуры
        if n > 0:
            dT = dE_dt * dt / (1.5 * n * K_B)
        else:
            dT = 0.0

        self.temperature += dT

        # Защита от нефизичных значений
        if np.isnan(self.temperature):
            self.temperature = 0.0
        if self.temperature < 0:
            self.temperature = 0.0
        if self.temperature > 1e12:
            self.temperature = 1e12

        self.sim_time += dt

        result = {
            "temperature": float(self.temperature),
            "T_keV": float(self._T_to_keV()),
            "dE_dt": float(dE_dt),
            "P_fusion": balance["P_fusion"],
            "P_rad": balance["P_rad"],
            "P_cond": balance["P_cond"],
            "tau_E": balance["tau_E"],
            "sim_time": float(self.sim_time),
            "lawson_ignited": self.check_lawson(),
        }

        return result
