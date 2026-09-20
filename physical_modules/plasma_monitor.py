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
  v0.1 — базовая реализация
"""

import numpy as np

from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


# --- Константы ---
MU_0 = 4 * np.pi * 1e-7       # магнитная проницаемость вакуума, Гн/м
K_B = 1.380649e-23            # постоянная Больцмана, Дж/К
E_CHARGE = 1.602176634e-19    # элементарный заряд, Кл

# --- Топливные пресеты (массы ионов, кг) ---
FUEL_MASSES = {
    "D-He3": [3.34e-27, 5.01e-27],   # дейтерий, гелий-3
    "D-T":   [3.34e-27, 5.01e-27],   # дейтерий, тритий
    "D-D":   [3.34e-27, 3.34e-27],   # дейтерий, дейтерий
}


@ModuleRegistry.register("plasma_monitor")
class PlasmaMonitor(BaseModule):
    """Монитор плазменной оболочки внутри сферы."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}

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
        suppression = 1.0 / (1.0 + (B / self.breakdown_B) ** 2)

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
