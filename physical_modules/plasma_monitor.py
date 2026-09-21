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
  v0.3 — автозапитка (плазменное динамо, индукция в стенке)
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

        # Автозапитка (v0.3)
        self.auto_feed_enabled = cfg.get("auto_feed_enabled", True)
        self.sigma_wall = cfg.get("sigma_wall", 5.96e7)    # медь, См/м
        self.delta_wall = cfg.get("delta_wall", 0.01)      # 1 см

        self.logger = get_logger("plasma_monitor")
        self.results = None

    def _apply_fuel_preset(self):
        """Пересчитать массу иона по топливному пресету."""
        if self.fuel_type in FUEL_MASSES:
            masses = FUEL_MASSES[self.fuel_type]
            self.ion_mass = max(masses)

    def init(self) -> bool:
        self._set_initialized(True)
        self.logger.info(
            f"Инициализация Plasma Monitor: "
            f"R={self.radius:.2e} м, T={self.temperature:.2e} К, "
            f"B={self.external_B:.1f} Тл, fuel={self.fuel_type}"
        )
        return True

    def compute_ekman_layer(self) -> float:
        """Толщина пристеночного слоя Экмана."""
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
        """Базовое магнитное поле: внешнее + равновесное."""
        B_ext = self.external_B
        n = self.density_number
        T = self.temperature
        B_eq = np.sqrt(2.0 * MU_0 * n * K_B * T)
        return np.sqrt(B_ext**2 + B_eq**2)

    def compute_viscous_stress(self, delta_E: float, B: float) -> float:
        """Вязкое напряжение в слое Экмана."""
        if delta_E <= 0:
            return 0.0
        sigma_0 = self.viscosity * self.flow_velocity / delta_E
        suppression = 1.0 / (1.0 + (B / self.breakdown_B) ** 4)
        return sigma_0 * suppression

    def compute_barrier_index(self, delta_E: float, B: float) -> float:
        """Барьерный индекс устойчивости плазма-стенка (0..1)."""
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
        B_z = min(B_z * 1e6, 0.5)
        return B_z

    def run(self, external_stress=None) -> dict:
        """Полный прогон модуля."""
        ext = external_stress or {}

        B = self.compute_magnetic_field()
        B_z = self._compute_z_pinch_field()
        if B_z > 0:
            B = np.sqrt(B**2 + B_z**2)

        delta_E = self.compute_ekman_layer()
        sigma_viscous = self.compute_viscous_stress(delta_E, B)
        barrier = self.compute_barrier_index(delta_E, B)
        diffusion = self._compute_diffusion_coeff(B)
        T_plasma = self.temperature

        sigma_thermal = ext.get("sigma_thermal", 0.0)
        if sigma_thermal > 0:
            volume = (4.0 / 3.0) * np.pi * self.radius**3
            dT = sigma_thermal * volume / (self.density_number * K_B * self.radius**2)
            T_plasma += dT

        sigma_osmotic = ext.get("sigma_osmotic", 0.0)
        if sigma_osmotic > 0:
            T_plasma += sigma_osmotic * 1e-3 / (self.density_number * K_B)

        if self.auto_feed:
            T_plasma = max(T_plasma, self.temperature)
        else:
            T_plasma *= 0.95

        Omega = self.rotation_freq
        acoustic_shift = ext.get("acoustic_freq_shift", 0.0)
        if acoustic_shift != 0:
            Omega += acoustic_shift

        plasma_stress = sigma_viscous + self.mass_density * self.flow_velocity**2
        magnetic_coupling = B / self.breakdown_B

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
        """Время удержания энергии (gyro-Bohm)."""
        T_keV = self._T_to_keV()
        if T_keV <= 0:
            return 0.0
        n_20 = self.density_number / 1e20
        if n_20 <= 0:
            return 0.0
        B = self.compute_magnetic_field()
        B_z = self._compute_z_pinch_field()
        if B_z > 0:
            B = np.sqrt(B**2 + B_z**2)
        if B <= 0:
            return 0.0
        tau_E = C_GYRO_BOHM * self.radius**2 * B**2 / (np.sqrt(T_keV) * n_20)
        return float(tau_E)

    def compute_reactivity(self) -> float:
        """Реактивность <σv>(T) — NRL аппроксимация."""
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
            sv = 3.68e-18 * T_pow * np.exp(-19.94 * T_keV ** (-1.0 / 3.0))
        return float(sv)

    def compute_energy_balance(self) -> dict:
        """Энергобаланс плазмы: P_fusion vs P_rad + P_cond."""
        T_keV = self._T_to_keV()
        n = self.density_number
        sv = self.compute_reactivity()
        E_fus = FUEL_ENERGY.get(self.fuel_type, 17.6e6 * E_CHARGE)

        if self.fuel_type == "D-D":
            P_fusion = 0.5 * n**2 * sv * E_fus
        else:
            P_fusion = 0.25 * n**2 * sv * E_fus

        Z_eff = 1.0
        P_rad = 1.69e-38 * Z_eff * n**2 * np.sqrt(max(T_keV, 0.0))

        tau_E = self.compute_tau_E()
        if tau_E > 0:
            P_cond = 3.0 * n * K_B * self.temperature / tau_E
        else:
            P_cond = float('inf')

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
        """Критерий Лоусона: n * T_keV * tau_E >= threshold?"""
        threshold = LAWSON_THRESHOLDS.get(self.fuel_type, 3.0e21)
        T_keV = self._T_to_keV()
        tau_E = self.compute_tau_E()
        product = self.density_number * T_keV * tau_E
        return product >= threshold

    def step(self, dt: float) -> dict:
        """Один шаг временной динамики."""
        balance = self.compute_energy_balance()
        n = self.density_number
        dE_dt = balance["dE_dt"]

        if n > 0:
            dT = dE_dt * dt / (1.5 * n * K_B)
        else:
            dT = 0.0

        self.temperature += dT

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

    # === Автозапитка (v0.3) — плазменное динамо ===

    def compute_wall_current(self, v_radial: float) -> float:
        """Ток Фарадея в проводящей стенке.

        J_wall = sigma_wall * v_radial * B
        """
        if not self.auto_feed_enabled:
            return 0.0
        if abs(v_radial) < 1e-30:
            return 0.0
        B = self.compute_magnetic_field()
        B_z = self._compute_z_pinch_field()
        if B_z > 0:
            B = np.sqrt(B**2 + B_z**2)
        J = self.sigma_wall * v_radial * B
        return float(J)

    def compute_wall_field(self, J_wall: float) -> float:
        """Магнитное поле от тока в стенке (правило Ленца — против движения).

        B_wall = -mu_0 * J_wall * delta_wall
        """
        if not self.auto_feed_enabled:
            return 0.0
        B_wall = -MU_0 * J_wall * self.delta_wall
        return float(B_wall)

    def compute_wall_decay_time(self) -> float:
        """L/R-время затухания тока в стенке.

        tau_wall = mu_0 * sigma_wall * delta_wall * a
        """
        if self.sigma_wall <= 0 or self.delta_wall <= 0 or self.radius <= 0:
            return 0.0
        tau = MU_0 * self.sigma_wall * self.delta_wall * self.radius
        return float(tau)

    def compute_damping_rate(self) -> float:
        """Коэффициент затухания gamma = 1 / tau_wall."""
        tau = self.compute_wall_decay_time()
        if tau <= 0:
            return 0.0
        return 1.0 / tau

    def compute_auto_feed(self, delta: float, v_radial: float) -> dict:
        """Полный отклик автозапитки на возмущение."""
        if not self.auto_feed_enabled:
            return {
                "J_wall": 0.0,
                "B_wall": 0.0,
                "F_lorentz": 0.0,
                "gamma": 0.0,
                "tau_wall": 0.0,
                "delta": float(delta),
                "v_radial": float(v_radial),
                "damped": False,
                "auto_feed": False,
            }

        J_wall = self.compute_wall_current(v_radial)
        B_wall = self.compute_wall_field(J_wall)
        F_lorentz = J_wall * B_wall * self.delta_wall
        tau_wall = self.compute_wall_decay_time()
        gamma = 1.0 / tau_wall if tau_wall > 0 else 0.0

        return {
            "J_wall": float(J_wall),
            "B_wall": float(B_wall),
            "F_lorentz": float(F_lorentz),
            "gamma": float(gamma),
            "tau_wall": float(tau_wall),
            "delta": float(delta),
            "v_radial": float(v_radial),
            "damped": True,
            "auto_feed": True,
        }

    def step_auto_feed(self, dt: float, delta: float, v_radial: float) -> dict:
        """Шаг затухающего осциллятора (аналитическое решение).

        Уравнение: x'' + 2*gamma*x' + omega^2*x = 0

        Аналитическое решение устойчиво при любом dt
        (в отличие от явного Эйлера, который расходится при dt > 2/omega).
        """
        if not self.auto_feed_enabled:
            return {
                "delta": float(delta),
                "v_radial": float(v_radial),
                "J_wall": 0.0,
                "B_wall": 0.0,
                "F_lorentz": 0.0,
                "damped": False,
                "gamma": 0.0,
                "omega": 0.0,
                "auto_feed": False,
            }

        # --- Параметры осциллятора ---
        tau_wall = self.compute_wall_decay_time()
        gamma = 1.0 / tau_wall if tau_wall > 0 else 0.0

        B = self.compute_magnetic_field()
        B_z = self._compute_z_pinch_field()
        if B_z > 0:
            B = np.sqrt(B**2 + B_z**2)

        rho = self.mass_density if self.mass_density > 0 else 1e-3
        omega = B / np.sqrt(MU_0 * rho)

        # --- Ток и поле (для отчёта) ---
        J_wall = self.compute_wall_current(v_radial)
        B_wall = self.compute_wall_field(J_wall)
        F_lorentz = J_wall * B_wall * self.delta_wall

        # --- Аналитический шаг ---
        if omega <= 0 or gamma <= 0:
            return {
                "delta": float(delta),
                "v_radial": float(v_radial),
                "J_wall": float(J_wall),
                "B_wall": float(B_wall),
                "F_lorentz": float(F_lorentz),
                "damped": False,
                "gamma": float(gamma),
                "omega": float(omega),
                "auto_feed": True,
            }

        omega2 = omega ** 2
        gamma2 = gamma ** 2

        if gamma2 < omega2:
            # --- Underdamped ---
            omega_d = np.sqrt(omega2 - gamma2)
            cos_wt = np.cos(omega_d * dt)
            sin_wt = np.sin(omega_d * dt)
            exp_decay = np.exp(-gamma * dt)

            delta_new = exp_decay * (
                delta * cos_wt + (v_radial + gamma * delta) / omega_d * sin_wt
            )
            v_new = exp_decay * (
                v_radial * cos_wt
                - (gamma * (v_radial + gamma * delta) / omega_d + delta * omega_d) * sin_wt
            )
        else:
            # --- Overdamped ---
            sqrt_disc = np.sqrt(gamma2 - omega2)
            r1 = -gamma + sqrt_disc
            r2 = -gamma - sqrt_disc
            exp_r1 = np.exp(r1 * dt)
            exp_r2 = np.exp(r2 * dt)

            A = (v_radial - r2 * delta) / (r1 - r2)
            B_coeff = (r1 * delta - v_radial) / (r1 - r2)

            delta_new = A * exp_r1 + B_coeff * exp_r2
            v_new = A * r1 * exp_r1 + B_coeff * r2 * exp_r2

        # --- Защита от nan/inf ---
        if np.isnan(delta_new) or np.isinf(delta_new):
            delta_new = 0.0
        if np.isnan(v_new) or np.isinf(v_new):
            v_new = 0.0

        return {
            "delta": float(delta_new),
            "v_radial": float(v_new),
            "J_wall": float(J_wall),
            "B_wall": float(B_wall),
            "F_lorentz": float(F_lorentz),
            "damped": True,
            "gamma": float(gamma),
            "omega": float(omega),
            "auto_feed": True,
        }
