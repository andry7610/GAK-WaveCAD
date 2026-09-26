"""
BioBridge — модуль био-моста: нейроны ↔ физика.

Связывает нейронную активность с физическими модулями:
  - Нейроны → acoustic_freq_shift (спайки сдвигают резонансные частоты)
  - Плазма → feedback_current (температура плазмы влияет на нейроны)
  - Вакуум → P_gas (давление газа модулирует активность)

История:
  v0.1 — адаптация под GAK-WaveCAD (BaseModule, ModuleRegistry)
"""

import numpy as np
from core.base_module import BaseModule
from core.module_registry import ModuleRegistry
from core.logger import get_logger


@ModuleRegistry.register("bio_bridge")
class BioBridge(BaseModule):
    """Био-мост: нейроны ↔ акустика ↔ плазма ↔ вакуум."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config)

        cfg = config or {}
        # Параметры нейронной сети
        self.n_neurons = cfg.get("n_neurons", kwargs.get("n_neurons", 100))
        self.firing_rate = cfg.get("firing_rate", kwargs.get("firing_rate", 10.0))  # Гц
        self.decay_tau = cfg.get("decay_tau", kwargs.get("decay_tau", 0.1))  # с
        self.threshold = cfg.get("threshold", kwargs.get("threshold", 0.5))
        self.refractory = cfg.get("refractory", kwargs.get("refractory", 0.002))  # с

        # Параметры связи
        self.coupling_strength = cfg.get("coupling_strength", kwargs.get("coupling_strength", 1.0))
        self.acoustic_coupling = cfg.get("acoustic_coupling", kwargs.get("acoustic_coupling", 0.01))  # Гц/спайк
        self.plasma_feedback_gain = cfg.get("plasma_feedback_gain", kwargs.get("plasma_feedback_gain", 1e-4))
        self.vacuum_coupling = cfg.get("vacuum_coupling", kwargs.get("vacuum_coupling", 1e-6))

        # Внутреннее состояние
        self.membrane_potential = np.zeros(self.n_neurons)
        self.spike_train = np.zeros(self.n_neurons, dtype=int)
        self.last_spike_time = np.full(self.n_neurons, -1.0)
        self.activity = 0.0
        self.time = 0.0

        self.logger = get_logger("bio_bridge")
        self.results = None

    def init(self) -> bool:
        self._set_initialized(True)
        # Случайная инициализация мембранных потенциалов
        rng = np.random.default_rng(42)
        self.membrane_potential = rng.uniform(0.0, self.threshold * 0.8, self.n_neurons)
        self.logger.info(f"BioBridge инициализирован: {self.n_neurons} нейронов")
        return True

    def step(self, dt, neuron_activity=None, T_plasma=0.0, P_gas=0.0):
        """Один шаг эволюции нейронной сети.
        
        Args:
            dt: шаг времени (с)
            neuron_activity: внешний входной сигнал (опционально)
            T_plasma: температура плазмы (К) — обратная связь
            P_gas: давление газа (Па) — обратная связь от вакуума
        """
        self.time += dt
        rng = np.random.default_rng()

        # Внешний вход
        if neuron_activity is not None:
            if isinstance(neuron_activity, (int, float)):
                external_input = np.full(self.n_neurons, float(neuron_activity))
            else:
                external_input = np.asarray(neuron_activity, dtype=float)
                if external_input.size < self.n_neurons:
                    external_input = np.resize(external_input, self.n_neurons)
                else:
                    external_input = external_input[:self.n_neurons]
        else:
            external_input = rng.normal(0, 0.1, self.n_neurons)

        # Обратная связь от плазмы: высокая T → возбуждение, низкая → торможение
        if T_plasma > 0:
            plasma_signal = self.plasma_feedback_gain * np.log10(max(T_plasma, 1.0))
            external_input += plasma_signal

        # Обратная связь от вакуума: давление газа модулирует активность
        if P_gas > 0:
            vacuum_signal = self.vacuum_coupling * np.log10(max(P_gas, 1.0))
            external_input += vacuum_signal

        # Обновление мембранных потенциалов (Leaky Integrate-and-Fire)
        alpha = 1.0 - np.exp(-dt / self.decay_tau)
        self.membrane_potential += alpha * (external_input - self.membrane_potential)

        # Проверка порога и рефрактерности
        in_refractory = (self.time - self.last_spike_time) < self.refractory
        self.spike_train = np.zeros(self.n_neurons, dtype=int)
        spike_mask = (self.membrane_potential >= self.threshold) & (~in_refractory)
        self.spike_train[spike_mask] = 1
        self.last_spike_time[spike_mask] = self.time

        # Сброс потенциала после спайка
        self.membrane_potential[spike_mask] = 0.0
        self.membrane_potential[in_refractory] *= 0.5  # Частичный сброс в рефрактерности

        # Подавление отрицательных потенциалов
        self.membrane_potential = np.maximum(self.membrane_potential, 0.0)

        # Общая активность
        n_spikes = int(np.sum(self.spike_train))
        self.activity = n_spikes / (self.n_neurons * max(dt, 1e-10))

        # Защита от NaN/Inf
        if not np.isfinite(self.activity):
            self.activity = 0.0
        self.membrane_potential = np.nan_to_num(self.membrane_potential, nan=0.0, posinf=self.threshold, neginf=0.0)

        return self.activity

    def compute_acoustic_freq_shift(self):
        """Сдвиг акустической частоты от спайковой активности."""
        return self.coupling_strength * self.acoustic_coupling * self.activity

    def compute_feedback_current(self, T_plasma=0.0):
        """Ток обратной связи от плазмы к нейронам."""
        if T_plasma <= 0:
            return 0.0
        return self.plasma_feedback_gain * np.log10(T_plasma) * self.activity

    def run(self, external=None) -> dict:
        """Полный прогон без внешнего шага (стационарный расчёт)."""
        if not self.is_initialized():
            raise RuntimeError("Module not initialized. Call init() first.")

        results = {
            "n_neurons": self.n_neurons,
            "activity": float(self.activity),
            "n_spikes": int(np.sum(self.spike_train)),
            "spike_rate": float(self.activity),
            "acoustic_freq_shift": float(self.compute_acoustic_freq_shift()),
            "feedback_current": float(self.compute_feedback_current()),
            "mean_potential": float(np.mean(self.membrane_potential)),
            "time": float(self.time),
            "bio_intensity": float(min(self.activity / self.firing_rate, 1.0)),
        }

        self.logger.info(f"BioBridge: activity={results['activity']:.2f} Hz, "
                         f"spikes={results['n_spikes']}, "
                         f"freq_shift={results['acoustic_freq_shift']:.4e} Hz")
        self.results = results
        return results

    def get_results(self) -> dict:
        if self.results is None:
            self.run()
        return self.results

    def print_report(self):
        r = self.get_results()
        print("\n  BioBridge — отчёт:")
        print(f"    Нейронов          : {r['n_neurons']}")
        print(f"    Активность        : {r['activity']:.2f} Гц")
        print(f"    Спайков (last)    : {r['n_spikes']}")
        print(f"    Сдвиг частоты     : {r['acoustic_freq_shift']:.4e} Гц")
        print(f"    Ток обратной связи: {r['feedback_current']:.4e}")
        print(f"    Ср. потенциал     : {r['mean_potential']:.4f}")
        print(f"    Время симуляции   : {r['time']:.3f} с")
        print(f"    Био-интенсивность : {r['bio_intensity']:.3f}")
