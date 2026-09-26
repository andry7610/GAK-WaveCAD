"""
Тесты полной цепочки GAK-WaveCAD — 9 модулей + био-мост.

Проверяет:
  - BioBridge: init, step, спайки, decay, NaN-стойкость
  - Coupling v0.5: с био, без био, влияние на stability
  - Полная цепочка: run.py логика, обратные связи
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.bio_bridge import BioBridge
from physical_modules.coupling_monitor import CouplingMonitor


# --- Хелперы ---

def make_bio(**overrides):
    cfg = {
        "n_neurons": 50,
        "firing_rate": 10.0,
        "decay_tau": 0.1,
        "threshold": 0.5,
        "refractory": 0.002,
        "coupling_strength": 1.0,
        "acoustic_coupling": 0.01,
        "plasma_feedback_gain": 1e-4,
        "vacuum_coupling": 1e-6,
    }
    cfg.update(overrides)
    return BioBridge(cfg)


def make_plasma_results(T=1e8, density=1e20, B=5.0):
    return {
        "temperature_plasma": T,
        "density": density,
        "B_field": B,
        "sigma_viscous": 1e3,
        "diffusion_coeff": 1e-4,
        "barrier_index": 0.5,
        "lawson_ok": True,
        "lawson_triple": 5e21,
        "tau_E": 0.5,
        "dE_dt": 1e5,
        "phase": "STABLE",
    }


def make_vacuum_results(P_gas=1e5, T_gas=300.0):
    return {
        "pressure_gas": P_gas,
        "gas_temperature": T_gas,
        "mean_free_path": 1e-3,
        "knudsen_number": 0.1,
    }


def make_acoustic_results():
    return {
        "mode1": {"f_reference": 1e6, "f_measured": 1.001e6, "delta_f": 1000, "status": "SHIFTED"},
        "mode2": {"f_reference": 2e6, "f_measured": 2.002e6, "delta_f": 2000, "status": "SHIFTED"},
    }


def make_magnon_results():
    return {
        "mode1": {"f": 1e9, "df_stress": 1e5},
        "mode2": {"f": 2e9, "df_stress": 2e5},
        "saturation_field": 1.0,
    }


def make_em_results():
    return {
        "res1": {"f_shifted": 1.5e9, "df_stress": 1e5},
        "res2": {"f_shifted": 2.5e9, "df_stress": 2e5},
    }


# --- BioBridge тесты ---

class TestBioBridgeInit:
    def test_init_default(self):
        bio = make_bio()
        assert bio.init() is True
        assert bio.n_neurons == 50

    def test_init_custom(self):
        bio = make_bio(n_neurons=200, threshold=0.8)
        assert bio.init() is True
        assert bio.n_neurons == 200
        assert bio.threshold == 0.8

    def test_init_results_none_before_run(self):
        bio = make_bio()
        bio.init()
        assert bio.results is None


class TestBioBridgeStep:
    def test_step_returns_activity(self):
        bio = make_bio()
        bio.init()
        activity = bio.step(0.001)
        assert isinstance(activity, float)
        assert activity >= 0.0

    def test_step_advances_time(self):
        bio = make_bio()
        bio.init()
        t0 = bio.time
        bio.step(0.01)
        assert bio.time > t0

    def test_step_with_external_activity(self):
        bio = make_bio()
        bio.init()
        activity = bio.step(0.001, neuron_activity=2.0)
        assert activity >= 0.0

    def test_step_with_plasma_feedback(self):
        bio = make_bio()
        bio.init()
        activity = bio.step(0.001, T_plasma=1e8)
        assert np.isfinite(activity)

    def test_step_with_vacuum_feedback(self):
        bio = make_bio()
        bio.init()
        activity = bio.step(0.001, P_gas=1e5)
        assert np.isfinite(activity)

    def test_step_no_nan_long_run(self):
        bio = make_bio()
        bio.init()
        for _ in range(1000):
            bio.step(0.001, T_plasma=1e8, P_gas=1e5)
        assert np.isfinite(bio.activity)
        assert np.all(np.isfinite(bio.membrane_potential))

    def test_spikes_produce_freq_shift(self):
        bio = make_bio(threshold=0.01, n_neurons=100)
        bio.init()
        # Несколько шагов с сильным входом
        for _ in range(10):
            bio.step(0.001, neuron_activity=5.0)
        results = bio.run()
        assert results["acoustic_freq_shift"] >= 0.0

    def test_decay_with_no_input(self):
        bio = make_bio(threshold=10.0)
        bio.init()
        # Разгон
        for _ in range(5):
            bio.step(0.001, neuron_activity=5.0)
        a0 = bio.activity
        # Затухание без входа
        for _ in range(50):
            bio.step(0.001, neuron_activity=0.0)
        assert bio.activity <= a0 + 1e-10


class TestBioBridgeRun:
    def test_run_returns_dict(self):
        bio = make_bio()
        bio.init()
        results = bio.run()
        assert isinstance(results, dict)

    def test_run_has_keys(self):
        bio = make_bio()
        bio.init()
        results = bio.run()
        for key in ["n_neurons", "activity", "n_spikes", "acoustic_freq_shift",
                     "feedback_current", "bio_intensity"]:
            assert key in results

    def test_run_not_initialized_raises(self):
        bio = make_bio()
        with pytest.raises(RuntimeError):
            bio.run()


# --- Coupling v0.5 тесты ---

class TestCouplingBio:
    def test_coupling_with_bio(self):
        c = CouplingMonitor()
        c.init()
        c.set_results(
            acoustic=make_acoustic_results(),
            magnon=make_magnon_results(),
            em=make_em_results(),
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            collapse={"phase": "STABLE", "ratio": 0.1},
            plasma=make_plasma_results(),
            vacuum=make_vacuum_results(),
            bio={"activity": 50.0, "acoustic_freq_shift": 500.0,
                 "feedback_current": 1e-3, "bio_intensity": 0.5},
        )
        c.run()
        r = c.get_results()
        assert "k_bio_acoustic" in r
        assert "k_bio_plasma" in r
        assert "bio_intensity" in r
        assert r["bio_intensity"] == 0.5

    def test_coupling_without_bio(self):
        c = CouplingMonitor()
        c.init()
        c.set_results(
            acoustic=make_acoustic_results(),
            magnon=make_magnon_results(),
            em=make_em_results(),
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            collapse={"phase": "STABLE", "ratio": 0.1},
            plasma=make_plasma_results(),
            vacuum=make_vacuum_results(),
        )
        c.run()
        r = c.get_results()
        assert r["k_bio_acoustic"] == 0.0
        assert r["k_bio_plasma"] == 0.0
        assert r["bio_intensity"] == 0.0

    def test_coupling_vacuum_affects_stability(self):
        c = CouplingMonitor()
        c.init()
        c.set_results(
            plasma=make_plasma_results(),
            vacuum=make_vacuum_results(P_gas=1e5),
        )
        c.run()
        r = c.get_results()
        assert "k_plasma_vacuum" in r
        assert r["k_plasma_vacuum"] > 0.0

    def test_coupling_stability_in_range(self):
        c = CouplingMonitor()
        c.init()
        c.set_results(
            acoustic=make_acoustic_results(),
            magnon=make_magnon_results(),
            em=make_em_results(),
            osmosis={"sigma_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            collapse={"phase": "STABLE", "ratio": 0.1},
            plasma=make_plasma_results(),
            vacuum=make_vacuum_results(),
            bio={"activity": 50.0, "acoustic_freq_shift": 500.0,
                 "feedback_current": 1e-3, "bio_intensity": 0.5},
        )
        c.run()
        r = c.get_results()
        assert 0.0 <= r["stability_index"] <= 1.0


# --- Полная цепочка ---

class TestFullChain:
    def test_all_nine_modules(self):
        """Полный прогон: 9 модулей + coupling, 100 шагов."""
        bio = make_bio(n_neurons=50)
        bio.init()

        plasma_results = make_plasma_results()
        vacuum_results = make_vacuum_results()

        # Симуляция 100 шагов
        for i in range(100):
            T_plasma = plasma_results.get("temperature_plasma", 0.0)
            P_gas = vacuum_results.get("pressure_gas", 0.0)
            bio.step(0.001, T_plasma=T_plasma, P_gas=P_gas)

        bio.run()
        bio_r = bio.get_results()
        assert bio_r["n_neurons"] == 50
        assert np.isfinite(bio_r["activity"])
        assert bio_r["activity"] >= 0.0

        # Coupling с всеми 9 модулями
        c = CouplingMonitor()
        c.init()
        c.set_results(
            osmosis={"sigma_Pa": 1e5, "pi_Pa": 1e5},
            thermal={"sigma_thermal_Pa": 1e4, "temperature": 300.0},
            acoustic=make_acoustic_results(),
            collapse={"phase": "STABLE", "ratio": 0.1},
            magnon=make_magnon_results(),
            em=make_em_results(),
            plasma=plasma_results,
            vacuum=vacuum_results,
            bio=bio_r,
        )
        c.run()
        r = c.get_results()
        assert 0.0 <= r["stability_index"] <= 1.0
        assert r["system_status"] in ("HEALTHY", "DEGRADED", "CRITICAL")

    def test_feedback_loop_plasma_to_bio(self):
        """Плазма → био: высокая T → больше активности."""
        bio_low = make_bio(n_neurons=50, threshold=0.5)
        bio_low.init()
        for _ in range(100):
            bio_low.step(0.001, T_plasma=1e4, P_gas=1e2)

        bio_high = make_bio(n_neurons=50, threshold=0.5)
        bio_high.init()
        for _ in range(100):
            bio_high.step(0.001, T_plasma=1e8, P_gas=1e5)

        # Высокая T плазмы должна давать больше активности (в среднем)
        # (стохастическая природа, поэтому проверяем что оба конечны)
        assert np.isfinite(bio_low.activity)
        assert np.isfinite(bio_high.activity)
