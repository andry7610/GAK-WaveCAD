"""
Тесты интеграции Coupling Monitor с плазмой v0.2.

Проверяют:
  1. Lawson penalty — lawson_ok=False → stability × 0.7
  2. dE_dt → thermal coupling — разогрев плазмы усиливает k_plasma_thermal
  3. Phase WARNING → system_status=DEGRADED
  4. Новые ключи в результатах (lawson_ok, tau_E, dE_dt)
"""

import pytest
from physical_modules.coupling_monitor import CouplingMonitor


def _make_coupling():
    c = CouplingMonitor()
    c.init()
    return c


def _plasma_results(**overrides):
    """Мок результатов плазмы с полями v0.2."""
    base = {
        "module": "plasma_monitor",
        "ekman_thickness": 1e-4,
        "B_field": 2.0,
        "sigma_viscous": 1e3,
        "diffusion_coeff": 1e-4,
        "barrier_index": 0.5,
        "phase": "STABLE",
        "plasma_stress": 1e5,
        "magnetic_coupling": 0.13,
        "temperature_plasma": 5e6,
        "rotation_omega": 1e3,
        # v0.2 fields
        "lawson_ok": False,
        "lawson_triple": 1e16,
        "tau_E": 1e-5,
        "dE_dt": 0.0,
        "P_fusion": 0.0,
        "P_rad": 1e6,
        "P_cond": 1e6,
    }
    base.update(overrides)
    return base


# --- 1. Lawson penalty ---

def test_coupling_lawson_penalty():
    """lawson_ok=False → stability падает на 30%."""
    # С плазмой, lawson_ok=True
    c1 = _make_coupling()
    c1.set_results(plasma=_plasma_results(lawson_ok=True, phase="STABLE"))
    c1.run()
    stability_ignited = c1.get_results()["stability_index"]

    # С плазмой, lawson_ok=False
    c2 = _make_coupling()
    c2.set_results(plasma=_plasma_results(lawson_ok=False, phase="STABLE"))
    c2.run()
    stability_not_ignited = c2.get_results()["stability_index"]

    assert stability_not_ignited < stability_ignited, \
        "Незажигание должно снижать стабильность"

    ratio = stability_not_ignited / stability_ignited
    assert abs(ratio - 0.7) < 0.01, \
        f"Штраф должен быть ~70%, получили {ratio:.3f}"


def test_coupling_lawson_ok_in_results():
    """lawson_ok пробрасывается в результаты coupling."""
    c = _make_coupling()
    c.set_results(plasma=_plasma_results(lawson_ok=True))
    c.run()

    assert c.get_results()["lawson_ok"] is True

    c2 = _make_coupling()
    c2.set_results(plasma=_plasma_results(lawson_ok=False))
    c2.run()

    assert c2.get_results()["lawson_ok"] is False


# --- 2. dE_dt → thermal coupling ---

def test_coupling_dE_dt_to_thermal():
    """dE_dt > 0 → k_plasma_thermal растёт."""
    # Плазма без разогрева (T_plasma = T_thermal → k=0)
    c1 = _make_coupling()
    c1.set_results(
        plasma=_plasma_results(dE_dt=0.0, temperature_plasma=300.0),
        thermal={"temperature": 300.0},
    )
    c1.run()
    k_base = c1.get_results()["k_plasma_thermal"]

    # Плазма с разогревом
    c2 = _make_coupling()
    c2.set_results(
        plasma=_plasma_results(dE_dt=1e10, temperature_plasma=300.0),
        thermal={"temperature": 300.0},
    )
    c2.run()
    k_heating = c2.get_results()["k_plasma_thermal"]

    assert k_heating > k_base, \
        f"dE_dt > 0 должно увеличивать k_plasma_thermal: {k_base:.4e} → {k_heating:.4e}"


def test_coupling_dE_dt_in_results():
    """dE_dt пробрасывается в результаты coupling."""
    c = _make_coupling()
    c.set_results(plasma=_plasma_results(dE_dt=1e10))
    c.run()

    assert c.get_results()["dE_dt"] == 1e10


def test_coupling_tau_E_in_results():
    """tau_E пробрасывается в результаты coupling."""
    c = _make_coupling()
    c.set_results(plasma=_plasma_results(tau_E=1.5))
    c.run()

    assert c.get_results()["tau_E"] == 1.5


# --- 3. Phase WARNING → DEGRADED ---

def test_coupling_plasma_warning():
    """phase=WARNING → system_status=DEGRADED."""
    c = _make_coupling()
    c.set_results(plasma=_plasma_results(phase="WARNING", lawson_ok=True))
    c.run()

    status = c.get_results()["system_status"]
    assert status == "DEGRADED", \
        f"WARNING должно давать DEGRADED, получили {status}"


def test_coupling_plasma_stable_healthy():
    """phase=STABLE + lawson_ok=True + высокая stability → HEALTHY."""
    c = _make_coupling()
    c.set_results(plasma=_plasma_results(phase="STABLE", lawson_ok=True))
    c.run()

    status = c.get_results()["system_status"]
    # stability с lawson_ok=True должна быть > 0.7 → HEALTHY
    assert status == "HEALTHY", \
        f"STABLE + lawson_ok должен давать HEALTHY, получили {status}"


def test_coupling_plasma_breakdown_critical():
    """phase=BREAKDOWN → system_status=CRITICAL (независимо от stability)."""
    c = _make_coupling()
    c.set_results(plasma=_plasma_results(phase="BREAKDOWN", lawson_ok=True))
    c.run()

    status = c.get_results()["system_status"]
    assert status == "CRITICAL", \
        f"BREAKDOWN должно давать CRITICAL, получили {status}"


def test_coupling_no_plasma():
    """Без плазмы — lawson_ok=None, stability базовая."""
    c = _make_coupling()
    c.set_results()
    c.run()

    r = c.get_results()
    assert r["lawson_ok"] is None
    assert r["stability_index"] > 0.9
    assert r["system_status"] == "HEALTHY"
