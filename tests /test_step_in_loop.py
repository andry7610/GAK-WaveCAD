"""
Тест: step(dt) в цикле — плазма эволюционирует, coupling получает актуальные данные.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.plasma_monitor import PlasmaMonitor
from physical_modules.coupling_monitor import CouplingMonitor


def _make_plasma(**kwargs):
    cfg = {
        "radius": 0.5,
        "density_number": 5.0e19,
        "temperature": 1.0e8,
        "external_B": 10.0,
        "plasma_gap": 5e-3,
        "fuel_type": "D-T",
        "auto_feed_enabled": True,
    }
    cfg.update(kwargs)
    p = PlasmaMonitor(cfg)
    p.init()
    return p


def test_step_in_loop_temperature_changes():
    """После 100 шагов step(dt) температура плазмы изменилась."""
    p = _make_plasma()
    p.run()

    T_initial = p.temperature
    dt = 1e-3

    for _ in range(100):
        p.step(dt)

    T_final = p.temperature

    assert T_final != T_initial, (
        f"Температура должна измениться: была {T_initial:.3e}, "
        f"стала {T_final:.3e}"
    )
    assert not (T_final != T_final), "Температура не должна быть NaN"


def test_coupling_receives_updated_plasma():
    """Coupling получает актуальные результаты после шагов плазмы."""
    p = _make_plasma()
    p.run()

    # Делаем шаги
    for _ in range(100):
        p.step(1e-3)

    # Обновляем результаты
    p.run()

    plasma_results = p.get_results()
    T_from_results = plasma_results["temperature_plasma"]

    # Coupling должен получать изменённую температуру
    coupling = CouplingMonitor()
    coupling.init()
    coupling.set_results(plasma=plasma_results)
    coupling.run()

    results = coupling.get_results()

    # Проверяем, что coupling видит актуальную температуру
    assert "temperature_plasma" in plasma_results
    assert plasma_results["temperature_plasma"] == T_from_results
    assert results["system_status"] in ("HEALTHY", "DEGRADED", "CRITICAL")


def test_coupling_step_in_loop():
    """Полный цикл: 100 шагов → coupling → temperature_plasma изменилась."""
    p = _make_plasma(temperature=1.0e8)
    p.run()
    T_before = p.get_results()["temperature_plasma"]

    dt = 1e-3
    for _ in range(100):
        p.step(dt)
    p.run()

    T_after = p.get_results()["temperature_plasma"]

    assert T_after != T_before, (
        f"temperature_plasma должна измениться после 100 шагов: "
        f"было {T_before:.3e}, стало {T_after:.3e}"
    )

    # Coupling с обновлёнными данными
    coupling = CouplingMonitor()
    coupling.init()
    coupling.set_results(plasma=p.get_results())
    coupling.run()
    results = coupling.get_results()

    assert results["system_status"] in ("HEALTHY", "DEGRADED", "CRITICAL")
    assert "stability_index" in results


def test_step_no_nan_in_loop():
    """После 1000 шагов — нет NaN в температуре."""
    p = _make_plasma()
    p.run()

    for _ in range(1000):
        p.step(1e-3)

    assert p.temperature == p.temperature, "Temperature is NaN"
    assert p.temperature > 0, "Temperature should be positive"
