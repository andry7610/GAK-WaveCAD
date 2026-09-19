#!/usr/bin/env python3
"""
GAK-WaveCAD — Coupling Monitor Test
Тест кросс-связей между модулями.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.logger import get_logger
from physical_modules.coupling_monitor import CouplingMonitor


def make_mock_results():
    """Создать мок-результаты для всех модулей."""
    return {
        'osmosis': {
            'sigma_Pa': 500.0,
            'pi_Pa': 1230.0,
        },
        'thermal': {
            'sigma_thermal_Pa': 300.0,
        },
        'acoustic': {
            'breathing_l0': {'delta_f': 1.5},
            'flexural_l1': {'delta_f': 0.8},
            'flexural_l2': {'delta_f': 0.5},
            'flexural_l3': {'delta_f': 0.3},
        },
        'collapse': {
            'phase': 'STABLE',
            'ratio': 0.1,
        },
        'magnon': {
            'kittel_l0': {'f': 2.8e9, 'df_stress': 1.0e5},
            'exchange_l1': {'f': 3.1e9, 'df_stress': 1.2e5},
            'exchange_l2': {'f': 3.5e9, 'df_stress': 1.5e5},
            'exchange_l3': {'f': 4.0e9, 'df_stress': 1.8e5},
        },
        'em': {
            'TM_l1_n1': {'f_shifted': 8.5e9, 'df_stress': 2.0e6},
            'TE_l1_n1': {'f_shifted': 1.2e10, 'df_stress': 3.0e6},
            'TM_l2_n1': {'f_shifted': 1.5e10, 'df_stress': 4.0e6},
            'TE_l2_n1': {'f_shifted': 1.9e10, 'df_stress': 5.0e6},
        },
    }


def main():
    logger = get_logger("coupling_test")

    print("=" * 64)
    print("  GAK-WaveCAD — Coupling Monitor Test")
    print("=" * 64)

    mock = make_mock_results()

    # Тест 1: нормальная система
    monitor = CouplingMonitor()
    monitor.init()
    monitor.set_results(
        osmosis=mock['osmosis'],
        thermal=mock['thermal'],
        acoustic=mock['acoustic'],
        collapse=mock['collapse'],
        magnon=mock['magnon'],
        em=mock['em'],
    )
    monitor.run()
    results = monitor.get_results()

    monitor.print_report()

    print("\n  Проверки:")
    assert results['stability_index'] > 0, "Stability должна быть положительной"
    assert 0 <= results['stability_index'] <= 1, "Stability в диапазоне 0..1"
    assert results['system_status'] in ("HEALTHY", "DEGRADED", "CRITICAL"), "Статус определён"
    assert results['geom_factor'] > 0, "Геометрия устойчива"
    logger.info("Тест 1 пройден: нормальная система")
    print("    ✅ Тест 1: нормальная система — OK")

    # Тест 2: коллапс
    mock['collapse']['phase'] = 'COLLAPSE'
    mock['collapse']['ratio'] = 1.2
    monitor.set_results(
        osmosis=mock['osmosis'],
        thermal=mock['thermal'],
        acoustic=mock['acoustic'],
        collapse=mock['collapse'],
        magnon=mock['magnon'],
        em=mock['em'],
    )
    monitor.run()
    results2 = monitor.get_results()

    print("\n  С коллапсом:")
    monitor.print_report()

    assert results2['geom_factor'] == 0, "При коллапсе geom=0"
    assert results2['stability_index'] < results['stability_index'], "Stability упала"
    logger.info("Тест 2 пройден: коллапс снижает stability")
    print("    ✅ Тест 2: коллапс снижает stability — OK")

    # Тест 3: все связи ненулевые
    mock['collapse']['phase'] = 'STABLE'
    mock['collapse']['ratio'] = 0.1
    monitor.set_results(
        osmosis=mock['osmosis'],
        thermal=mock['thermal'],
        acoustic=mock['acoustic'],
        collapse=mock['collapse'],
        magnon=mock['magnon'],
        em=mock['em'],
    )
    monitor.run()
    results3 = monitor.get_results()
    assert results3['k_magnetoelastic'] > 0, "Магнитоупругая связь > 0"
    assert results3['k_piezoelectric'] > 0, "Пьезосвязь > 0"
    assert results3['k_thermal_osmosis'] > 0, "Тепло-осмос связь > 0"
    logger.info("Тест 3 пройден: все кросс-связи ненулевые")
    print("    ✅ Тест 3: все кросс-связи ненулевые — OK")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
