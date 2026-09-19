#!/usr/bin/env python3
"""
GAK-WaveCAD — Integration Test
Проверяет связку: osmosis_monitor → acoustic_monitor.

Запуск:
    python tests/test_integration.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.osmosis_monitor import OsmosisMonitor
from physical_modules.acoustic_monitor import AcousticMonitor


def main():
    print("=" * 64)
    print("  GAK-WaveCAD — Integration Test")
    print("  osmosis_monitor → acoustic_monitor")
    print("=" * 64)

    # 1. Запуск без осмотического давления (свободная оболочка)
    print("\n  Фаза 1: Свободная оболочка (без напряжения)")
    acm_free = AcousticMonitor()
    acm_free.init()
    acm_free.set_internal_stress(0.0)
    acm_free.run()
    results_free = acm_free.get_results()

    for key, r in results_free.items():
        print(f"  {key:12s} | {r['status']:16s} | f = {r['f_measured']:.3f} Гц")

    # 2. Запуск осмотического монитора
    print("\n  Фаза 2: Расчёт осмотического давления")
    osm = OsmosisMonitor()
    osm.init()
    osm.run()
    osm.print_report()
    sigma_Pa = osm.get_stress_Pa()
    print(f"  → Передаём в акустический монитор: {sigma_Pa:.1f} Па")

    # 3. Запуск акустического монитора с напряжением
    print("\n  Фаза 3: Оболочка под осмотическим напряжением")
    acm_stressed = AcousticMonitor()
    acm_stressed.init()
    acm_stressed.set_internal_stress(sigma_Pa)
    acm_stressed.run()
    results_stressed = acm_stressed.get_results()

    for key, r in results_stressed.items():
        print(f"  {key:12s} | {r['status']:16s} | f = {r['f_measured']:.3f} Гц | "
              f"σ = {r['sigma_MPa']:.4f} МПа")

    # 4. Сравнение: частоты должны сдвинуться
    print("\n  Фаза 4: Сравнение частот")
    print("  " + "-" * 60)
    shifted = 0
    for key in results_free:
        f_free = results_free[key]['f_measured']
        f_stressed = results_stressed[key]['f_measured']
        delta = f_stressed - f_free
        status_changed = results_free[key]['status'] != results_stressed[key]['status']
        marker = " ⚠" if abs(delta) > 0 else ""
        print(f"  {key:12s} | свободная: {f_free:.3f} Гц → "
              f"напряжённая: {f_stressed:.3f} Гц | Δf = {delta:+.3f} Гц{marker}")
        if abs(delta) > 0:
            shifted += 1

    print("  " + "-" * 60)

    # 5. Итоговые проверки
    print("\n  Итоги:")
    free_count = sum(1 for r in results_free.values()
                     if r['status'] == 'FREE_OR_DAMPED')
    stressed_count = sum(1 for r in results_stressed.values()
                         if r['status'] == 'CRITICAL_STRESS')

    print(f"  Свободных мод (фаза 1):   {free_count} {'✅' if free_count > 0 else '❌'}")
    print(f"  Напряжённых мод (фаза 3): {stressed_count} {'✅' if stressed_count > 0 else '❌'}")
    print(f"  Сдвинутых частот:         {shifted} / {len(results_free)}")

    assert free_count > 0, "Без напряжения должны быть свободные моды"
    assert shifted > 0, "При напряжении частоты должны сдвинуться"
    print("\n  ✅ Интеграционный тест пройден")

    print("=" * 64)


if __name__ == "__main__":
    main()
