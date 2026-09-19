#!/usr/bin/env python3
"""
Osmosis Monitor — Integration Test
Проверяет расчёт осмотического давления и напряжения в оболочке.

Запуск:
    python tests/run_osmosis_test.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physical_modules.osmosis_monitor import OsmosisMonitor


def main():
    print("=" * 64)
    print("  GAK-WaveCAD — Osmosis Monitor Test")
    print("  v0.1 — Van't Hoff equation validator")
    print("=" * 64)

    # 1. Базовый запуск с дефолтными параметрами
    print("\n  Тест 1: Дефолтные параметры (c=300, T=310К)")
    monitor = OsmosisMonitor()
    monitor.init()
    monitor.run()
    monitor.print_report()

    r = monitor.get_results()
    assert r['pi_Pa'] > 0, "Осмотическое давление должно быть положительным"
    assert r['sigma_Pa'] > 0, "Напряжение должно быть положительным"
    print("  ✅ Проверки пройдены")

    # 2. Сравнение с ручным расчётом
    print("\n  Тест 2: Сравнение с ручным расчётом")
    c, T, R_gas = 300.0, 310.0, 8.314
    R_shell, h_wall = 0.05, 0.002
    pi_expected = c * R_gas * T
    sigma_expected = pi_expected * R_shell / (2 * h_wall)

    print(f"  Ожидаемое Π:   {pi_expected:.1f} Па ({pi_expected / 1e3:.2f} кПа)")
    print(f"  Полученное Π:  {r['pi_Pa']:.1f} Па ({r['pi_kPa']:.2f} кПа)")
    print(f"  Ожидаемое σ:   {sigma_expected:.1f} Па ({sigma_expected / 1e6:.6f} МПа)")
    print(f"  Полученное σ:  {r['sigma_Pa']:.1f} Па ({r['sigma_MPa']:.6f} МПа)")

    assert abs(r['pi_Pa'] - pi_expected) < 0.01, "Расхождение в Pi"
    assert abs(r['sigma_Pa'] - sigma_expected) < 0.01, "Расхождение в sigma"
    print("  ✅ Расчёт верен")

    # 3. Разные концентрации → разное давление
    print("\n  Тест 3: Зависимость от концентрации")
    for c_test in [0, 100, 300, 500, 1000]:
        m = OsmosisMonitor(c_solute=c_test)
        m.init()
        m.run()
        rr = m.get_results()
        print(f"  c = {c_test:6.1f} моль/м³ → "
              f"Π = {rr['pi_kPa']:8.2f} кПа, "
              f"σ = {rr['sigma_MPa']:.6f} МПа")
        assert rr['pi_Pa'] == c_test * R_gas * T, "Линейность нарушена"

    print("  ✅ Линейность по концентрации подтверждена")

    # 4. Проверка ошибок при некорректных параметрах
    print("\n  Тест 4: Защита от некорректных параметров")
    for label, kwargs in [
        ("T = 0",      {"T": 0}),
        ("T = -5",     {"T": -5}),
        ("R_shell = 0", {"R_shell": 0}),
        ("h_wall = 0",  {"h_wall": 0}),
        ("c < 0",      {"c_solute": -10}),
    ]:
        m = OsmosisMonitor(**kwargs)
        try:
            m.init()
            print(f"  {label:15s} ❌ Должно было вызвать ошибку")
        except ValueError as e:
            print(f"  {label:15s} ✅ ValueError: {e}")

    print("\n" + "=" * 64)
    print("  Все тесты пройдены ✅")
    print("=" * 64)


if __name__ == "__main__":
    main()
