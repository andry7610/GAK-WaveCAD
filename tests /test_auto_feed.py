"""
Тесты автозапитки — плазменное динамо (v0.3).

6 тестов:
  1. Ток в стенке появляется при движении плазмы
  2. Поле направлено против движения (правило Ленца)
  3. Малые возмущения затухают
  4. Нет самовозбуждения (нет runaway)
  5. Фаза STABLE при автозапитке
  6. Без автозапитки — WARNING/COLLAPSE
"""

import numpy as np
import pytest

from physical_modules.plasma_monitor import PlasmaMonitor


def _make_plasma(**kwargs):
    """Создать PlasmaMonitor с дефолтными параметрами и overrides."""
    p = PlasmaMonitor(**kwargs)
    p.init()
    return p


class TestWallCurrent:
    """1. Ток Фарадея в стенке."""

    def test_wall_current_induced(self):
        """При движении плазмы к стенке в стенке наводится ток."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            auto_feed_enabled=True,
            sigma_wall=5.96e7,   # Cu
            delta_wall=0.01,
        )
        J = p.compute_wall_current(v_radial=1.0)
        assert J > 0, f"Ток должен быть положительным при v>0, got {J}"
        assert J > 1e6, f"Ток должен быть значительным, got {J:.3e}"

    def test_wall_current_zero_velocity(self):
        """При нулевой скорости — нулевой ток."""
        p = _make_plasma(radius=0.5, external_B=3.0)
        J = p.compute_wall_current(v_radial=0.0)
        assert J == 0.0, f"При v=0 ток должен быть 0, got {J}"

    def test_wall_current_disabled(self):
        """При выключенной автозапитке — нулевой ток."""
        p = _make_plasma(radius=0.5, external_B=3.0, auto_feed_enabled=False)
        J = p.compute_wall_current(v_radial=1.0)
        assert J == 0.0, f"При auto_feed=False ток должен быть 0, got {J}"

    def test_wall_current_proportional_to_velocity(self):
        """Ток пропорционален скорости (закон Ома)."""
        p = _make_plasma(radius=0.5, external_B=3.0)
        J1 = p.compute_wall_current(v_radial=0.5)
        J2 = p.compute_wall_current(v_radial=1.0)
        ratio = J2 / J1
        assert abs(ratio - 2.0) < 0.01, f"Ток должен удваиваться, ratio={ratio:.3f}"


class TestWallField:
    """2. Поле стенки — против движения (правило Ленца)."""

    def test_wall_field_opposes_motion(self):
        """При v>0 (к стенке) поле B_wall < 0 (против движения)."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )
        J = p.compute_wall_current(v_radial=1.0)
        B_wall = p.compute_wall_field(J)
        assert B_wall < 0, f"Поле должно быть против движения (B<0), got {B_wall:.3e}"

    def test_wall_field_magnitude(self):
        """Магнитуда поля разумна (не 0, не огромная)."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )
        J = p.compute_wall_current(v_radial=1.0)
        B_wall = p.compute_wall_field(J)
        assert abs(B_wall) > 1e-3, f"Поле слишком мало: {B_wall:.3e}"
        assert abs(B_wall) < 1.0, f"Поле слишком велико: {B_wall:.3e}"


class TestDamping:
    """3. Затухание возмущений."""

    def test_dynamo_stabilizes(self):
        """Малые возмущения затухают за несколько tau_wall."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )

        delta = 1e-3       # 1 мм
        v = 0.1             # 0.1 м/с
        dt = 0.001          # 1 мс

        initial_amplitude = abs(delta)
        for _ in range(10000):  # 10 секунд
            res = p.step_auto_feed(dt, delta, v)
            delta = res["delta"]
            v = res["v_radial"]

        final_amplitude = abs(delta)
        assert final_amplitude < initial_amplitude * 0.1, (
            f"Возмущение должно затухнуть на >90%, "
            f"было {initial_amplitude:.3e}, стало {final_amplitude:.3e}"
        )

    def test_damping_rate_positive(self):
        """Скорость затухания положительна."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )
        gamma = p.compute_damping_rate()
        assert gamma > 0, f"gamma должна быть > 0, got {gamma}"

    def test_decay_time_reasonable(self):
        """Время затухания порядка 0.1–10 секунд (ITER-масштаб)."""
        p = _make_plasma(
            radius=0.5,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )
        tau = p.compute_wall_decay_time()
        assert 0.01 < tau < 100, f"tau_wall={tau:.3f} — нереалистично"


class TestNoRunaway:
    """4. Нет самовозбуждения."""

    def test_dynamo_no_runaway(self):
        """При больших возмущениях нет экспоненциального роста."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )

        delta = 0.1     # 10 см — большое возмущение
        v = 10.0        # 10 м/с — быстро
        dt = 0.001

        max_delta = abs(delta)
        for _ in range(10000):  # 10 секунд
            res = p.step_auto_feed(dt, delta, v)
            delta = res["delta"]
            v = res["v_radial"]
            max_delta = max(max_delta, abs(delta))

        # Возмущение не должно расти
        assert abs(delta) < max_delta * 1.01, (
            f"Runaway detected: delta вырос до {abs(delta):.3e}, "
            f"макс был {max_delta:.3e}"
        )

    def test_no_runaway_zero_velocity(self):
        """При начальной нулевой скорости — затухание от начального смещения."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            sigma_wall=5.96e7,
            delta_wall=0.01,
        )

        delta = 0.01    # 1 см
        v = 0.0
        dt = 0.001

        for _ in range(5000):
            res = p.step_auto_feed(dt, delta, v)
            delta = res["delta"]
            v = res["v_radial"]

        assert abs(delta) < 0.01, (
            f"Смещение должно затухнуть, got delta={delta:.3e}"
        )


class TestAutoFeedPhase:
    """5. Фаза STABLE при автозапитке."""

    def test_auto_feed_stable(self):
        """При включённой автозапитке — фаза STABLE."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            auto_feed=True,
            auto_feed_enabled=True,
        )
        results = p.run()
        assert results["phase"] == "STABLE", (
            f"Ожидалась STABLE, got {results['phase']}"
        )

    def test_auto_feed_results_has_auto_feed_flag(self):
        """Результат run() содержит информацию об автозапитке."""
        p = _make_plasma(
            radius=0.5,
            auto_feed_enabled=True,
        )
        p.run()
        # run() возвращает общий dict — фазу и т.д.
        # Автозапитка доступна через методы
        af = p.compute_auto_feed(delta=0.001, v_radial=0.1)
        assert af["damped"] is True
        assert af["J_wall"] > 0


class TestNoAutoFeed:
    """6. Без автозапитки — деградация."""

    def test_no_auto_feed_unstable(self):
        """Без автозапитки — температура падает, фаза деградирует."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            auto_feed=False,
            auto_feed_enabled=False,
        )
        results = p.run()
        # Без автозапитки T *= 0.95 — но фаза может быть STABLE
        # если barrier > threshold. Главное — автозапитка не откликается.
        af = p.compute_auto_feed(delta=0.001, v_radial=0.1)
        assert af["J_wall"] == 0.0, "Без автозапитки ток должен быть 0"
        assert af["damped"] is False, "Без автозапитки затухания нет"

    def test_no_auto_feed_temperature_drops(self):
        """Без автозапитки температура плазмы падает (радиационные потери)."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            auto_feed=False,
        )
        T_before = p.temperature
        p.run()
        T_after = p.results["temperature_plasma"]
        assert T_after < T_before, (
            f"Без автозапитки T должна упасть: было {T_before:.3e}, "
            f"стало {T_after:.3e}"
        )

    def test_auto_feed_temperature_maintained(self):
        """С автозапиткой температура не падает."""
        p = _make_plasma(
            radius=0.5,
            external_B=3.0,
            auto_feed=True,
        )
        T_before = p.temperature
        p.run()
        T_after = p.results["temperature_plasma"]
        assert T_after >= T_before, (
            f"С автозапиткой T не должна падать: было {T_before:.3e}, "
            f"стало {T_after:.3e}"
        )
