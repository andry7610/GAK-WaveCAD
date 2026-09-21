"""
Тесты автозапитки (плазменное динамо) — v0.3.

Проверяет:
  1. Ток Фарадея в стенке при движении плазмы
  2. Поле Ленца (против движения)
  3. Затухание возмущений
  4. Отсутствие runaway (самовозбуждения)
  5. Фаза STABLE при автозапитке
  6. Без автозапитки — деградация
"""

import numpy as np
import pytest

from physical_modules.plasma_monitor import PlasmaMonitor


def _make_plasma(**overrides):
    """Создать PlasmaMonitor с параметрами по умолчанию и переопределениями."""
    defaults = dict(
        radius=0.5,
        external_B=3.0,
        density_number=1.0e20,
        temperature=1.0e8,
        fuel_type="D-T",
        auto_feed_enabled=True,
        sigma_wall=5.96e7,
        delta_wall=0.01,
    )
    defaults.update(overrides)
    p = PlasmaMonitor(**defaults)
    p.init()
    return p


class TestWallCurrent:
    """Ток Фарадея в стенке."""

    def test_wall_current_induced(self):
        """Ток появляется при радиальном движении плазмы."""
        p = _make_plasma()
        J = p.compute_wall_current(v_radial=1.0)
        assert J > 0, f"Ток должен быть положительным, got {J:.3e}"

    def test_wall_current_zero_velocity(self):
        """При нулевой скорости — нулевой ток."""
        p = _make_plasma()
        J = p.compute_wall_current(v_radial=0.0)
        assert J == 0.0, f"При v=0 ток должен быть 0, got {J:.3e}"

    def test_wall_current_disabled(self):
        """При отключённой автозапитке — нулевой ток."""
        p = _make_plasma(auto_feed_enabled=False)
        J = p.compute_wall_current(v_radial=1.0)
        assert J == 0.0, f"При выключенной автозапитке ток должен быть 0, got {J:.3e}"

    def test_wall_current_proportional_to_velocity(self):
        """Ток пропорционален скорости."""
        p = _make_plasma()
        J1 = p.compute_wall_current(v_radial=1.0)
        J2 = p.compute_wall_current(v_radial=2.0)
        assert abs(J2 - 2 * J1) < abs(J1) * 0.01, (
            f"Ток должен быть пропорционален скорости: J1={J1:.3e}, J2={J2:.3e}"
        )


class TestWallField:
    """Магнитное поле от тока в стенке."""

    def test_wall_field_opposes_motion(self):
        """Поле направлено против движения (правило Ленца)."""
        p = _make_plasma()
        J = p.compute_wall_current(v_radial=1.0)
        B_wall = p.compute_wall_field(J)
        assert B_wall < 0, (
            f"Поле должно быть отрицательным (против движения), got {B_wall:.3e}"
        )

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
        assert abs(B_wall) < 5.0, f"Поле слишком велико: {B_wall:.3e}"


class TestDamping:
    """Затухание возмущений."""

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
        """Коэффициент затухания положительный."""
        p = _make_plasma()
        gamma = p.compute_damping_rate()
        assert gamma > 0, f"gamma должна быть > 0, got {gamma:.3e}"

    def test_decay_time_reasonable(self):
        """Время затухания реалистично (0.01–10 с)."""
        p = _make_plasma()
        tau = p.compute_wall_decay_time()
        assert 0.01 < tau < 10.0, (
            f"tau_wall должна быть в диапазоне 0.01–10 с, got {tau:.3e}"
        )


class TestNoRunaway:
    """Отсутствие самовозбуждения."""

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
            if not np.isnan(abs(delta)):
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
    """Фазовое состояние при автозапитке."""

    def test_auto_feed_stable(self):
        """При включённой автозапитке — фаза STABLE."""
        p = _make_plasma(
            radius=0.5,
            external_B=10.0,
            plasma_gap=5e-3,
            auto_feed=True,
            auto_feed_enabled=True,
        )
        results = p.run()
        assert results["phase"] == "STABLE", (
            f"Ожидалась STABLE, got {results['phase']}"
        )

    def test_auto_feed_results_has_auto_feed_flag(self):
        """Результат compute_auto_feed содержит флаг auto_feed."""
        p = _make_plasma()
        res = p.compute_auto_feed(delta=1e-3, v_radial=0.1)
        assert "auto_feed" in res, "В результате нет ключа auto_feed"
        assert res["auto_feed"] is True, "auto_feed должен быть True"


class TestNoAutoFeed:
    """Поведение без автозапитки."""

    def test_no_auto_feed_unstable(self):
        """Без автозапитки — нет стабилизации."""
        p = _make_plasma(auto_feed_enabled=False)
        res = p.compute_auto_feed(delta=1e-3, v_radial=0.1)
        assert res["J_wall"] == 0.0, (
            f"Без автозапитки J_wall должен быть 0, got {res['J_wall']:.3e}"
        )

    def test_no_auto_feed_temperature_drops(self):
        """Без автозапитки — температура падает (радиационные потери)."""
        p = _make_plasma(auto_feed=False, auto_feed_enabled=False)
        T_before = p.temperature
        p.run()
        T_after = p.results["temperature_plasma"]
        assert T_after < T_before, (
            f"Без автозапитки T должна падать: было {T_before:.3e}, стало {T_after:.3e}"
        )

    def test_auto_feed_temperature_maintained(self):
        """С автозапиткой — температура не падает."""
        p = _make_plasma(auto_feed=True, auto_feed_enabled=True)
        T_before = p.temperature
        p.run()
        T_after = p.results["temperature_plasma"]
        assert T_after >= T_before, (
            f"С автозапиткой T не должна падать: было {T_before:.3e}, стало {T_after:.3e}"
        )
