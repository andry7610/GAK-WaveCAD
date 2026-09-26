"""
Тесты для GAK Quantum Vacuum Module
19 тестов: Казимир, Швингер, нулевые колебания, интеграция.
"""

import math
import pytest
from core.constants import (
    HBAR, C_LIGHT, E_CHARGE, M_ELECTRON, PI,
    SCHWINGER_FIELD, COMPTON_WAVELENGTH, ELECTRON_REST_ENERGY,
)
from physical_modules.quantum_vacuum import GAKQuantumVacuum


# ── Фикстуры ──

@pytest.fixture
def vacuum_default():
    """Вакуум по умолчанию: gap=3.5нм, area=1мкм²."""
    return GAKQuantumVacuum({'gap': 3.5e-9, 'area': 1e-6, 'E_field': 0.0})

@pytest.fixture
def vacuum_strong_field():
    """Вакуум с сильным полем: 0.1·E_c."""
    return GAKQuantumVacuum({
        'gap': 3.5e-9, 'area': 1e-6,
        'E_field': 0.1 * SCHWINGER_FIELD
    })

@pytest.fixture
def vacuum_lab_field():
    """Вакуум с лабораторным полем: 1e6 В/м."""
    return GAKQuantumVacuum({
        'gap': 3.5e-9, 'area': 1e-6,
        'E_field': 1e6
    })


# ── Тесты констант ──

class TestConstants:
    def test_schwinger_field_value(self):
        """E_c ≈ 1.32×10¹⁸ В/м."""
        assert abs(SCHWINGER_FIELD - 1.32e18) / 1.32e18 < 0.01

    def test_schwinger_field_formula(self):
        """E_c = m²c³/(eℏ) — проверка формулы."""
        expected = M_ELECTRON**2 * C_LIGHT**3 / (E_CHARGE * HBAR)
        assert abs(SCHWINGER_FIELD - expected) / expected < 1e-10

    def test_compton_wavelength(self):
        """λ_c ≈ 3.86×10⁻¹³ м."""
        assert abs(COMPTON_WAVELENGTH - 3.86e-13) / 3.86e-13 < 0.01

    def test_electron_rest_energy(self):
        """E_0 = m_e·c² ≈ 8.19×10⁻¹⁴ Дж."""
        assert abs(ELECTRON_REST_ENERGY - 8.19e-14) / 8.19e-14 < 0.01


# ── Тесты Казимира ──

class TestCasimir:
    def test_energy_negative(self, vacuum_default):
        """Энергия Казимира — притяжение (отрицательная)."""
        assert vacuum_default.casimir_energy() < 0

    def test_force_negative(self, vacuum_default):
        """Сила Казимира — притяжение (отрицательная)."""
        assert vacuum_default.casimir_force() < 0

    def test_energy_scaling_1_over_d3(self, vacuum_default):
        """E ~ 1/d³ — при удвоении зазора энергия уменьшается в 8 раз."""
        E1 = abs(vacuum_default.casimir_energy(gap=2e-9))
        E2 = abs(vacuum_default.casimir_energy(gap=4e-9))
        ratio = E1 / E2
        assert abs(ratio - 8.0) / 8.0 < 0.01

    def test_force_scaling_1_over_d4(self, vacuum_default):
        """F ~ 1/d⁴ — при удвоении зазора сила уменьшается в 16 раз."""
        F1 = abs(vacuum_default.casimir_force(gap=2e-9))
        F2 = abs(vacuum_default.casimir_force(gap=4e-9))
        ratio = F1 / F2
        assert abs(ratio - 16.0) / 16.0 < 0.01

    def test_pressure_independent_of_area(self, vacuum_default):
        """Давление Казимира не зависит от площади."""
        v1 = GAKQuantumVacuum({'gap': 3.5e-9, 'area': 1e-6})
        v2 = GAKQuantumVacuum({'gap': 3.5e-9, 'area': 1e-3})
        assert abs(v1.casimir_pressure() - v2.casimir_pressure()) < 1e-30

    def test_energy_at_35nm(self, vacuum_default):
        """Энергия при gap=3.5 нм ≈ -1.01×10⁻²⁰ Дж."""
        E = vacuum_default.casimir_energy()
        expected = -(PI**2) * HBAR * C_LIGHT * 1e-6 / (720 * (3.5e-9)**3)
        assert abs(E - expected) / abs(expected) < 1e-10

    def test_force_at_35nm(self, vacuum_default):
        """Сила при gap=3.5 нм."""
        F = vacuum_default.casimir_force()
        expected = -(PI**2) * HBAR * C_LIGHT * 1e-6 / (240 * (3.5e-9)**4)
        assert abs(F - expected) / abs(expected) < 1e-10

    def test_zero_gap_raises(self, vacuum_default):
        """Нулевой зазор — ошибка."""
        with pytest.raises(ValueError):
            vacuum_default.casimir_energy(gap=0)

    def test_negative_gap_raises(self, vacuum_default):
        """Отрицательный зазор — ошибка."""
        with pytest.raises(ValueError):
            vacuum_default.casimir_force(gap=-1e-9)


# ── Тесты Швингера ──

class TestSchwinger:
    def test_critical_field_value(self, vacuum_default):
        """E_c ≈ 1.32×10¹⁸ В/м."""
        Ec = vacuum_default.schwinger_critical_field()
        assert abs(Ec - SCHWINGER_FIELD) < 1.0

    def test_pair_probability_zero_field(self, vacuum_default):
        """При E=0 вероятность = 0."""
        assert vacuum_default.pair_probability(E_field=0.0) == 0.0

    def test_pair_probability_negligible_lab(self, vacuum_lab_field):
        """При лабораторном поле 1e6 В/м вероятность ≈ 0."""
        W = vacuum_lab_field.pair_probability()
        assert W < 1e-100

    def test_pair_probability_visible_strong(self, vacuum_strong_field):
        """При E=0.1·E_c вероятность измеримо отлична от нуля."""
        W = vacuum_strong_field.pair_probability()
        expected = math.exp(-PI * 10)  # exp(-10π)
        assert W > 1e-15
        assert abs(W - expected) / expected < 1e-10

    def test_pair_rate_zero_field(self, vacuum_default):
        """При E=0 скорость пар = 0."""
        assert vacuum_default.pair_rate(E_field=0.0) == 0.0

    def test_pair_rate_lab_field(self, vacuum_lab_field):
        """При лабораторном поле скорость ≈ 0."""
        gamma = vacuum_lab_field.pair_rate()
        assert gamma < 1e-50

    def test_pair_rate_strong_field(self, vacuum_strong_field):
        """При E=0.1·E_c скорость пар отлична от нуля."""
        gamma = vacuum_strong_field.pair_rate()
        assert gamma > 0
        assert gamma < 1e12  # разумная верхняя граница


# ── Тесты нулевых колебаний ──

class TestZeroPoint:
    def test_amplitude_positive(self, vacuum_default):
        """Амплитуда нулевых колебаний положительна."""
        assert vacuum_default.zero_point_amplitude(freq=1e9) > 0

    def test_amplitude_increases_with_freq(self, vacuum_default):
        """При росте частоты амплитуда растёт."""
        a1 = vacuum_default.zero_point_amplitude(freq=1e9)
        a2 = vacuum_default.zero_point_amplitude(freq=1e12)
        assert a2 > a1


# ── Тесты step() и run() ──

class TestStepRun:
    def test_step_accumulates_energy(self, vacuum_default):
        """step() накапливает энергию Казимира."""
        for _ in range(100):
            vacuum_default.step(dt=1e-9)
        assert vacuum_default._total_casimir_energy != 0
        assert vacuum_default._step_count == 100

    def test_run_returns_all_keys(self, vacuum_default):
        """run() возвращает все ключи."""
        result = vacuum_default.run()
        required_keys = [
            'casimir_energy', 'casimir_force', 'casimir_pressure',
            'schwinger_field', 'E_field', 'pair_probability',
            'pair_rate', 'zero_point_amplitude',
            'total_casimir_energy', 'total_pairs_created', 'step_count'
        ]
        for key in required_keys:
            assert key in result, f"Отсутствует ключ: {key}"

    def test_repr_shows_params(self, vacuum_default):
        """__repr__ показывает параметры."""
        r = repr(vacuum_default)
        assert 'GAKQuantumVacuum' in r
        assert 'gap' in r
