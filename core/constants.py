"""
GAK Core Constants — CODATA 2018
Единый источник физических констант для всех модулей.
"""

import math

# ── Фундаментальные константы (CODATA 2018) ──
HBAR = 1.054571817e-34      # приведённая постоянная Планка, Дж·с
C_LIGHT = 299792458.0       # скорость света, м/с (точное значение)
E_CHARGE = 1.602176634e-19  # элементарный заряд, Кл (точное значение)
M_ELECTRON = 9.1093837015e-31  # масса электрона, кг
PI = math.pi

# ── Производные константы ──
# Критическое поле Швингера: E_c = m_e²·c³ / (e·ℏ)
SCHWINGER_FIELD = (M_ELECTRON ** 2) * (C_LIGHT ** 3) / (E_CHARGE * HBAR)

# Комптоновская длина волны электрона: λ_c = ℏ / (m_e·c)
COMPTON_WAVELENGTH = HBAR / (M_ELECTRON * C_LIGHT)

# Энергия покоя электрона: E_0 = m_e·c²
ELECTRON_REST_ENERGY = M_ELECTRON * C_LIGHT ** 2
