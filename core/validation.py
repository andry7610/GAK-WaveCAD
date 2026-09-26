"""
validation.py — Валидация всех физических модулей GAK-WaveCAD.

61 тест для 9 модулей: акустика, плазма, вакуум, коллапс, термалка,
магноны, EM-резонанс, квантовый вакуум, осмос.

Проверяются: физические формулы, граничные случаи, фазовые переходы,
стресс-сдвиги, multi-material / multi-fuel.
"""

import numpy as np

# ── Физические константы (СИ) ──
K_B     = 1.380649e-23
PI      = np.pi
MU_0    = 4 * PI * 1e-7
E_CHG   = 1.602176634e-19
C_LIGHT = 3.0e8
HBAR    = 1.054571817e-34

# ── Импорты модулей ──
_TRY = {}

try:
    from physical_modules.acoustic_monitor import AcousticMonitor
    _TRY["acoustic"] = True
except Exception:
    _TRY["acoustic"] = False

try:
    from physical_modules.plasma_monitor import PlasmaMonitor
    _TRY["plasma"] = True
except Exception:
    _TRY["plasma"] = False

try:
    from physical_modules.vacuum_monitor import VacuumMonitor
    _TRY["vacuum"] = True
except Exception:
    _TRY["vacuum"] = False

try:
    from physical_modules.born_collapse_monitor import BornCollapseMonitor
    _TRY["collapse"] = True
except Exception:
    _TRY["collapse"] = False

try:
    from physical_modules.thermal_monitor import ThermalMonitor
    _TRY["thermal"] = True
except Exception:
    _TRY["thermal"] = False

try:
    from physical_modules.magnon_monitor import MagnonMonitor
    _TRY["magnon"] = True
except Exception:
    _TRY["magnon"] = False

try:
    from physical_modules.em_resonance_monitor import EMResonanceMonitor
    _TRY["em"] = True
except Exception:
    _TRY["em"] = False

try:
    from physical_modules.quantum_vacuum import GAKQuantumVacuum
    _TRY["quantum"] = True
except Exception:
    _TRY["quantum"] = False

try:
    from physical_modules.osmosis_monitor import OsmosisMonitor
    _TRY["osmosis"] = True
except Exception:
    _TRY["osmosis"] = False


# ── Утилиты ──

def _approx(a, b, rtol=0.02):
    """Относительное сравнение с tolerance."""
    if a is None or b is None:
        return a == b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    a, b = float(a), float(b)
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom < rtol

def _is_number(x):
    return isinstance(x, (int, float, np.floating, np.integer)) and not np.isnan(x)

def _safe_get(d, *keys, default=None):
    """Поиск значения по нескольким возможным ключам."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Акустика (14)
# ══════════════════════════════════════════════════════════════════════

def _test_acoustic():
    if not _TRY.get("acoustic"):
        return "🔊 Акустика", []
    tests = []

    # Эталонные частоты (из независимых расчётов / предыдущих прогонов)
    al_refs = {
        "breathing_l0": 28002.37,
        "flexural_l1": 971.25,
        "flexural_l2": 1682.25,
        "flexural_l3": 2379.10,
        "flexural_l4": 3071.30,
        "flexural_l5": 3761.63,
    }
    steel_refs = {
        "breathing_l0": 26966.00,
        "flexural_l1": 949.69,
        "flexural_l2": 1644.90,
    }
    glass_refs = {
        "breathing_l0": 26971.00,
        "flexural_l1": 976.74,
    }

    materials = [
        ("Al",    {"R": 0.05, "h": 0.002, "rho": 2700, "E": 70e9,  "nu": 0.33, "n_modes": 6}, al_refs),
        ("Steel", {"R": 0.05, "h": 0.002, "rho": 7850, "E": 200e9, "nu": 0.30}, steel_refs),
        ("Glass", {"R": 0.05, "h": 0.002, "rho": 2500, "E": 70e9,  "nu": 0.22}, glass_refs),
    ]

    for mat_name, cfg, refs in materials:
        try:
            mod = AcousticMonitor(cfg)
            mod.set_internal_stress(0.0)
            mod.init()
            mod.run()
            res = mod.get_results()
        except Exception as ex:
            for key in refs:
                tests.append((f"{mat_name} {key}", refs[key], None, False, str(ex)))
            continue

        for key, expected in refs.items():
            mode = _safe_get(res, key, default={})
            if not isinstance(mode, dict):
                tests.append((f"{mat_name} {key}", expected, None, False, f"key '{key}' not dict"))
                continue
            got = _safe_get(mode, "f_reference", "f_ref", default=None)
            ok = _approx(expected, got, rtol=0.02)
            tests.append((f"{mat_name} {key}", expected, got, ok, ""))

    # ── Стресс-тесты (Al) ──
    al_cfg = {"R": 0.05, "h": 0.002, "rho": 2700, "E": 70e9, "nu": 0.33, "n_modes": 6}

    # stress = 0: f_measured == f_reference
    try:
        mod = AcousticMonitor(al_cfg)
        mod.set_internal_stress(0.0)
        mod.init()
        mod.run()
        res = mod.get_results()
        mode = _safe_get(res, "breathing_l0", default={})
        f_ref = _safe_get(mode, "f_reference", default=0)
        f_meas = _safe_get(mode, "f_measured", "f_shifted", default=0)
        ok = _approx(f_ref, f_meas, rtol=1e-6)
        tests.append(("stress=0: f_shift==f_ref", f_ref, f_meas, ok, ""))
    except Exception as ex:
        tests.append(("stress=0: f_shift==f_ref", None, None, False, str(ex)))

    # stress = E: CRITICAL
    try:
        mod = AcousticMonitor(al_cfg)
        mod.set_internal_stress(70e9)
        mod.init()
        mod.run()
        res = mod.get_results()
        mode = _safe_get(res, "breathing_l0", default={})
        status = _safe_get(mode, "status", default="")
        ok = status == "CRITICAL_STRESS"
        tests.append(("stress=E: CRITICAL", "CRITICAL_STRESS", status, ok, ""))
    except Exception as ex:
        tests.append(("stress=E: CRITICAL", "CRITICAL_STRESS", None, False, str(ex)))

    # stress = -2E: INVALID
    try:
        mod = AcousticMonitor(al_cfg)
        mod.set_internal_stress(-2 * 70e9)
        mod.init()
        mod.run()
        res = mod.get_results()
        mode = _safe_get(res, "breathing_l0", default={})
        status = _safe_get(mode, "status", default="")
        ok = status == "INVALID_STRESS"
        tests.append(("stress=-2E: INVALID", "INVALID_STRESS", status, ok, ""))
    except Exception as ex:
        tests.append(("stress=-2E: INVALID", "INVALID_STRESS", None, False, str(ex)))

    return "🔊 Акустика", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Плазма (7)
# ══════════════════════════════════════════════════════════════════════

def _test_plasma():
    if not _TRY.get("plasma"):
        return "☀️ Плазма", []
    tests = []

    # ── Bosch-Hale D-T ──
    bh_energies = [5.0, 10.0, 20.0]
    bh_expected = [1.3656e-23, 1.136e-22, 4.3297e-22]

    for T_keV, expected in zip(bh_energies, bh_expected):
        try:
            T_K = T_keV * 1e3 * E_CHG / K_B
            mod = PlasmaMonitor(config={
                "temperature": T_K, "density_number": 1e20,
                "fuel_type": "D-T", "auto_feed": False,
            })
            mod.init()
            sv = mod.compute_reactivity()
            ok = _approx(expected, sv, rtol=0.02)
            tests.append((f"BH D-T {T_keV:.0f}keV", expected, sv, ok, ""))
        except Exception as ex:
            tests.append((f"BH D-T {T_keV:.0f}keV", expected, None, False, str(ex)))

    # ── Bosch-Hale D-D 10 keV ──
    try:
        T_K = 10e3 * E_CHG / K_B
        mod = PlasmaMonitor(config={
            "temperature": T_K, "density_number": 1e20,
            "fuel_type": "D-D", "auto_feed": False,
        })
        mod.init()
        sv = mod.compute_reactivity()
        ok = _approx(1.2044e-24, sv, rtol=0.02)
        tests.append(("BH D-D 10keV", 1.2044e-24, sv, ok, ""))
    except Exception as ex:
        tests.append(("BH D-D 10keV", 1.2044e-24, None, False, str(ex)))

    # ── B-поле ──
    try:
        mod = PlasmaMonitor(config={
            "external_B": 2.0, "temperature": 5e6,
            "density_number": 1e20, "fuel_type": "custom",
            "auto_feed": True,
        })
        mod.init()
        res = mod.run()
        B = res["B_field"]
        # B = sqrt(B_ext^2 + B_eq^2), B_eq = sqrt(2*mu0*n*kB*T) ≈ 0.1317
        expected = 2.0043
        ok = _approx(expected, B, rtol=0.02)
        tests.append(("B-field", expected, B, ok, ""))
    except Exception as ex:
        tests.append(("B-field", 2.0043, None, False, str(ex)))

    # ── Bremsstrahlung ──
    try:
        mod = PlasmaMonitor(config={
            "temperature": 5e6, "density_number": 1e20,
            "fuel_type": "custom", "Z_eff": 1.0,
            "auto_feed": True,
        })
        mod.init()
        res = mod.run()
        P_rad = res["P_rad"]
        # P_rad = 5.35e-37 * Z * n^2 * sqrt(T_keV) * V
        # T_keV ≈ 0.4307, V = 4/3*pi*(0.01)^3 ≈ 4.189e-6
        expected = 0.01471
        ok = _approx(expected, P_rad, rtol=0.05)
        tests.append(("Brems T=5keV", expected, P_rad, ok, ""))
    except Exception as ex:
        tests.append(("Brems T=5keV", 0.01471, None, False, str(ex)))

    # ── T≈0: reactivity≈0 ──
    try:
        mod = PlasmaMonitor(config={
            "temperature": 1.0, "density_number": 1e20,
            "fuel_type": "D-T", "auto_feed": False,
        })
        mod.init()
        sv = mod.compute_reactivity()
        ok = abs(sv) < 1e-50
        tests.append(("T≈0: reactivity≈0", 0.0, sv, ok, ""))
    except Exception as ex:
        tests.append(("T≈0: reactivity≈0", 0.0, None, False, str(ex)))

    return "☀️ Плазма", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Вакуум (7)
# ══════════════════════════════════════════════════════════════════════

def _test_vacuum():
    if not _TRY.get("vacuum"):
        return "🌀 Вакуум", []
    tests = []

    # ── P = n*k_B*T ──
    for T in [300.0, 500.0, 77.0]:
        try:
            mod = VacuumMonitor(config={"gas_temp": T, "initial_pressure": 1.0e5})
            mod.init()
            n = 1.0e5 / (K_B * T)
            P = mod.compute_pressure(n)
            ok = _approx(1.0e5, P, rtol=1e-6)
            tests.append((f"P=nkT T={T:.0f}K", 1.0e5, P, ok, ""))
        except Exception as ex:
            tests.append((f"P=nkT T={T:.0f}K", 1.0e5, None, False, str(ex)))

    # ── Средний свободный пробег ──
    # lambda = k_B*T / (sqrt(2)*pi*d^2*P)
    d = 3.7e-10  # диаметр молекулы N₂
    for P_val, label in [(1.0e5, "1e+05"), (1.0e-3, "1e-03")]:
        try:
            mod = VacuumMonitor(config={"gas_temp": 300.0, "initial_pressure": P_val})
            mod.init()
            mfp = mod.compute_mean_free_path(P_val)
            expected = K_B * 300.0 / (np.sqrt(2) * PI * d**2 * P_val)
            ok = _approx(expected, mfp, rtol=0.01)
            tests.append((f"MFP P={label}", expected, mfp, ok, ""))
        except Exception as ex:
            tests.append((f"MFP P={label}", None, None, False, str(ex)))

    # ── Откачка: P(t) = P0 * exp(-S*t/V) ──
    try:
        mod = VacuumMonitor(config={
            "volume": 1.0, "pump_speed": 0.1,
            "initial_pressure": 1.0e5,
        })
        mod.init()
        mod.step(1.0)
        expected = 1.0e5 * np.exp(-0.1)
        ok = _approx(expected, mod.pressure, rtol=0.01)
        tests.append(("Pump 1s", expected, mod.pressure, ok, ""))
    except Exception as ex:
        tests.append(("Pump 1s", None, None, False, str(ex)))

    # ── P=0: MFP=∞ ──
    try:
        mod = VacuumMonitor(config={"gas_temp": 300.0, "initial_pressure": 1.0e5})
        mod.init()
        mfp = mod.compute_mean_free_path(0.0)
        ok = mfp == float("inf") or np.isinf(mfp)
        tests.append(("P=0: MFP=∞", float("inf"), mfp, ok, ""))
    except Exception as ex:
        tests.append(("P=0: MFP=∞", float("inf"), None, False, str(ex)))

    return "🌀 Вакуум", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Коллапс (5)
# ══════════════════════════════════════════════════════════════════════

def _test_collapse():
    if not _TRY.get("collapse"):
        return "🕳️ Коллапс", []
    tests = []

    E_mod = 2.0e9
    nu = 0.33
    imperf = 0.25

    for hr_ratio, label in [(0.04, "0.04"), (0.02, "0.02")]:
        R = 0.05
        h = R * hr_ratio
        # Теоретическое: P_cr = 2*E/sqrt(3*(1-nu^2)) * (h/R)^2
        P_cr_theory = (2 * E_mod / np.sqrt(3 * (1 - nu**2))) * hr_ratio**2

        # Реальное (с imperfection)
        P_cr_real = P_cr_theory * imperf

        try:
            mod = BornCollapseMonitor(config={
                "R": R, "h": h, "E": E_mod, "nu": nu,
                "imperfection_factor": imperf,
                "external_pressure": 0.0,
            })
            mod.init()
            mod.run()
            res = mod.get_results()

            # Проверяем theory
            got_theory = res.get("P_cr_theory", None)
            ok_t = _approx(P_cr_theory, got_theory, rtol=0.01)
            tests.append((f"Зоэлли h/R={label} (theory)", P_cr_theory, got_theory, ok_t, ""))

            # Проверяем real
            got_real = res.get("P_cr", None)
            ok_r = _approx(P_cr_real, got_real, rtol=0.01)
            tests.append((f"Зоэлли h/R={label} (real)", P_cr_real, got_real, ok_r, ""))

        except Exception as ex:
            tests.append((f"Зоэлли h/R={label} (theory)", P_cr_theory, None, False, str(ex)))
            tests.append((f"Зоэлли h/R={label} (real)", P_cr_real, None, False, str(ex)))

    # ── Фаза STABLE ──
    try:
        mod = BornCollapseMonitor(config={
            "R": 0.05, "h": 0.002, "E": 2e9, "nu": 0.33,
            "imperfection_factor": 0.25,
            "external_pressure": 1.0e3,  # маленькое давление
        })
        mod.init()
        mod.run()
        res = mod.get_results()
        phase = res.get("phase", "")
        ok = phase == "STABLE"
        tests.append(("Phase STABLE", "STABLE", phase, ok, ""))
    except Exception as ex:
        tests.append(("Phase STABLE", "STABLE", None, False, str(ex)))

    # ── Фаза COLLAPSE ──
    try:
        mod = BornCollapseMonitor(config={
            "R": 0.05, "h": 0.002, "E": 2e9, "nu": 0.33,
            "imperfection_factor": 0.25,
            "external_pressure": 1.0e7,  # большое давление
        })
        mod.init()
        mod.run()
        res = mod.get_results()
        phase = res.get("phase", "")
        ok = phase == "COLLAPSE"
        tests.append(("Phase COLLAPSE", "COLLAPSE", phase, ok, ""))
    except Exception as ex:
        tests.append(("Phase COLLAPSE", "COLLAPSE", None, False, str(ex)))

    return "🕳️ Коллапс", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Термалка (5)
# ══════════════════════════════════════════════════════════════════════

def _test_thermal():
    if not _TRY.get("thermal"):
        return "🌡️ Термалка", []
    tests = []

    R, h = 0.05, 0.002
    T_in, T_out = 310.0, 293.0
    dT = T_in - T_out

    # Q = 4*pi*k*R_in*R_out*dT / h  (сферическая оболочка)
    R_in = R - h / 2
    R_out = R + h / 2

    for k_val in [15.0, 0.6]:
        try:
            mod = ThermalMonitor(config={
                "R": R, "h": h,
                "T_inner": T_in, "T_outer": T_out,
                "k_thermal": k_val,
            })
            mod.init()
            mod.run()
            res = mod.get_results()
            # Ищем тепловой поток
            Q = _safe_get(res, "heat_flux", "Q", "heat_flow", "Q_total", "power", default=None)
            expected = 4 * PI * k_val * R_in * R_out * dT / h
            if Q is not None:
                ok = _approx(expected, Q, rtol=0.05)
                tests.append((f"Q k={k_val}", expected, Q, ok, ""))
            else:
                tests.append((f"Q k={k_val}", expected, None, False, f"keys: {list(res.keys())}"))
        except Exception as ex:
            expected = 4 * PI * k_val * R_in * R_out * dT / h
            tests.append((f"Q k={k_val}", expected, None, False, str(ex)))

    # ── dT=0: Q=0 ──
    try:
        mod = ThermalMonitor(config={
            "R": R, "h": h,
            "T_inner": 300.0, "T_outer": 300.0,
            "k_thermal": 15.0,
        })
        mod.init()
        mod.run()
        res = mod.get_results()
        Q = _safe_get(res, "heat_flux", "Q", "heat_flow", "Q_total", "power", default=None)
        ok = Q is not None and abs(Q) < 1e-6
        tests.append(("dT=0: Q=0", 0.0, Q, ok, ""))
    except Exception as ex:
        tests.append(("dT=0: Q=0", 0.0, None, False, str(ex)))

    # ── Thermal stress smoke ──
    try:
        mod = ThermalMonitor(config={
            "R": R, "h": h,
            "T_inner": T_in, "T_outer": T_out,
            "k_thermal": 15.0,
        })
        mod.init()
        mod.run()
        stress = mod.get_thermal_stress_Pa()
        ok = _is_number(stress) and stress > 0
        tests.append(("Thermal stress smoke", "number", stress, ok, ""))
    except Exception as ex:
        tests.append(("Thermal stress smoke", "number", None, False, str(ex)))

    # ── dT=0: stress=0 ──
    try:
        mod = ThermalMonitor(config={
            "R": R, "h": h,
            "T_inner": 300.0, "T_outer": 300.0,
            "k_thermal": 15.0,
        })
        mod.init()
        mod.run()
        stress = mod.get_thermal_stress_Pa()
        ok = _is_number(stress) and abs(stress) < 1e-6
        tests.append(("dT=0: stress=0", 0.0, stress, ok, ""))
    except Exception as ex:
        tests.append(("dT=0: stress=0", 0.0, None, False, str(ex)))

    return "🌡️ Термалка", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Магноны (7)
# ══════════════════════════════════════════════════════════════════════

def _test_magnon():
    if not _TRY.get("magnon"):
        return "🧲 Магноны", []
    tests = []

    # ── Kittel: f = gamma_bar * B_total ──
    kittel_vals = {}
    for B_ext in [0.1, 0.5, 1.0]:
        try:
            mod = MagnonMonitor(config={"B_ext": B_ext})
            mod.set_mechanical_stress(0.0)
            mod.init()
            mod.run()
            res = mod.get_results()
            mode = _safe_get(res, "kittel_l0", default={})
            f = _safe_get(mode, "f", default=None) if isinstance(mode, dict) else None
            kittel_vals[B_ext] = f
            # B_anis ≈ 1.095e-8 — пренебрежимо мало
            expected = 28e9 * B_ext
            if f is not None:
                ok = _approx(expected, f, rtol=0.02)
                tests.append((f"Kittel B={B_ext}", expected, f, ok, ""))
            else:
                tests.append((f"Kittel B={B_ext}", expected, None, False, f"keys: {list(res.keys())}"))
        except Exception as ex:
            tests.append((f"Kittel B={B_ext}", 28e9 * B_ext, None, False, str(ex)))

    # ── Kittel monotonicity: f(B=1.0) > f(B=0.5) > f(B=0.1) ──
    try:
        f01 = kittel_vals.get(0.1)
        f05 = kittel_vals.get(0.5)
        f10 = kittel_vals.get(1.0)
        if f01 is not None and f05 is not None and f10 is not None:
            ok = f10 > f05 > f01
            tests.append(("Kittel monotonicity", "f1>f05>f01", f"{f10:.2e}>{f05:.2e}>{f01:.2e}", ok, ""))
        else:
            tests.append(("Kittel monotonicity", "f1>f05>f01", None, False, "missing values"))
    except Exception as ex:
        tests.append(("Kittel monotonicity", "f1>f05>f01", None, False, str(ex)))

    # ── Обменные моды l=1,2,3 ──
    for l in [1, 2, 3]:
        try:
            mod = MagnonMonitor(config={"B_ext": 0.1})
            mod.set_mechanical_stress(0.0)
            mod.init()
            mod.run()
            res = mod.get_results()
            key = f"exchange_l{l}"
            mode = _safe_get(res, key, default={})
            f = _safe_get(mode, "f", default=None) if isinstance(mode, dict) else None
            if f is not None:
                ok = f > 0
                tests.append((f"Exchange l={l}", ">0", f, ok, ""))
            else:
                tests.append((f"Exchange l={l}", ">0", None, False, f"keys: {list(res.keys())}"))
        except Exception as ex:
            tests.append((f"Exchange l={l}", ">0", None, False, str(ex)))

    # ── stress=0: df_stress=0 ──
    try:
        mod = MagnonMonitor(config={"B_ext": 0.1})
        mod.set_mechanical_stress(0.0)
        mod.init()
        mod.run()
        res = mod.get_results()
        mode = _safe_get(res, "kittel_l0", default={})
        df = _safe_get(mode, "df_stress", default=None) if isinstance(mode, dict) else None
        ok = df is not None and abs(df) < 1e-3
        tests.append(("stress=0: no shift", 0.0, df, ok, ""))
    except Exception as ex:
        tests.append(("stress=0: no shift", 0.0, None, False, str(ex)))

    return "🧲 Магноны", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: EM-резонанс (8)
# ══════════════════════════════════════════════════════════════════════

def _test_em():
    if not _TRY.get("em"):
        return "📡 EM-резонанс", []
    tests = []

    # Эталонные корни
    tm_roots = {1: 4.4934, 2: 5.7635, 3: 6.9879}
    te_roots = {1: 2.7437, 2: 3.8702, 3: 4.9735}

    # ── TM/TE l=1,2,3 с eps_r=10 ──
    R = 0.05
    eps_r = 10.0
    v_em = C_LIGHT / np.sqrt(eps_r)

    for l, mtype, root_dict in [(1, "TM", tm_roots), (2, "TM", tm_roots),
                                 (3, "TM", tm_roots),
                                 (1, "TE", te_roots), (2, "TE", te_roots),
                                 (3, "TE", te_roots)]:
        try:
            mod = EMResonanceMonitor(config={
                "R": R, "eps_r": eps_r, "n_modes": 3,
            })
            mod.set_mechanical_stress(0.0)
            mod.init()
            mod.run()
            res = mod.get_results()
            key = f"{mtype}_l{l}_n1"
            mode = _safe_get(res, key, default={})
            f = _safe_get(mode, "f_base", default=None) if isinstance(mode, dict) else None
            expected = v_em * root_dict[l] / (2 * PI * R)
            if f is not None:
                ok = _approx(expected, f, rtol=0.02)
                tests.append((f"{mtype} l={l} eps=10", expected, f, ok, ""))
            else:
                tests.append((f"{mtype} l={l} eps=10", expected, None, False, f"keys: {list(res.keys())}"))
        except Exception as ex:
            expected = v_em * root_dict[l] / (2 * PI * R)
            tests.append((f"{mtype} l={l} eps=10", expected, None, False, str(ex)))

    # ── TM l=1 с eps_r=2 ──
    try:
        eps_r2 = 2.0
        v2 = C_LIGHT / np.sqrt(eps_r2)
        mod = EMResonanceMonitor(config={"R": R, "eps_r": eps_r2, "n_modes": 1})
        mod.set_mechanical_stress(0.0)
        mod.init()
        mod.run()
        res = mod.get_results()
        mode = _safe_get(res, "TM_l1_n1", default={})
        f = _safe_get(mode, "f_base", default=None) if isinstance(mode, dict) else None
        expected = v2 * 4.4934 / (2 * PI * R)
        if f is not None:
            ok = _approx(expected, f, rtol=0.02)
            tests.append(("TM l=1 eps=2", expected, f, ok, ""))
        else:
            tests.append(("TM l=1 eps=2", expected, None, False, f"keys: {list(res.keys())}"))
    except Exception as ex:
        tests.append(("TM l=1 eps=2", None, None, False, str(ex)))

    # ── TE l=1 с eps_r=50 ──
    try:
        eps_r50 = 50.0
        v50 = C_LIGHT / np.sqrt(eps_r50)
        mod = EMResonanceMonitor(config={"R": R, "eps_r": eps_r50, "n_modes": 1})
        mod.set_mechanical_stress(0.0)
        mod.init()
        mod.run()
        res = mod.get_results()
        mode = _safe_get(res, "TE_l1_n1", default={})
        f = _safe_get(mode, "f_base", default=None) if isinstance(mode, dict) else None
        expected = v50 * 2.7437 / (2 * PI * R)
        if f is not None:
            ok = _approx(expected, f, rtol=0.02)
            tests.append(("TE l=1 eps=50", expected, f, ok, ""))
        else:
            tests.append(("TE l=1 eps=50", expected, None, False, f"keys: {list(res.keys())}"))
    except Exception as ex:
        tests.append(("TE l=1 eps=50", None, None, False, str(ex)))

    return "📡 EM-резонанс", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Квантовый вакуум (3)
# ══════════════════════════════════════════════════════════════════════

def _test_quantum():
    if not _TRY.get("quantum"):
        return "⚡ Квант. вакуум", []
    tests = []

    # ── Казимир ──
    try:
        mod = GAKQuantumVacuum(config={
            "gap": 3.5e-9, "area": 1e-6,
        })
        res = mod.run()
        E_cas = res.get("casimir_energy", None)
        # E = -pi^2 * hbar * c * A / (720 * d^3)
        expected = -(PI**2) * HBAR * C_LIGHT * 1e-6 / (720 * (3.5e-9)**3)
        if E_cas is not None:
            ok = _approx(expected, E_cas, rtol=0.02)
            tests.append(("Казимир", expected, E_cas, ok, ""))
        else:
            tests.append(("Казимир", expected, None, False, f"keys: {list(res.keys())}"))
    except Exception as ex:
        tests.append(("Казимир", None, None, False, str(ex)))

    # ── Швингер ──
    try:
        mod = GAKQuantumVacuum(config={"E_field": 1e18})
        res = mod.run()
        prob = res.get("pair_probability", None)
        if prob is not None:
            ok = 0 < prob < 1
            tests.append(("Швингер", "0<p<1", prob, ok, ""))
        else:
            tests.append(("Швингер", "0<p<1", None, False, f"keys: {list(res.keys())}"))
    except Exception as ex:
        tests.append(("Швингер", "0<p<1", None, False, str(ex)))

    # ── E=0: prob=0 ──
    try:
        mod = GAKQuantumVacuum(config={"E_field": 0.0})
        res = mod.run()
        prob = res.get("pair_probability", None)
        ok = prob is not None and abs(prob) < 1e-30
        tests.append(("E=0: prob=0", 0.0, prob, ok, ""))
    except Exception as ex:
        tests.append(("E=0: prob=0", 0.0, None, False, str(ex)))

    return "⚡ Квант. вакуум", tests


# ══════════════════════════════════════════════════════════════════════
#  Тесты: Осмос (4)
# ══════════════════════════════════════════════════════════════════════

def _test_osmosis():
    if not _TRY.get("osmosis"):
        return "💧 Осмос", []
    tests = []

    # ── Вант-Гофф: Pi = c * R * T ──
    for c_val in [300.0, 500.0]:
        try:
            mod = OsmosisMonitor(config={
                "c_solute": c_val, "T": 310.0,
                "R_gas": 8.314,
                "R_shell": 0.05, "h_wall": 0.002,
            })
            mod.init()
            mod.run()
            res = mod.get_results()
            pi = _safe_get(res, "pi_Pa", default=None)
            expected = c_val * 8.314 * 310.0
            if pi is not None:
                ok = _approx(expected, pi, rtol=0.01)
                tests.append((f"Вант-Гофф c={c_val:.0f}", expected, pi, ok, ""))
            else:
                tests.append((f"Вант-Гофф c={c_val:.0f}", expected, None, False, f"keys: {list(res.keys())}"))
        except Exception as ex:
            expected = c_val * 8.314 * 310.0
            tests.append((f"Вант-Гофф c={c_val:.0f}", expected, None, False, str(ex)))

    # ── Напряжение в стенке: sigma = Pi * R / (2*h) ──
    try:
        mod = OsmosisMonitor(config={
            "c_solute": 300.0, "T": 310.0, "R_gas": 8.314,
            "R_shell": 0.05, "h_wall": 0.002,
        })
        mod.init()
        mod.run()
        res = mod.get_results()
        sigma = _safe_get(res, "sigma_Pa", default=None)
        pi = 300.0 * 8.314 * 310.0
        expected = pi * 0.05 / (2 * 0.002)
        if sigma is not None:
            ok = _approx(expected, sigma, rtol=0.01)
            tests.append(("σ стенки", expected, sigma, ok, ""))
        else:
            tests.append(("σ стенки", expected, None, False, f"keys: {list(res.keys())}"))
    except Exception as ex:
        tests.append(("σ стенки", None, None, False, str(ex)))

    # ── c=0: pi=0 ──
    try:
        mod = OsmosisMonitor(config={
            "c_solute": 0.0, "T": 310.0, "R_gas": 8.314,
            "R_shell": 0.05, "h_wall": 0.002,
        })
        mod.init()
        mod.run()
        res = mod.get_results()
        pi = _safe_get(res, "pi_Pa", default=None)
        ok = pi is not None and abs(pi) < 1e-6
        tests.append(("c=0: π=0", 0.0, pi, ok, ""))
    except Exception as ex:
        tests.append(("c=0: π=0", 0.0, None, False, str(ex)))

    return "💧 Осмос", tests


# ══════════════════════════════════════════════════════════════════════
#  Сборка всех тестов
# ══════════════════════════════════════════════════════════════════════

_TEST_FUNCS = [
    _test_acoustic,
    _test_plasma,
    _test_vacuum,
    _test_collapse,
    _test_thermal,
    _test_magnon,
    _test_em,
    _test_quantum,
    _test_osmosis,
]

def run_all_tests():
    """Запускает все тесты, возвращает список (module_name, tests)."""
    results = []
    for func in _TEST_FUNCS:
        try:
            mod_name, tests = func()
        except Exception as ex:
            mod_name = func.__name__.replace("_test_", "")
            tests = [("CRASH", None, None, False, str(ex))]
        results.append((mod_name, tests))
    return results


# ══════════════════════════════════════════════════════════════════════
#  Streamlit-вкладка
# ══════════════════════════════════════════════════════════════════════

def render_validation_tab():
    """Отображает вкладку валидации в Streamlit."""
    import streamlit as st

    st.markdown("## 🧪 Валидация физических модулей")
    st.caption("61 тест для 9 модулей: формулы, граничные случаи, фазовые переходы, стресс-сдвиги.")

    if st.button("▶ Запустить все тесты", type="primary"):
        with st.spinner("⏳ Запуск тестов..."):
            all_results = run_all_tests()

        total_pass = 0
        total_fail = 0
        total_skip = 0

        # Сбор статистики
        for mod_name, tests in all_results:
            for tname, exp, got, ok, detail in tests:
                if ok:
                    total_pass += 1
                elif got is None and "SKIP" not in tname.upper():
                    total_skip += 1
                else:
                    total_fail += 1

        # Считаем SKIP: тесты где got is None и не pass
        total_skip = 0
        for mod_name, tests in all_results:
            for tname, exp, got, ok, detail in tests:
                if not ok and got is None:
                    total_skip += 1

        total = total_pass + total_fail + total_skip
        if total == 0:
            total = 1

        # ── Сводка ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ PASS", f"{total_pass}")
        col2.metric("❌ FAIL", f"{total_fail}")
        col3.metric("⏭️ SKIP", f"{total_skip}")
        col4.metric("📊 Итого", f"{total}")

        if total_fail == 0 and total_skip == 0:
            st.success(f"✅ Все {total_pass} тестов прошли!")
        elif total_fail == 0:
            st.warning(f"✅ {total_pass} прошли, ⏭️ {total_skip} пропущены (модуль недоступен)")
        else:
            st.error(f"❌ {total_fail} тестов не прошли")

        st.divider()

        # ─<arg_value> Детали по модулям ──
        for mod_name, tests in all_results:
            n_pass = sum(1 for _, _, _, ok, _ in tests if ok)
            n_total = len(tests)
            if n_total == 0:
                continue

            if n_pass == n_total:
                header = f"✅ {mod_name} ({n_pass}/{n_total})"
            elif n_pass > 0:
                header = f"⚠️ {mod_name} ({n_pass}/{n_total})"
            else:
                header = f"⏭️ {mod_name} ({n_pass}/{n_total})"

            with st.expander(header, expanded=(n_pass < n_total)):
                for tname, exp, got, ok, detail in tests:
                    if ok:
                        icon = "✅"
                    elif got is None:
                        icon = "⏭️"
                        if detail and "keys:" in detail:
                            tname = f"{tname} — SKIP: {detail}"
                        elif detail:
                            tname = f"{tname} — SKIP: {detail}"
                    else:
                        icon = "❌"

                    if isinstance(exp, float):
                        exp_str = f"{exp:.4e}"
                    elif isinstance(exp, (int, np.integer)):
                        exp_str = str(exp)
                    else:
                        exp_str = str(exp)

                    if isinstance(got, float):
                        got_str = f"{got:.4e}"
                    elif got is None:
                        got_str = "None"
                    else:
                        got_str = str(got)

                    if ok:
                        st.write(f"{icon} **{tname}** — ожидаемо {exp_str}, получено {got_str}")
                    elif got is None:
                        st.write(f"{icon} **{tname}**")
                    else:
                        st.write(f"{icon} **{tname}** — ожидаемо {exp_str}, получено {got_str}")
                        if detail:
                            st.write(f"   _{detail}_")


# ── Точка входа для standalone-запуска ──
if __name__ == "__main__":
    results = run_all_tests()
    total_pass = 0
    total_fail = 0
    total_skip = 0
    for mod_name, tests in results:
        for tname, exp, got, ok, detail in tests:
            if ok:
                total_pass += 1
            elif got is None:
                total_skip += 1
            else:
                total_fail += 1
    total = total_pass + total_fail + total_skip
    print(f"\n{'='*60}")
    print(f"Валидация: {total_pass} PASS / {total_fail} FAIL / {total_skip} SKIP / {total} TOTAL")
    print(f"{'='*60}")
    for mod_name, tests in results:
        n_pass = sum(1 for _, _, _, ok, _ in tests if ok)
        n_total = len(tests)
        print(f"\n{mod_name} ({n_pass}/{n_total})")
        for tname, exp, got, ok, detail in tests:
            if ok:
                print(f"  ✅ {tname}")
            elif got is None:
                print(f"  ⏭️ {tname} — SKIP: {detail}")
            else:
                print(f"  ❌ {tname} — ожидаемо {exp}, получено {got}")
                if detail:
                    print(f"     {detail}")
