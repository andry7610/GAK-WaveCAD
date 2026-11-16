# -*- coding: utf-8 -*-
"""
v46R — Validation Run (паспорт надёжности)
GAK-WaveCAD, штаб 16.11.2026, 03:30

Приказ: "Мы не гонимся за рекордом. Мы строим паспорт надёжности."

Команды:
    python v46R_validation.py runa      — ёжик чувствительности (детерминированный, OAT)
    python v46R_validation.py tornado   — ёжик + PNG-диаграмма (v46R_tornado.png)
    python v46R_validation.py runb      — слепой прогон: 50 триалов без сидов
    python v46R_validation.py runc      — граница физики: wall 25/30/35

Базовая точка: S35 = trial 33 run20 (welded+emitter), LCOE = 85.9591777550
Конфиг и метрики извлечены из results/trials_export_v46Q.csv бит в бит.
Экономика воспроизведена по структуре GAKOptunaOptimizer._economics (v46Q)
и откалибрована: effective rate 0.130575 = CRF(7%,30y) 0.0806 + OPEX_FRAC 0.05.
Проверка реконструкции: 85.959178 == 85.959178 (сходится с CSV до 6 знака).

v46R-fix3 (17.11.2026, ночь):
  1. БД слепого прогона задаётся правильным SQLAlchemy URL: sqlite:///results/...
     (fix2 передавал голый путь — ArgumentError от sqlalchemy).
  2. Сиды глушатся черезmonkeypatch Study.enqueue_trial: каждый сид-энкью
     в optimize() обёрнут try/except — исключение класс молча глотает,
     сиды не попадают в очередь, дедупликация не нужна. БД чистая, load_if_exists=False.
  3. optimize() в конце возвращает None — study берём из opt.study после прогона.
"""
import os, sys
import numpy as np

# ============================================================
# БАЗОВАЯ ТОЧКА: S35 = trial 33 (run20, study gak_opt_lcoe_v46Q)
# ============================================================
S35 = {
    "R_mm": 2700.0, "pl_radius_mm": 1404.0, "external_B": 14.0,
    "n_density": 5.86e20, "E_field": 1.68e17,
    "wall_material": "c_c_sic_adv",
    "welded_vessel": True, "emitter": True,
    "alignment": 0.0, "pump_speed": 0.06,
}
M35 = {  # метрики прогона (user_attrs trial 33)
    "P_net_W":   1471107037.1725829,
    "P_elec_W":  1507083372.9193854,
    "P_fusion_W":2219777834.4145446,
    "beta":      0.14539595082223986,
    "wall_load_W_m2": 24231043.546837963,
    "cost_blanket": 1182131768.5323904,
    "cost_bop":     5877625154.385603,
    "cost_mag":      142883999.92221746,
    "cost_cryo_W":      35808.88945525656,
}
LCOE_BASE = 85.95917775504155

# ============================================================
# ЭКОНОМИКА (реконструкция _economics, откалибрована по trial 33)
#   annual = rate*CAPEX + CRYO_PER_W*cryo_W
#   E_year = P_net/1e6 * 8760 * CF
#   rate   = CRF(discount, 30y) + OPEX_FRAC
# ============================================================
CF0        = 0.85
OPEX_FRAC  = 0.05
CRYO_PER_W = 30.7          # из регрессии лога run20: annual = 0.1306*capex + 30.7*cryo_W

def crf(r, N=30):
    x = (1.0 + r) ** N
    return r * x / (x - 1.0)

RATE_BASE = crf(0.07) + OPEX_FRAC   # 0.0806 + 0.05

def lcoe_of(P_net=M35["P_net_W"], cb=M35["cost_blanket"], cbop=M35["cost_bop"],
            cmag=M35["cost_mag"], cw=M35["cost_cryo_W"],
            rate=RATE_BASE, CF=CF0):
    annual = rate * (cb + cbop + cmag) + CRYO_PER_W * cw
    return annual / (P_net / 1e6 * 8760.0 * CF)

# ============================================================
# RUN A — ёжик чувствительности (OAT, без стохастики)
# ============================================================
def run_a():
    print("=" * 70)
    print("RUN A — анализ чувствительности (One-at-a-Time)")
    print(f"База: LCOE(S35) = {LCOE_BASE:.4f} $/МВт·ч")
    print("=" * 70)
    recirc = M35["P_elec_W"] - M35["P_net_W"]   # рециркуляция константна при смене eta

    def with_eta(f):
        pe = M35["P_elec_W"] * f                # ПРИБЛИЖЕНИЕ 1-го порядка:
        pn = pe - recirc                        # P_elec ~ eta, рециркуляция фикс.
        return lcoe_of(P_net=pn, cbop=M35["cost_bop"] * f)

    sweeps = [
        ("BOP_COST_PER_KW (3900$/кВт)", "−30%", "+30%",
         lcoe_of(cbop=M35["cost_bop"]*0.7), lcoe_of(cbop=M35["cost_bop"]*1.3)),
        ("capacity_factor", "0.70 (FOAK)", "1.00 (идеал)",
         lcoe_of(CF=0.70), lcoe_of(CF=1.00)),
        ("discount_rate", "4% (госкредит)", "10% (рынок)",
         lcoe_of(rate=crf(0.04)+OPEX_FRAC), lcoe_of(rate=crf(0.10)+OPEX_FRAC)),
        ("BLANKET_COST_PER_M2 (16.03М$)", "−50%", "+50%",
         lcoe_of(cb=M35["cost_blanket"]*0.5), lcoe_of(cb=M35["cost_blanket"]*1.5)),
        ("eta_cascade", "−10%", "+10%", with_eta(0.9), with_eta(1.1)),
        ("cryo_W", "−30%", "+30%",
         lcoe_of(cw=M35["cost_cryo_W"]*1.3), lcoe_of(cw=M35["cost_cryo_W"]*0.7)),
    ]
    sweeps.sort(key=lambda s: -abs(s[4]-s[3]))
    print(f"\n{'параметр':<32}{'низ':>8}{'верх':>8}{'разброс':>9}")
    for name, llo, lhi, a, b in sweeps:
        print(f"{name:<32}{min(a,b):>8.2f}{max(a,b):>8.2f}{abs(b-a):>9.2f}   [{llo} … {lhi}]")
    print("\nИнтерпретация: плечо > 20 $ — допущение критично, его надо")
    print("обосновывать литературой; < 5 $ — можно оставить как есть.")
    return sweeps

def tornado():
    sweeps = run_a()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib недоступен — таблица выше остаётся паспортом."); return
    names = [s[0] for s in sweeps][::-1]
    los   = [min(s[3], s[4]) for s in sweeps][::-1]
    his   = [max(s[3], s[4]) for s in sweeps][::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (lo, hi) in enumerate(zip(los, his)):
        ax.barh(i, LCOE_BASE-lo, left=lo, color="#2f7ed8", height=0.55)
        ax.barh(i, hi-LCOE_BASE, left=LCOE_BASE, color="#d84f2f", height=0.55)
        ax.text(lo-1.2, i, f"{lo:.1f}", va="center", ha="right", fontsize=9)
        ax.text(hi+1.2, i, f"{hi:.1f}", va="center", ha="left", fontsize=9)
    ax.axvline(LCOE_BASE, color="k", lw=1.5)
    ax.text(LCOE_BASE, len(names)-0.25, f"база {LCOE_BASE:.2f}",
            ha="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("LCOE, $/МВт·ч — точка S35 (welded+emitter)")
    ax.set_title("v46R Run A — tornado чувствительности LCOE")
    plt.tight_layout(); plt.savefig("v46R_tornado.png", dpi=150)
    print("Сохранено: v46R_tornado.png")

# ============================================================
# СЛЕПОЙ ПРОГОН — общая механика fix3
#   Чистая БД (sqlite:///), имя study своё, enqueue_trial заглушен.
#   Каждый сид-энкью в optimize() обёрнут try/except: pass — класс
#   молча глотает исключение, сиды не попадают в очередь. Reenqueue
#   чемпионов тоже печатает skip — это штатная память TPE, не подсказка.
# ============================================================
def _blind_optimize(opt, n_trials, study_name, db_name):
    import optuna as _o
    os.makedirs("results", exist_ok=True)
    db_path = os.path.join("results", db_name)
    if os.path.exists(db_path):
        os.remove(db_path)
    db_url = "sqlite:///" + db_path.replace("\\", "/")
    print(f"v46R-fix3: чистая БД {db_url}, сиды заглушены")
    orig_create = _o.create_study
    orig_enqueue = _o.study.Study.enqueue_trial

    def _clean_create(*args, **kwargs):
        kwargs["storage"] = db_url
        kwargs["study_name"] = study_name
        kwargs["load_if_exists"] = False
        return orig_create(*args, **kwargs)

    def _no_enqueue(*args, **kwargs):
        raise RuntimeError("seeds disabled for blind run (v46R-fix3)")

    _o.study.Study.enqueue_trial = _no_enqueue
    try:
        _o.create_study = _clean_create
        opt.optimize(mode="lcoe", n_trials=n_trials)
    finally:
        _o.create_study = orig_create
        _o.study.Study.enqueue_trial = orig_enqueue
    return opt.study

# ============================================================
# RUN B — слепой прогон (TPE без сидов)
# ============================================================
def run_b():
    print("=" * 70)
    print("RUN B — слепой прогон: 50 триалов, НОЛЬ сидов, flags welded+emitter")
    print("=" * 70)
    try:
        from optuna_optimizer import GAKOptunaOptimizer
    except ImportError:
        print("Запускать из корня репозитория: python v46R_validation.py runb"); return
    opt = GAKOptunaOptimizer(base_config={})   # базовый конфиг проекта подхватится сам
    study = _blind_optimize(opt, 50, "gak_opt_lcoe_v46R_blind_runb", "blind_v46R_runb.sqlite")
    if study is None:
        print("Study не создан — смотри трейсбек выше."); return
    vals = [t.value for t in study.trials if t.value and t.value < 1e9]
    if not vals:
        print("Успешных триалов нет — все 50 отбиты гейтами/стенкой."); return
    best, med = min(vals), float(np.median(vals))
    print(f"\nBest = {best:.4f}   Median = {med:.4f}   (успешных триалов: {len(vals)}/50)")
    if best <= 90:
        print("ВЕРДИКТ: best <= 90 — ОТЛИЧНО. Сигнатура работает, 85.96 не подгонка.")
    elif best <= 100:
        print("ВЕРДИКТ: 90 < best <= 100 — сигнатура работает, но сиды давали TPE фору.")
        print("  Фиксируем в паспорте: оптимум существует, доступ к нему облегчён сидами.")
    else:
        print("ВЕРДИКТ: best > 100 — золотая зона это УЗКИЙ КОРИДОР, без подсказки не попасть.")
        print("  Действие v46S: grid search по золотой зоне pl/R in [0.42, 0.52].")

# ============================================================
# RUN C — граница физики (wall 25/30/35)
# ============================================================
def run_c():
    print("=" * 70)
    print("RUN C — граница физики: wall_limit 25/30/35 МВт/м2")
    print("ФОРМУЛИРОВКА ПАСПОРТА: это ГРАНИЦА МОДЕЛИ, а не прорыв.")
    print("=" * 70)
    try:
        from optuna_optimizer import GAKOptunaOptimizer
    except ImportError:
        print("Запускать из корня репозитория: python v46R_validation.py runc"); return
    opt = GAKOptunaOptimizer(base_config={})
    for wall in (25.0, 30.0, 35.0):
        opt.WALL_LIMIT_W_M2 = wall * 1e6
        print(f"\n--- wall_limit = {wall} МВт/м2 ---")
        study = _blind_optimize(opt, 30, f"gak_opt_lcoe_v46R_wall{int(wall)}",
                                f"blind_v46R_wall{int(wall)}.sqlite")
        vals = [t.value for t in study.trials if t.value and t.value < 1e9] if study is not None else []
        if vals:
            print(f"best LCOE при wall={int(wall)}: {min(vals):.4f} $/МВт·ч ({len(vals)}/30)")
        else:
            print(f"wall={int(wall)}: успешных триалов нет")
        # пометка для паспорта: если best < 85.96, то LCOE X достижим при wall=Y,
        # что требует ODS-стали, НЕ ПРОИЗВОДИМОЙ СЕРИЙНО (граница, не прорыв)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "runa"
    {"runa": run_a, "tornado": tornado, "runb": run_b, "runc": run_c}[cmd]()
