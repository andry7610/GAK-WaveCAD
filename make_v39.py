# -*- coding: utf-8 -*-
# ============================================================
# make_v39.py — патч-скрипт v38 -> v39 «ПЛОТНОСТНАЯ СТЕНА»
# Делает три правки в optuna_optimizer.py:
#   1) версия: все упоминания v38 -> v39
#   2) потолок плотности: 1.6e21 -> 2.2e21
#   3) диапазон external_B: верх 31.0 -> 30.5
#   4) сид-чемпион: trial 37 -> trial 46 (значения из прогона v38)
# Создаёт резервную копию optuna_optimizer_v38_backup.py
# ============================================================
import re, shutil, sys, os

SRC = "optuna_optimizer.py"
BAK = "optuna_optimizer_v38_backup.py"

if not os.path.exists(SRC):
    print(f"!! не найден файл {SRC} — запусти скрипт из папки репозитория")
    sys.exit(1)

shutil.copy2(SRC, BAK)
text = open(BAK, encoding="utf-8").read()
report = []

# ---------- 1. версия ----------
n_ver = text.count("v38")
text = text.replace("v38", "v39")
report.append(("OK" if n_ver > 0 else "!!", f"упоминаний версии v38 -> v39: {n_ver}"))

# ---------- 2. потолок n_density ----------
pat_nd = re.compile(r"1[.,]6e\+?21", re.IGNORECASE)
hits = pat_nd.findall(text)
# меняем только верхнюю границу: любое вхождение 1.6e21 рядом с n_density
def nd_replace(m):
    return "2.2e21"
new_text, n_nd = pat_nd.subn(nd_replace, text)
text = new_text
report.append(("OK" if n_nd > 0 else "!!", f"потолок n_density 1.6e21 -> 2.2e21: замен {n_nd}"))

# ---------- 3. external_B верх 31.0 -> 30.5 ----------
pat_b = re.compile(r"(28[.,]0\s*,\s*)31[.,]0")
new_text, n_b = pat_b.subn(r"\g<1>30.5", text)
text = new_text
report.append(("OK" if n_b > 0 else "!!", f"диапазон external_B (28.0, 31.0) -> (28.0, 30.5): замен {n_b}"))

# ---------- 4. сид-чемпион trial 46 ----------
# Значения trial 46 из прогона v38 (export 2026-10-25T16-34)
CHAMPION = {
    'R_mm':          1190.4334026848562,
    'h_mm':          11.519994778347218,
    'T_inner':       386.18985904795517,
    'T_outer':       297.34260698886783,
    'k_thermal':     49.44157110112505,
    'pl_radius_mm':  614.3783024358249,
    'T_plasma_init': 128590142.09780777,
    'n_density':     1.5971752597294838e+21,
    'P_initial':     0.0943352766272453,
    'pump_speed':    0.09555844817595341,
    'E_field':       5.068630123167281e+17,
    'gap_nm':        2,
    'external_B':    28.831820132016368,
}
ok_keys, miss_keys = 0, []
for key, val in CHAMPION.items():
    pat = re.compile(r"('" + re.escape(key) + r"'\s*:\s*)([-+0-9.eE]+)")
    new_text, n = pat.subn(lambda m: m.group(1) + repr(val), text)
    if n > 0:
        text = new_text
        ok_keys += n
    else:
        miss_keys.append(key)
if miss_keys:
    report.append(("!!", f"сид-чемпион: не найдены ключи {miss_keys} — патч сид не применён полностью"))
else:
    report.append(("OK", f"сид-чемпион trial 37 -> trial 46 (x{ok_keys}/{len(CHAMPION)})"))

# ---------- 5. номер сида и комментарий ----------
new_text, n_seed = re.subn(r"SEED_TRIAL\s*=\s*37", "SEED_TRIAL = 46", text)
text = new_text
new_text, n_cmt = re.subn(r"trial 37, P_net = 597[.,]24", "trial 46, P_net = 646.58", text)
text = new_text
report.append(("OK" if (n_seed + n_cmt) > 0 else "!!", f"номер сида/комментарий 37 -> 46: замен {n_seed + n_cmt}"))

# ---------- запись ----------
open(SRC, "w", encoding="utf-8").write(text)

print("=== ОТЧЁТ ПАТЧА v39 ===")
for status, msg in report:
    print(f"{status}  {msg}")
if any(s == "!!" for s, _ in report):
    print("\nЕсть строки с !! — ЗАПУСКАТЬ ОПТИМИЗАТОР НЕЛЬЗЯ.")
    print("Резервная копия сохранена:", BAK)
    print("Пришли вывод скрипта автору патча (Алисе) — подгоним регулярки под твой файл.")
    sys.exit(2)
print("\nВсе правки применены. Резервная копия:", BAK)
print("Теперь: del optuna_study.db  &&  python optuna_optimizer.py")
print("Контрольная метка: Trial 1 должен выдать ровно 646.58 МВт.")
