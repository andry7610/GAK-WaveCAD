# check_study.py
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

s = optuna.load_study(
    study_name='gak_opt_lcoe_v45R',
    storage='sqlite:///models/optuna_study.db',
)
print(f"Всего триалов: {len(s.trials)}")
KEYS = ('Q_plasma', 'f_ext', 'P_fusion_W', 'P_electric_W',
        'f_He', 'Z_eff', 'wall_load_W_m2', 'lcoe', 'lcoe_raw',
        'P_net_W_final', 'error')
for t in s.trials:
    ua = t.user_attrs
    print(f"trial {t.number} | {t.state.name} | value={t.value}")
    for k in KEYS:
        if k in ua:
            print(f"    {k} = {ua[k]}")
    if not ua:
        print("    (атрибутов нет — reject до записи телеметрии)")
