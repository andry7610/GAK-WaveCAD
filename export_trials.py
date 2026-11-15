# -*- coding: utf-8 -*-
"""
export_trials.py — выгрузка всех trial'ов Optuna из SQLite-базы в CSV.
Использование:
    python export_trials.py                      # база models/optuna_study.db -> trials_export.csv
    python export_trials.py путь/к/базе.db [выход.csv]
"""
import sys, os
import pandas as pd

DEFAULT_DB = os.path.join("models", "optuna_study.db")
DEFAULT_OUT = "trials_export.csv"

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    if not os.path.exists(db_path):
        print(f"[ОШИБКА] База не найдена: {db_path}")
        print("Укажи путь к базе явно: python export_trials.py models/optuna_study.db")
        return

    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Список studies в базе
    try:
        studies = cur.execute("SELECT study_id, study_name FROM studies").fetchall()
    except Exception as e:
        print(f"[ОШИБКА] Не удалось прочитать базу (это Optuna SQLite?): {e}")
        return
    if not studies:
        print("[ПУСТО] В базе нет ни одного study.")
        return
    print(f"Найдено studies: {[(sid, name) for sid, name in studies]}")

    frames = []
    for study_id, study_name in studies:
        # Параметры trial'ов: ключ -> значение (распаковываем JSON-числа)
        try:
            rows = pd.read_sql_query(
                """SELECT t.trial_id, t.number, t.state,
                          p.param_name, p.param_value
                   FROM trials t
                   LEFT JOIN trial_params p ON p.trial_id = t.trial_id
                   WHERE t.study_id = ?""",
                conn, params=(study_id,))
        except Exception as e:
            print(f"[ОШИБКА] study '{study_name}': {e}")
            continue
        # Значения objective: value в trials или trial_values
        try:
            vals = pd.read_sql_query(
                "SELECT trial_id, value FROM trial_values", conn)
        except Exception:
            vals = pd.read_sql_query(
                "SELECT trial_id, value FROM trials WHERE value IS NOT NULL", conn)

        if rows.empty:
            print(f"[ПУСТО] study '{study_name}': нет trial'ов")
            continue

        rows = rows.dropna(subset=["param_name"])
        # Распаковка Optuna-кодировки параметров (1 колонка = param_value с префиксом типа)
        def unpack(v):
            try:
                if v is None:
                    return None
                s = str(v)
                # Optuna хранит как "число-строку"; пробуем float
                return float(s)
            except Exception:
                return v
        rows["param_value"] = rows["param_value"].map(unpack)
        params_df = rows.pivot_table(index=["trial_id", "number", "state"],
                                     columns="param_name",
                                     values="param_value",
                                     aggfunc="first").reset_index()
        # Оставляем только таблицу с pivot-колонками без MultiIndex имени
        params_df.columns.name = None
        if not vals.empty:
            params_df = params_df.merge(vals, on="trial_id", how="left")
        params_df["study"] = study_name
        frames.append(params_df)
        print(f"  study '{study_name}': {len(params_df)} trial'ов -> выгружено")

    conn.close()

    if not frames:
        print("[ПУСТО] Нечего сохранять.")
        return

    result = pd.concat(frames, ignore_index=True, sort=False)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[ГОТОВО] Сохранено: {out_path} ({len(result)} строк, {len(result.columns)} колонок)")
    print(f"Колонки: {list(result.columns)}")

if __name__ == "__main__":
    main()
