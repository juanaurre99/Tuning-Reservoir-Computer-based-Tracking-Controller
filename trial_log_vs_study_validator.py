import pandas as pd
import numpy as np
import ast


def validate_logger_vs_study(csv_path, study, sampler_type, run_id, atol=1e-8):
    """
    Vergleicht jeden Trial-Eintrag im CSV mit der Optuna-Study.

    Parameters:
    - csv_path: Pfad zur trial_log.csv-Datei.
    - study: Optuna Study-Objekt (nach Tuning).
    - sampler_type: z. B. "tpe", "random", etc.
    - run_id: int, um trial_id korrekt zu rekonstruieren.
    - atol: Toleranz für numerische Abweichungen.
    """
    print(f"[Validator] Lade Logger-CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Fehler] CSV konnte nicht geladen werden: {e}")
        return

    if "params" not in df.columns or "trial_id" not in df.columns:
        print("[Fehler] CSV-Datei enthält nicht alle benötigten Spalten.")
        return

    df["params"] = df["params"].apply(ast.literal_eval)

    mismatches = []

    for trial in study.trials:
        expected_id = f"{sampler_type}_{run_id:02}_{trial.number:02}"

        df_row = df[df["trial_id"] == expected_id]

        if df_row.empty:
            mismatches.append((trial.number, "Fehlt im Logger"))
            continue

        row = df_row.iloc[0]

        # 1. Prüfe RMSE/Value
        csv_rmse = float(row["rmse"])
        study_value = float(trial.value)
        if not np.isclose(csv_rmse, study_value, atol=atol):
            mismatches.append((trial.number, f"RMSE: CSV={csv_rmse:.6f}, Study={study_value:.6f}"))

        # 2. Prüfe Hyperparameter
        csv_params = row["params"]
        for k, v in trial.params.items():
            if k not in csv_params:
                mismatches.append((trial.number, f"Param '{k}' fehlt in CSV"))
                continue

            csv_val = csv_params[k]
            if isinstance(v, float):
                if not np.isclose(csv_val, v, atol=atol):
                    mismatches.append((trial.number, f"Param '{k}': CSV={csv_val} vs Study={v}"))
            else:
                if csv_val != v:
                    mismatches.append((trial.number, f"Param '{k}': CSV={csv_val} vs Study={v}"))

    # Ergebnis anzeigen
    if not mismatches:
        print("✅ Alle Logger-Einträge stimmen mit der Optuna-Study überein.")
    else:
        print(f"❌ {len(mismatches)} Abweichungen gefunden:")
        for trial_num, reason in mismatches:
            print(f"  - Trial {trial_num:02}: {reason}")


# Optionaler direkter Aufruf
if __name__ == "__main__":
    import joblib
    import optuna

    # Beispielhafte Nutzung – passe an deinen Pfad und Study an
    study_path = "optuna_study.pkl"
    csv_path = "plots/manual_labeling/trial_log.csv"
    sampler_type = "tpe"
    run_id = 0

    # Lade Optuna-Study
    try:
        study = joblib.load(study_path)
    except Exception as e:
        print(f"[Fehler] Study konnte nicht geladen werden: {e}")
        exit()

    validate_logger_vs_study(csv_path, study, sampler_type, run_id)
