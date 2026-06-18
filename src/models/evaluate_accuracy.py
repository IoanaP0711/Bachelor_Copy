from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURARE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fișierul trebuie să conțină:
# true_label, anomaly_score
PREDICTIONS_FILE = PROJECT_ROOT / "validation_results" / "validation_predictions.csv"

# Fișierul cu pragurile modelului
THRESHOLDS_FILE = PROJECT_ROOT / "data" / "models" / "ae_threshold_bands.json"


# ============================================================
# FUNCȚII
# ============================================================

def load_anomaly_threshold() -> float:
    """
    Încarcă pragul de la care un flow este considerat anomal.

    Scriptul caută mai multe denumiri posibile în fișierul JSON.
    """

    if not THRESHOLDS_FILE.exists():
        raise FileNotFoundError(
            f"Nu există fișierul cu praguri:\n{THRESHOLDS_FILE}"
        )

    with THRESHOLDS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    possible_keys = [
        "anomaly_threshold",
        "threshold",
        "warn",
        "warning",
        "review",
    ]

    for key in possible_keys:
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])

    # Caută și în obiecte interne, de exemplu:
    # {"thresholds": {"warn": 3.0}}
    for value in data.values():
        if isinstance(value, dict):
            for key in possible_keys:
                if key in value and isinstance(value[key], (int, float)):
                    return float(value[key])

    raise ValueError(
        "Nu am găsit un prag utilizabil în ae_threshold_bands.json.\n"
        f"Conținutul găsit este:\n{json.dumps(data, indent=2)}"
    )


def normalize_true_labels(series: pd.Series) -> pd.Series:
    """
    Transformă etichetele reale în:
    0 = trafic normal
    1 = trafic malițios/anomal
    """

    if pd.api.types.is_numeric_dtype(series):
        labels = pd.to_numeric(series, errors="coerce")

        if labels.isna().any():
            raise ValueError("Coloana true_label conține valori numerice invalide.")

        return labels.astype(int)

    mapping = {
        "0": 0,
        "normal": 0,
        "benign": 0,
        "ok": 0,

        "1": 1,
        "attack": 1,
        "malicious": 1,
        "anomaly": 1,
        "anomalous": 1,
        "review": 1,
        "critical": 1,
    }

    labels = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if labels.isna().any():
        invalid_values = series[labels.isna()].unique().tolist()

        raise ValueError(
            "Există etichete reale necunoscute în coloana true_label: "
            f"{invalid_values}"
        )

    return labels.astype(int)


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Nu există fișierul cu rezultatele de validare:\n"
            f"{PREDICTIONS_FILE}\n\n"
            "Fișierul trebuie să conțină coloanele:\n"
            "true_label,anomaly_score"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {"true_label", "anomaly_score"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Lipsesc următoarele coloane din CSV: "
            f"{sorted(missing_columns)}\n"
            f"Coloane existente: {list(df.columns)}"
        )

    df = df.dropna(subset=["true_label", "anomaly_score"]).copy()

    if df.empty:
        raise ValueError("Fișierul nu conține rezultate valide.")

    y_true = normalize_true_labels(df["true_label"])

    anomaly_scores = pd.to_numeric(
        df["anomaly_score"],
        errors="coerce",
    )

    if anomaly_scores.isna().any():
        raise ValueError(
            "Coloana anomaly_score conține valori care nu sunt numerice."
        )

    threshold = load_anomaly_threshold()

    # 0 = normal
    # 1 = anomal/malițios
    y_pred = (anomaly_scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    detection_rate = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    print("=" * 60)
    print("EVALUAREA AUTOENCODERULUI")
    print("=" * 60)

    print(f"Fișier analizat: {PREDICTIONS_FILE}")
    print(f"Număr total de flow-uri: {len(df)}")
    print(f"Prag de anomalie utilizat: {threshold:.6f}")

    print()
    print("METRICE")
    print("-" * 60)
    print(f"Accuracy:            {accuracy * 100:.2f}%")
    print(f"Precision:           {precision * 100:.2f}%")
    print(f"Recall:              {recall * 100:.2f}%")
    print(f"Detection rate:      {detection_rate * 100:.2f}%")
    print(f"F1-score:            {f1 * 100:.2f}%")
    print(f"False-positive rate: {false_positive_rate * 100:.2f}%")

    print()
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"TN — normal detectat corect:          {tn}")
    print(f"FP — normal detectat ca atac:         {fp}")
    print(f"FN — atac detectat ca normal:         {fn}")
    print(f"TP — atac detectat corect:            {tp}")

    print()
    print("RAPORT COMPLET")
    print("-" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )

    results = df.copy()
    results["predicted_label"] = y_pred
    results["prediction_correct"] = y_true == y_pred

    output_file = (
        PROJECT_ROOT
        / "validation_results"
        / "accuracy_evaluation_results.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print(f"Rezultatele detaliate au fost salvate în:\n{output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEROARE: {exc}", file=sys.stderr)
        sys.exit(1)