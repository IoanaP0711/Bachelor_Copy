import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


GROUND_TRUTH_MAPPING = {
    "0": 0,
    "NORMAL": 0,
    "BENIGN": 0,
    "OK": 0,

    "1": 1,
    "ATTACK": 1,
    "MALICIOUS": 1,
    "ANOMALOUS": 1,
    "ANOMALY": 1,
    "SUSPICIOUS": 1,
}


SYSTEM_DECISION_MAPPING = {
    "OK": 0,
    "BENIGN": 0,
    "NORMAL": 0,

    "WARN": 1,
    "WARNING": 1,
    "MED": 1,
    "MEDIUM": 1,
    "REVIEW": 1,
    "CRIT": 1,
    "CRITICAL": 1,
    "ATTACK": 1,
    "MALICIOUS": 1,
    "ANOMALOUS": 1,
    "SUSPICIOUS": 1,
}


def normalize_column(
    series: pd.Series,
    mapping: dict,
    column_name: str,
) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map(mapping)
    )

    if normalized.isna().any():
        invalid_values = (
            series[normalized.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown values in column '{column_name}': "
            f"{invalid_values}"
        )

    return normalized.astype(int)


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "specificity": float(specificity),
    }


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print()
    print(title)
    print("-" * 68)

    print(
        f"True negatives:      "
        f"{metrics['true_negatives']}"
    )

    print(
        f"False positives:     "
        f"{metrics['false_positives']}"
    )

    print(
        f"False negatives:     "
        f"{metrics['false_negatives']}"
    )

    print(
        f"True positives:      "
        f"{metrics['true_positives']}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:           "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:              "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score:            "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print(
        f"False-positive rate: "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )

    print(
        f"Specificity:         "
        f"{metrics['specificity'] * 100:.2f}%"
    )


def save_confusion_matrix(
    matrix: list,
    title: str,
    output_path: str,
) -> None:
    matrix_array = np.array(matrix)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix_array)

    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")

    plt.xticks(
        [0, 1],
        ["Normal", "Suspicious"],
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Attack"],
    )

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(matrix_array[row, column]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_comparison_chart(
    raw_metrics: dict,
    final_metrics: dict,
    output_path: str,
) -> None:
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    display_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    raw_values = [
        raw_metrics[name] * 100
        for name in metric_names
    ]

    final_values = [
        final_metrics[name] * 100
        for name in metric_names
    ]

    positions = np.arange(
        len(metric_names)
    )

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        positions - width / 2,
        raw_values,
        width,
        label="Raw decision",
    )

    plt.bar(
        positions + width / 2,
        final_values,
        width,
        label="Final system decision",
    )

    plt.xticks(
        positions,
        display_names,
    )

    plt.ylabel("Score (%)")

    plt.title(
        "Raw Model Output versus Final System Decision"
    )

    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the complete intrusion detection "
            "system using controlled labelled flows."
        )
    )

    parser.add_argument(
        "--csv",
        default=(
            "validation_results/"
            "full_system_validation.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        default=(
            "validation_results/"
            "full_system_evaluation"
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"Validation file not found: {args.csv}"
        )

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    dataframe = pd.read_csv(args.csv)

    required_columns = {
        "validation_id",
        "scenario",
        "ground_truth",
        "final_decision",
    }

    missing_columns = (
        required_columns
        .difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_columns)}"
        )

    y_true = normalize_column(
        dataframe["ground_truth"],
        GROUND_TRUTH_MAPPING,
        "ground_truth",
    )

    final_prediction = normalize_column(
        dataframe["final_decision"],
        SYSTEM_DECISION_MAPPING,
        "final_decision",
    )

    final_metrics = calculate_metrics(
        y_true,
        final_prediction,
    )

    print("=" * 68)
    print("COMPLETE IDS EVALUATION")
    print("=" * 68)

    print(
        f"Validation file: {args.csv}"
    )

    print(
        f"Number of evaluated flows: "
        f"{len(dataframe)}"
    )

    print_metrics(
        "FINAL SYSTEM DECISION",
        final_metrics,
    )

    results = {
        "number_of_flows": int(
            len(dataframe)
        ),
        "binary_mapping": {
            "negative": [
                "OK",
                "BENIGN",
            ],
            "positive": [
                "REVIEW",
                "CRITICAL",
            ],
        },
        "final_system": final_metrics,
    }

    dataframe[
        "ground_truth_binary"
    ] = y_true

    dataframe[
        "final_prediction_binary"
    ] = final_prediction

    dataframe[
        "final_prediction_correct"
    ] = (
        y_true == final_prediction
    )

    save_confusion_matrix(
        final_metrics["confusion_matrix"],
        "Complete IDS Confusion Matrix",
        (
            f"{args.outdir}/"
            "final_system_confusion_matrix.png"
        ),
    )

    if "raw_decision" in dataframe.columns:
        raw_prediction = normalize_column(
            dataframe["raw_decision"],
            SYSTEM_DECISION_MAPPING,
            "raw_decision",
        )

        raw_metrics = calculate_metrics(
            y_true,
            raw_prediction,
        )

        print_metrics(
            "RAW MODEL DECISION",
            raw_metrics,
        )

        results["raw_model"] = raw_metrics

        dataframe[
            "raw_prediction_binary"
        ] = raw_prediction

        dataframe[
            "raw_prediction_correct"
        ] = (
            y_true == raw_prediction
        )

        save_confusion_matrix(
            raw_metrics["confusion_matrix"],
            "Raw Model Confusion Matrix",
            (
                f"{args.outdir}/"
                "raw_model_confusion_matrix.png"
            ),
        )

        save_comparison_chart(
            raw_metrics,
            final_metrics,
            (
                f"{args.outdir}/"
                "raw_vs_final_metrics.png"
            ),
        )

    results_path = (
        f"{args.outdir}/"
        "full_system_metrics.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    evaluated_csv_path = (
        f"{args.outdir}/"
        "evaluated_flows.csv"
    )

    dataframe.to_csv(
        evaluated_csv_path,
        index=False,
    )

    print()
    print("SAVED OUTPUT FILES")
    print("-" * 68)
    print(
        f"Metrics: {results_path}"
    )
    print(
        f"Evaluated flows: "
        f"{evaluated_csv_path}"
    )
    print(
        f"Final confusion matrix: "
        f"{args.outdir}/"
        "final_system_confusion_matrix.png"
    )

    if "raw_decision" in dataframe.columns:
        print(
            f"Raw versus final chart: "
            f"{args.outdir}/"
            "raw_vs_final_metrics.png"
        )


if __name__ == "__main__":
    main()