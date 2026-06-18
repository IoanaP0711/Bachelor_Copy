import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the autoencoder intrusion detection model."
    )

    parser.add_argument(
        "--csv",
        default="data/processed/valid.csv",
        help="Path to the labelled validation CSV file.",
    )

    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the ground-truth label column.",
    )

    parser.add_argument(
        "--scaler",
        default="data/models/ae_scaler.joblib",
        help="Path to the fitted feature scaler.",
    )

    parser.add_argument(
        "--onnx",
        default="data/models/ae.omx",
        help="Path to the exported ONNX autoencoder model.",
    )

    parser.add_argument(
        "--outdir",
        default="reports/figures",
        help="Directory in which figures and metrics are saved.",
    )

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.5,
        help="Percentile used to calculate the anomaly threshold.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # =========================================================
    # Input file validation
    # =========================================================

    required_files = [
        args.csv,
        args.scaler,
        args.onnx,
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file does not exist: {file_path}"
            )

    # =========================================================
    # Load the validation dataset
    # =========================================================

    df = pd.read_csv(args.csv)

    print("=" * 68)
    print("AUTOENCODER EVALUATION")
    print("=" * 68)
    print(f"Dataset: {args.csv}")
    print(f"Dataset shape: {df.shape}")
    print(f"ONNX model: {args.onnx}")
    print(f"Scaler: {args.scaler}")

    y = None

    if args.label_col in df.columns:
        # Expected convention:
        # 0 = normal traffic
        # 1 = attack traffic
        y = (
            pd.to_numeric(
                df[args.label_col],
                errors="raise",
            )
            .astype(int)
            .values
        )

        unique_labels, label_counts = np.unique(
            y,
            return_counts=True,
        )

        print("\nGround-truth label distribution:")

        for label, count in zip(
            unique_labels,
            label_counts,
        ):
            label_name = (
                "Normal"
                if label == 0
                else "Attack"
            )

            print(
                f"  {label} ({label_name}): {count}"
            )

    else:
        print(
            f"\nWarning: label column '{args.label_col}' "
            "was not found. Classification metrics will not be calculated."
        )

    # Use all numeric columns except the label column
    X_df = (
        df.select_dtypes(include=[np.number])
        .drop(
            columns=[args.label_col],
            errors="ignore",
        )
    )

    if X_df.empty:
        raise ValueError(
            "The dataset does not contain numeric input features."
        )

    X = X_df.values.astype(np.float32)

    print(
        f"\nNumber of input features: {X.shape[1]}"
    )

    # =========================================================
    # Scale the input features
    # =========================================================

    scaler = joblib.load(args.scaler)

    X_scaled = scaler.transform(X).astype(np.float32)

    # =========================================================
    # Run ONNX inference
    # =========================================================

    session = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    reconstruction = session.run(
        None,
        {
            input_name: X_scaled,
        },
    )[0]

    if reconstruction.shape != X_scaled.shape:
        raise ValueError(
            "The model output shape does not match the input shape.\n"
            f"Input shape: {X_scaled.shape}\n"
            f"Output shape: {reconstruction.shape}"
        )

    # =========================================================
    # Calculate the reconstruction error
    # =========================================================

    reconstruction_error = np.mean(
        (X_scaled - reconstruction) ** 2,
        axis=1,
    )

    if not np.all(np.isfinite(reconstruction_error)):
        nan_count = np.isnan(
            reconstruction_error
        ).sum()

        inf_count = np.isinf(
            reconstruction_error
        ).sum()

        raise ValueError(
            "The reconstruction errors contain invalid values.\n"
            f"NaN values: {nan_count}\n"
            f"Infinite values: {inf_count}"
        )

    # =========================================================
    # Reconstruction-error statistics
    # =========================================================

    error_statistics = {
        "minimum": np.min(reconstruction_error),
        "percentile_50": np.percentile(
            reconstruction_error,
            50,
        ),
        "percentile_90": np.percentile(
            reconstruction_error,
            90,
        ),
        "percentile_95": np.percentile(
            reconstruction_error,
            95,
        ),
        "percentile_97": np.percentile(
            reconstruction_error,
            97,
        ),
        "percentile_99": np.percentile(
            reconstruction_error,
            99,
        ),
        "percentile_99_5": np.percentile(
            reconstruction_error,
            99.5,
        ),
        "percentile_99_9": np.percentile(
            reconstruction_error,
            99.9,
        ),
        "maximum": np.max(reconstruction_error),
    }

    print("\nRECONSTRUCTION-ERROR STATISTICS")
    print("-" * 68)

    for name, value in error_statistics.items():
        print(f"{name:20s}: {value:.10f}")

    # =========================================================
    # Calculate the anomaly threshold and predictions
    # =========================================================

    if y is not None:
        normal_errors = reconstruction_error[y == 0]

        if len(normal_errors) == 0:
            raise ValueError(
                "No normal samples were found for threshold calibration."
            )

        threshold_reference = normal_errors
        threshold_source = "normal validation samples only"
    else:
        threshold_reference = reconstruction_error
        threshold_source = "all available samples"

    threshold = np.percentile(
        threshold_reference,
        args.threshold_percentile,
    )

    # Predicted classes:
    # 0 = normal
    # 1 = anomalous / attack
    y_pred = (
        reconstruction_error >= threshold
    ).astype(int)

    print("\nANOMALY THRESHOLD")
    print("-" * 68)
    print(
        f"Threshold percentile: "
        f"{args.threshold_percentile}"
    )
    print(f"Calculated threshold: {threshold:.10f}")
    print(f"Threshold source: {threshold_source}")
    print(
        f"Samples used for threshold calibration: "
        f"{len(threshold_reference)}"
    )
    print(
        f"Flows classified as anomalous: "
        f"{np.sum(y_pred == 1)}"
    )
    print(
        f"Flows classified as normal: "
        f"{np.sum(y_pred == 0)}"
    )

    # =========================================================
    # Full-range reconstruction-error histogram
    # =========================================================

    plt.figure(figsize=(10, 6))

    plt.hist(
        reconstruction_error,
        bins=80,
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Anomaly threshold = {threshold:.6f}",
    )

    plt.xlabel("Reconstruction error")
    plt.ylabel("Number of flows")
    plt.title(
        "Autoencoder Reconstruction-Error Distribution — Full Range"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{args.outdir}/ae_error_hist_full.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================
    # Zoomed reconstruction-error histogram
    # =========================================================

    zoom_limit = np.percentile(
        reconstruction_error,
        99.5,
    )

    zoomed_errors = reconstruction_error[
        reconstruction_error <= zoom_limit
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        zoomed_errors,
        bins=80,
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Anomaly threshold = {threshold:.6f}",
    )

    plt.xlabel("Reconstruction error")
    plt.ylabel("Number of flows")
    plt.title(
        "Autoencoder Reconstruction-Error Distribution — Zoomed View"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{args.outdir}/ae_error_hist.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.savefig(
        f"{args.outdir}/ae_error_hist_zoomed.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================
    # Log-transformed reconstruction-error histogram
    # =========================================================

    log_errors = np.log1p(
        reconstruction_error
    )

    log_threshold = np.log1p(
        threshold
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        log_errors,
        bins=80,
    )

    plt.axvline(
        log_threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Anomaly threshold = {threshold:.6f}"
        ),
    )

    plt.xlabel(
        "log(1 + reconstruction error)"
    )

    plt.ylabel("Number of flows")

    plt.title(
        "Autoencoder Reconstruction-Error Distribution — Logarithmic Scale"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{args.outdir}/ae_error_hist_log.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================
    # Reconstruction error for a subset of flows
    # =========================================================

    subset_size = min(
        2000,
        len(reconstruction_error),
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        reconstruction_error[:subset_size]
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Anomaly threshold = {threshold:.6f}",
    )

    plt.xlabel("Flow index")
    plt.ylabel("Reconstruction error")

    plt.title(
        "Reconstruction Error and Anomaly Threshold for a Flow Subset"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{args.outdir}/ae_error_timeseries_threshold.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================
    # General evaluation results
    # =========================================================

    results = {
        "dataset": args.csv,
        "number_of_flows": int(
            len(reconstruction_error)
        ),
        "number_of_features": int(
            X.shape[1]
        ),
        "threshold": float(threshold),
        "threshold_percentile": float(
            args.threshold_percentile
        ),
        "error_statistics": {
            key: float(value)
            for key, value in error_statistics.items()
        },
    }

    # =========================================================
    # Classification metrics
    # =========================================================

    if y is not None:
        confusion = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = confusion.ravel()

        accuracy = accuracy_score(
            y,
            y_pred,
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y,
                y_pred,
                average="binary",
                zero_division=0,
            )
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

        results.update({
            "confusion_matrix": confusion.tolist(),
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
            "specificity": float(
                specificity
            ),
        })

        print("\nCONFUSION MATRIX")
        print("-" * 68)
        print(f"True negatives:  {tn}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"True positives:  {tp}")

        print("\nCLASSIFICATION METRICS")
        print("-" * 68)
        print(
            f"Accuracy:            "
            f"{accuracy * 100:.2f}%"
        )
        print(
            f"Precision:           "
            f"{precision * 100:.2f}%"
        )
        print(
            f"Recall:              "
            f"{recall * 100:.2f}%"
        )
        print(
            f"F1-score:            "
            f"{f1 * 100:.2f}%"
        )
        print(
            f"False-positive rate: "
            f"{false_positive_rate * 100:.4f}%"
        )
        print(
            f"Specificity:         "
            f"{specificity * 100:.2f}%"
        )

        # =====================================================
        # ROC and precision-recall curves
        # =====================================================

        try:
            roc_auc = roc_auc_score(
                y,
                reconstruction_error,
            )

            pr_auc = average_precision_score(
                y,
                reconstruction_error,
            )

            results["roc_auc"] = float(
                roc_auc
            )

            results["pr_auc"] = float(
                pr_auc
            )

            print(
                f"ROC-AUC:             "
                f"{roc_auc * 100:.2f}%"
            )

            print(
                f"PR-AUC:              "
                f"{pr_auc * 100:.2f}%"
            )

            false_positive_rates, true_positive_rates, _ = roc_curve(
                y,
                reconstruction_error,
            )

            plt.figure(figsize=(8, 6))

            plt.plot(
                false_positive_rates,
                true_positive_rates,
                label=f"ROC-AUC = {roc_auc:.4f}",
            )

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Random classifier",
            )

            plt.xlabel("False-positive rate")
            plt.ylabel("True-positive rate")
            plt.title("Receiver Operating Characteristic Curve")
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                f"{args.outdir}/ae_roc_curve.png",
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()

            precision_values, recall_values, _ = (
                precision_recall_curve(
                    y,
                    reconstruction_error,
                )
            )

            plt.figure(figsize=(8, 6))

            plt.plot(
                recall_values,
                precision_values,
                label=f"Average precision = {pr_auc:.4f}",
            )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall Curve")
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                f"{args.outdir}/ae_pr_curve.png",
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()

        except Exception as error:
            results["auc_error"] = str(
                error
            )

        # =====================================================
        # Confusion-matrix figure
        # =====================================================

        plt.figure(figsize=(7, 6))

        plt.imshow(confusion)

        plt.title("Autoencoder Confusion Matrix")
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")

        plt.xticks(
            [0, 1],
            ["Normal", "Attack"],
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
                    str(confusion[row, column]),
                    ha="center",
                    va="center",
                )

        plt.tight_layout()

        plt.savefig(
            f"{args.outdir}/ae_confusion_matrix.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    # =========================================================
    # Save metrics and predictions
    # =========================================================

    metrics_path = (
        f"{args.outdir}/ae_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    predictions_df = pd.DataFrame({
        "true_label": (
            y
            if y is not None
            else np.full(
                len(reconstruction_error),
                np.nan,
            )
        ),
        "anomaly_score": reconstruction_error,
        "predicted_label": y_pred,
    })

    if y is not None:
        predictions_df[
            "prediction_correct"
        ] = y == y_pred

    predictions_path = (
        f"{args.outdir}/ae_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\nSAVED OUTPUT FILES")
    print("-" * 68)
    print(f"Output directory: {args.outdir}")
    print(f"Metrics file: {metrics_path}")
    print(f"Predictions file: {predictions_path}")
    print(
        f"Main histogram: "
        f"{args.outdir}/ae_error_hist.png"
    )


if __name__ == "__main__":
    main()