import argparse
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/valid.csv")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--scaler", default="data/models/ae_scaler.joblib")
    ap.add_argument("--onnx", default="data/models/ae.onnx")
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--threshold-percentile", type=float, default=99.5)
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    y = None
    if args.label_col in df.columns:
        # assumes: 0=normal, 1=attack (adjust if needed)
        y = df[args.label_col].astype(int).values

    X = df.select_dtypes(include=[np.number]).drop(columns=[args.label_col], errors="ignore").values.astype(np.float32)

    scaler = joblib.load(args.scaler)
    Xs = scaler.transform(X).astype(np.float32)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    recon = sess.run(None, {inp: Xs})[0]

    # reconstruction error per sample
    err = np.mean((Xs - recon) ** 2, axis=1)

    # choose threshold using percentile (common for unsupervised)
    tau = np.percentile(err, args.threshold_percentile)
    yhat = (err >= tau).astype(int)

    # Save histogram (you already have similar, but keep consistent)
    plt.figure()
    plt.hist(err, bins=80)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title("AE reconstruction error distribution")
    plt.savefig(f"{args.outdir}/ae_error_hist.png", dpi=200)
    plt.close()

    # Error with threshold line
    plt.figure()
    plt.plot(err[:2000])  # slice for readability
    plt.axhline(tau, linestyle="--")
    plt.xlabel("Flow index (subset)")
    plt.ylabel("Reconstruction error")
    plt.title(f"AE error over time (subset) with threshold tau={tau:.6f}")
    plt.savefig(f"{args.outdir}/ae_error_timeseries_threshold.png", dpi=200)
    plt.close()

    results = {"threshold": float(tau), "threshold_percentile": args.threshold_percentile}

    # If labels exist, compute full metrics
    if y is not None:
        cm = confusion_matrix(y, yhat)
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)

        results.update({
            "confusion_matrix": cm.tolist(),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

        # ROC/PR curves
        try:
            auc = roc_auc_score(y, err)
            ap_score = average_precision_score(y, err)
            results["roc_auc"] = float(auc)
            results["pr_auc"] = float(ap_score)

            fpr, tpr, _ = roc_curve(y, err)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title(f"ROC curve (AUC={auc:.4f})")
            plt.savefig(f"{args.outdir}/ae_roc_curve.png", dpi=200)
            plt.close()

            p, r, _ = precision_recall_curve(y, err)
            plt.figure()
            plt.plot(r, p)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR curve (AP={ap_score:.4f})")
            plt.savefig(f"{args.outdir}/ae_pr_curve.png", dpi=200)
            plt.close()

        except Exception as e:
            results["auc_error"] = str(e)

        # Confusion matrix plot
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion matrix (AE)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.savefig(f"{args.outdir}/ae_confusion_matrix.png", dpi=200)
        plt.close()

    with open(f"{args.outdir}/ae_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved figures + metrics to:", args.outdir)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
