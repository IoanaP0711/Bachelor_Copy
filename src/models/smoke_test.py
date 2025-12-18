import onnxruntime as ort
import pandas as pd
import numpy as np
import joblib
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/valid.csv")
    parser.add_argument("--model", default="data/models/iforest.onnx")
    parser.add_argument("--scaler", default="data/models/scaler.pkl")
    parser.add_argument("--n", type=int, default=5, help="Rows to test")
    args = parser.parse_args()

    # ---- Load CSV ----
    print(f"[INFO] Reading {args.csv} ...")
    df = pd.read_csv(args.csv)

    # Detect label column
    label_col = None
    for c in ["label", "Label", "y", "target"]:
        if c in df.columns:
            label_col = c
            break

    if label_col:
        X = df.drop(columns=[label_col])
        y = df[label_col].iloc[: args.n].values
    else:
        X = df
        y = None

    X_sample = X.iloc[: args.n].values.astype(np.float32)
    print(f"[INFO] Loaded {X_sample.shape[0]} samples with {X_sample.shape[1]} features")

    # ---- Load scaler ----
    if os.path.exists(args.scaler):
        print(f"[INFO] Loading scaler: {args.scaler}")
        scaler = joblib.load(args.scaler)
        X_scaled = scaler.transform(X_sample)
    else:
        print("[WARN] Scaler not found — using raw values")
        X_scaled = X_sample

    # ---- Load ONNX model ----
    print(f"[INFO] Loading ONNX model: {args.model}")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # ---- Run inference ----
    preds = sess.run([output_name], {input_name: X_scaled})[0]

    print("\n=== Smoke Test Results ===")
    for i, p in enumerate(preds):
        if y is not None:
            print(f"Row {i}: Pred={int(p)}  TrueLabel={y[i]}")
        else:
            print(f"Row {i}: Pred={int(p)}")

    print("==========================\n")


if __name__ == "__main__":
    main()
