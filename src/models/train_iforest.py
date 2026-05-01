#!/usr/bin/env python3
"""
Train an Isolation Forest on a CSV dataset and export to ONNX.
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ----------------------------------------------------------
# ARGUMENT PARSER
# ----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train IsolationForest on CSV and export ONNX.")
    p.add_argument("--in-csv", default="data/processed/train.csv")
    p.add_argument("--out-onnx", default="data/models/iforest.onnx")
    p.add_argument("--label-col", default="label")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------
# LOAD & VALIDATE DATASET
# ----------------------------------------------------------
def load_dataset(path, label_col):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    # numeric only
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found in CSV.")

    return X, y


# ----------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------
def train_model(X_train_scaled, contamination, random_state):
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train_scaled)
    return clf


# ----------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------
def evaluate_model(model, X_val_scaled, y_val):
    """
    Evaluate Isolation Forest using:
    - decision_function for AUC (continuous score)
    - predict() for F1 (binary anomaly decision)
    """

    # Isolation Forest scores:
    # decision_function -> higher = more normal
    # so we NEGATE it to get "anomaly score"
    scores = -model.decision_function(X_val_scaled)

    # Binary predictions (needed for F1)
    raw = model.predict(X_val_scaled)        # 1 = normal, -1 = anomaly
    pred = (raw == -1).astype(int)           # 1 = anomaly

    # Metrics
    auc = roc_auc_score(y_val, scores)
    f1 = f1_score(y_val, pred)

    print(f"[METRIC] AUC={auc:.4f}  F1={f1:.4f}")
    return auc, f1


# ----------------------------------------------------------
# SAVE ONNX
# ----------------------------------------------------------
def save_onnx(model, n_features, out_path):
    print(f"[INFO] Exporting ONNX model to {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset={"ai.onnx.ml": 3, "": 13}   # "" is the main ONNX opset
)


    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("[OK] Saved ONNX model.")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    args = parse_args()

    print(f"[INFO] Loading dataset: {args.in_csv}")
    X, y = load_dataset(args.in_csv, args.label_col)

    # Save feature order
    feature_order_path = "data/models/features.json"
    os.makedirs("data/models", exist_ok=True)
    with open(feature_order_path, "w") as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"[OK] Saved feature order to {feature_order_path}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Save scaler
    scaler_path = "data/models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Saved scaler to {scaler_path}")

    # Train model
    model = train_model(X_train_scaled, args.contamination, args.random_state)

    # Evaluate
    evaluate_model(model, X_val_scaled, y_val)

    # Save ONNX
    save_onnx(model, X_train_scaled.shape[1], args.out_onnx)

    print("\n[DONE] Training + export complete.\n")


if __name__ == "__main__":
    main()
