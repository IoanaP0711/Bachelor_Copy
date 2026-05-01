#!/usr/bin/env python3
"""
Train a small Autoencoder for flow-based anomaly detection, then export to
ONNX and TFLite.

Example:

(venv) $ python3 src/models/train_ae.py \
    --in-csv data/processed/train.csv \
    --out-onnx data/models/ae.omx \
    --out-tflite data/models/ae.tflite \
    --label-col label
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
import tf2onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Autoencoder for anomaly detection and export to ONNX + TFLite."
    )
    parser.add_argument(
        "--in-csv",
        type=str,
        default="data/processed/train.csv",
        help="Input CSV with flow features (and optionally label column).",
    )
    parser.add_argument(
        "--out-onnx",
        type=str,
        default="data/models/ae.omx",  # using .omx as you requested
        help="Output ONNX file path for the trained AE.",
    )
    parser.add_argument(
        "--out-tflite",
        type=str,
        default="data/models/ae.tflite",
        help="Output TFLite file path for the trained AE.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="data/models/ae_scaler.joblib",
        help="Where to save the StandardScaler used for feature normalization.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of the label column (0=normal,1=anomaly). If missing, train on all rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
    "--model-size",
    type=str,
    default="small",
    choices=["small", "medium", "large"],
    help="Choose autoencoder size: small, medium, large"
    )

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def load_and_prepare_data(csv_path: str, label_col: str, test_size: float, random_state: int):
    logging.info(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if label_col in df.columns:
        logging.info(f"Found label column '{label_col}' – will try to train on normal (label=0) traffic only.")
        # Try to train only on normal traffic (label == 0)
        normal_mask = df[label_col] == 0
        if normal_mask.sum() == 0:
            logging.warning("No rows with label==0 found; training on all rows instead.")
            features_df = df.drop(columns=[label_col])
        else:
            features_df = df.loc[normal_mask].drop(columns=[label_col])
    else:
        logging.warning(f"Label column '{label_col}' not found; training on all rows.")
        features_df = df

    # Use only numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found in the CSV.")

    logging.info(f"Using {len(numeric_cols)} numeric feature columns.")
    X = features_df[numeric_cols].astype(np.float32).values

    X_train, X_val = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    logging.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return X_train, X_val, numeric_cols


def scale_data(X_train: np.ndarray, X_val: np.ndarray, scaler_path: str):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Ensure output directory exists
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved StandardScaler to {scaler_path}")

    return X_train_scaled, X_val_scaled


def build_autoencoder(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    latent = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(latent)
    outputs = tf.keras.layers.Dense(input_dim)(x)
    return tf.keras.Model(inputs, outputs, name="flow_autoencoder_small")


def build_autoencoder_medium(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="input")

    # Encoder
    x = tf.keras.layers.Dense(64, activation="relu", name="enc_dense1")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", name="enc_dense2")(x)
    latent = tf.keras.layers.Dense(16, activation="relu", name="latent")(x)

    # Decoder
    x = tf.keras.layers.Dense(32, activation="relu", name="dec_dense1")(latent)
    x = tf.keras.layers.Dense(64, activation="relu", name="dec_dense2")(x)
    outputs = tf.keras.layers.Dense(input_dim, name="recon")(x)

    return tf.keras.Model(inputs, outputs, name="flow_autoencoder_medium")

def build_autoencoder_large(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="input")

    # Encoder
    x = tf.keras.layers.Dense(128, activation="relu", name="enc_dense1")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="enc_dense2")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="enc_dense3")(x)
    latent = tf.keras.layers.Dense(16, activation="relu", name="latent")(x)

    # Decoder
    x = tf.keras.layers.Dense(32, activation="relu", name="dec_dense1")(latent)
    x = tf.keras.layers.Dense(64, activation="relu", name="dec_dense2")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dec_dense3")(x)
    outputs = tf.keras.layers.Dense(input_dim, name="recon")(x)

    return tf.keras.Model(inputs, outputs, name="flow_autoencoder_large")




def train_autoencoder(
    model: tf.keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    batch_size: int,
):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        X_train,  # target is the input itself
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # import pandas as pd  # (better: move this to the top of the file)

    val_loss = float(history.history["val_loss"][-1])
    logging.info(f"Final validation MSE: {val_loss:.6f}")

    # Save training history for plots
    hist_df = pd.DataFrame(history.history)
    hist_path = "data/models/ae_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    logging.info(f"Saved training history to {hist_path}")

    return model



def export_to_onnx(model: tf.keras.Model, input_dim: int, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Define input signature for the conversion
    input_signature = [
        tf.TensorSpec([None, input_dim], tf.float32, name="input"),
    ]

    logging.info("Converting Keras model to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
    )

    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logging.info(f"Saved ONNX model to {out_path}")


def export_to_tflite(model: tf.keras.Model, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Converting Keras model to TFLite (XLA disabled)...")

    # 🔥 crucial fix: disable XLA so TFLite doesn't crash
    tf.config.optimizer.set_jit(False)

    # optional: disable other grappler optimizers
    tf.config.experimental_run_functions_eagerly(True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # ensure no auto-fusion
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    logging.info(f"Saved TFLite model to {out_path}")


def main():
    args = parse_args()
    setup_logging()

    X_train, X_val, feature_cols = load_and_prepare_data(
        csv_path=args.in_csv,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train_scaled, X_val_scaled = scale_data(
        X_train, X_val, scaler_path=args.scaler_path
    )

    input_dim = X_train_scaled.shape[1]
    if args.model_size == "small":
        logging.info("Using SMALL autoencoder")
        model = build_autoencoder(input_dim)

    elif args.model_size == "medium":
        logging.info("Using MEDIUM autoencoder")
        model = build_autoencoder_medium(input_dim)

    elif args.model_size == "large":
        logging.info("Using LARGE autoencoder")
        model = build_autoencoder_large(input_dim)

    else:
        raise ValueError("Invalid model size")
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
    )

    model = train_autoencoder(
        model,
        X_train_scaled,
        X_val_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    export_to_onnx(model, input_dim=input_dim, out_path=args.out_onnx)
    export_to_tflite(model, out_path=args.out_tflite)

    logging.info("Done.")


if __name__ == "__main__":
    main()
