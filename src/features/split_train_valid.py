#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Split a mapped dataset into train and validation CSVs."
    )
    parser.add_argument("--in", dest="input_path", required=True,
                        help="Input mapped CSV (e.g. data/processed/unsw_training_mapped.csv)")
    parser.add_argument("--train-out", dest="train_out", required=True,
                        help="Output train CSV path (e.g. data/processed/train.csv)")
    parser.add_argument("--valid-out", dest="valid_out", required=True,
                        help="Output validation CSV path (e.g. data/processed/valid.csv)")
    parser.add_argument("--valid-size", type=float, default=0.2,
                        help="Fraction of samples for validation (default: 0.2 = 20%%)")

    args = parser.parse_args()

    print(f"[INFO] Reading mapped dataset: {args.input_path}")
    df = pd.read_csv(args.input_path)

    print(f"[INFO] Original rows: {len(df)}")
    # Shuffle for randomness but keep it reproducible
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    valid_size = args.valid_size
    n_valid = int(len(df) * valid_size)
    n_train = len(df) - n_valid

    print(f"[INFO] Splitting with valid_size={valid_size:.2f}")
    print(f"[INFO] -> Train rows: {n_train}")
    print(f"[INFO] -> Valid rows: {n_valid}")

    df_train = df.iloc[:n_train].copy()
    df_valid = df.iloc[n_train:].copy()

    df_train.to_csv(args.train_out, index=False)
    df_valid.to_csv(args.valid_out, index=False)

    print(f"[OK] Wrote train CSV to: {args.train_out}")
    print(f"[OK] Wrote valid CSV to: {args.valid_out}")

if __name__ == "__main__":
    main()
