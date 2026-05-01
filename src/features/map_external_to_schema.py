# src/features/map_external_to_schema.py

import argparse
import logging
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

DatasetType = Literal["CICIDS2017", "UNSW_NB15"]


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def detect_dataset_type(columns: List[str]) -> Optional[DatasetType]:
    """
    Try to guess the dataset type from the available columns.
    Extend this function if you add more external datasets later.
    """
    cols = set(columns)

    # CIC-IDS-2017 / CICFlowMeter-style
    if (
        "Flow Duration" in cols
        and "Total Fwd Packets" in cols
        and "Total Backward Packets" in cols
    ):
        return "CICIDS2017"

    # UNSW-NB15 common CSV format
    if {"dur", "spkts", "dpkts", "sbytes", "dbytes"}.issubset(cols):
        return "UNSW_NB15"

    return None

def normalize_label(raw_label) -> int:
    """
    Convert various dataset-specific labels into a binary label:

        0 -> benign / normal
        1 -> attack / anomaly

    Adjust this mapping if you later want multi-class labels.
    """
    if pd.isna(raw_label):
        return 1  # if unknown, be conservative and mark as attack

    s = str(raw_label).strip().upper()

    benign_tokens = {
        "BENIGN",
        "NORMAL",
        "LEGITIMATE",
        "NON-ATTACK",
        "0",
    }

    # Many datasets mark attack with explicit category names
    if s in benign_tokens:
        return 0
    else:
        return 1


def map_cicids2017(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CIC-IDS-2017 (CICFlowMeter) columns to the compact schema.

    Expected columns (typical):
        - Flow Duration (microseconds)
        - Total Fwd Packets
        - Total Backward Packets
        - Total Length of Fwd Packets
        - Total Length of Bwd Packets
        - Protocol
        - Source Port / Src Port
        - Destination Port / Dst Port
        - Label
    """
    # Duration: CICFlowMeter uses microseconds
    duration = pd.to_numeric(df["Flow Duration"], errors="coerce") / 1e6

    pkts_fwd = pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
    pkts_rev = pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)

    bytes_fwd = pd.to_numeric(df["Total Length of Fwd Packets"], errors="coerce").fillna(0)
    bytes_rev = pd.to_numeric(df["Total Length of Bwd Packets"], errors="coerce").fillna(0)

    # Packet / byte rates
    safe_duration = duration.replace(0, np.nan)
    pkt_rate = (pkts_fwd + pkts_rev) / safe_duration
    byte_rate = (bytes_fwd + bytes_rev) / safe_duration

    pkt_rate = pkt_rate.fillna(0)
    byte_rate = byte_rate.fillna(0)

    # Fwd / rev ratio: avoid division by zero
    fwd_rev_ratio = pkts_fwd / pkts_rev.replace(0, np.nan)
    fwd_rev_ratio = fwd_rev_ratio.replace([np.inf, -np.inf], np.nan).fillna(
        pkts_fwd  # if no reverse pkts, ratio ~ pkts_fwd
    )

    # Protocol
    proto = pd.to_numeric(df.get("Protocol", np.nan), errors="coerce").fillna(-1).astype(int)

    # Label
    if "Label" not in df.columns:
        raise ValueError("CIC-IDS-2017 CSV must contain a 'Label' column.")

    label = df["Label"].map(normalize_label).astype(int)

    mapped = pd.DataFrame(
        {
            "pkts_fwd": pkts_fwd.astype("int64"),
            "pkts_rev": pkts_rev.astype("int64"),
            "bytes_fwd": bytes_fwd.astype("int64"),
            "bytes_rev": bytes_rev.astype("int64"),
            "duration": duration.astype("float64"),
            "pkt_rate": pkt_rate.astype("float64"),
            "byte_rate": byte_rate.astype("float64"),
            "fwd_rev_ratio": fwd_rev_ratio.astype("float64"),
            "proto": proto.astype("int32"),
            "label": label.astype("int8"),
        }
    )

    return mapped


def map_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UNSW-NB15 CSV columns to the compact schema.

    Typical columns:
        dur, spkts, dpkts, sbytes, dbytes, proto, label / class
    """

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    def col(name: str) -> pd.Series:
        if name not in df.columns:
            raise KeyError(
                f"Expected column '{name}' in UNSW-NB15 CSV, "
                f"but available columns are: {list(df.columns)}"
            )
        return df[name]

    # --- Core flow fields ---
    duration = pd.to_numeric(col("dur"), errors="coerce").fillna(0.0)

    pkts_fwd = pd.to_numeric(col("spkts"), errors="coerce").fillna(0)
    pkts_rev = pd.to_numeric(col("dpkts"), errors="coerce").fillna(0)

    bytes_fwd = pd.to_numeric(col("sbytes"), errors="coerce").fillna(0)
    bytes_rev = pd.to_numeric(col("dbytes"), errors="coerce").fillna(0)

    safe_duration = duration.replace(0, np.nan)
    pkt_rate = ((pkts_fwd + pkts_rev) / safe_duration).fillna(0)
    byte_rate = ((bytes_fwd + bytes_rev) / safe_duration).fillna(0)

    fwd_rev_ratio = pkts_fwd / pkts_rev.replace(0, np.nan)
    fwd_rev_ratio = fwd_rev_ratio.replace([np.inf, -np.inf], np.nan).fillna(pkts_fwd)

    # --- Protocol ---
    proto_raw = df.get("proto")
    if proto_raw is None:
        proto = pd.Series([-1] * len(df), index=df.index, dtype="int32")
    elif np.issubdtype(proto_raw.dtype, np.number):
        proto = pd.to_numeric(proto_raw, errors="coerce").fillna(-1).astype(int)
    else:
        proto_map = {"TCP": 6, "UDP": 17, "ICMP": 1}
        proto = proto_raw.astype(str).str.upper().map(proto_map).fillna(0).astype(int)

    # --- Label: some UNSW variants use 'label', others 'class' ---
    if "label" in df.columns:
        label_col = "label"
    elif "class" in df.columns:
        label_col = "class"
    else:
        raise ValueError(
            "UNSW-NB15 CSV must contain a 'label' or 'class' column for ground truth."
        )

    label = df[label_col].map(normalize_label).astype(int)

    mapped = pd.DataFrame(
        {
            "pkts_fwd": pkts_fwd.astype("int64"),
            "pkts_rev": pkts_rev.astype("int64"),
            "bytes_fwd": bytes_fwd.astype("int64"),
            "bytes_rev": bytes_rev.astype("int64"),
            "duration": duration.astype("float64"),
            "pkt_rate": pkt_rate.astype("float64"),
            "byte_rate": byte_rate.astype("float64"),
            "fwd_rev_ratio": fwd_rev_ratio.astype("float64"),
            "proto": proto.astype("int32"),
            "label": label.astype("int8"),
        }
    )

    return mapped




def map_dataset(df: pd.DataFrame, dataset_type: DatasetType) -> pd.DataFrame:
    if dataset_type == "CICIDS2017":
        logging.info("Mapping CIC-IDS-2017 columns to compact schema...")
        return map_cicids2017(df)
    elif dataset_type == "UNSW_NB15":
        logging.info("Mapping UNSW-NB15 columns to compact schema...")
        return map_unsw_nb15(df)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map an external IDS dataset (e.g. CIC-IDS-2017, UNSW-NB15) "
            "to the compact feature schema used by the project."
        )
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Input CSV file (external dataset).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Output CSV file with mapped schema.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_type",
        choices=["CICIDS2017", "UNSW_NB15"],
        help="Explicit dataset type. If omitted, it will be auto-detected.",
    )
    parser.add_argument(
        "--sep",
        dest="sep",
        default=",",
        help="CSV separator for input file (default: ',').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logging.info(f"Reading input CSV: {args.input_path}")

    # Try UTF-8 first, then fall back to latin-1 if needed
    try:
        df = pd.read_csv(
            args.input_path,
            sep=args.sep,
            low_memory=False,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        logging.warning("UTF-8 decode failed, trying latin-1 encoding...")
        df = pd.read_csv(
            args.input_path,
            sep=args.sep,
            low_memory=False,
            encoding="latin-1",
        )

    dataset_type = args.dataset_type
    if dataset_type is None:
        logging.info("Attempting to auto-detect dataset type from columns...")
        detected = detect_dataset_type(list(df.columns))
        if detected is None:
            logging.error("Could not auto-detect dataset type from input CSV columns.")
            logging.error(f"Columns:\n{list(df.columns)}")
            raise SystemExit(1)
        dataset_type = detected
        logging.info(f"Detected dataset type: {dataset_type}")

    mapped_df = map_dataset(df, dataset_type=dataset_type)

    logging.info(f"Writing mapped features to: {args.output_path}")
    mapped_df.to_csv(args.output_path, index=False)
    logging.info("Done.")


if __name__ == "__main__":
    main()
