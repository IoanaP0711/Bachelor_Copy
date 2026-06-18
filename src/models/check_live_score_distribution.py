import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.realtime.run_live_suricata import (
    build_features_from_suricata,
)


INPUT_PATH = Path("/tmp/eve_tail.jsonl")
FEATURES_PATH = Path("data/models/ae_features.json")
SCALER_PATH = Path("data/models/ae_scaler.joblib")
MODEL_PATH = Path("data/models/ae.omx")
OUTPUT_PATH = Path("reports/live_score_check.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with FEATURES_PATH.open("r", encoding="utf-8") as file:
        feature_names = json.load(file)

    scaler = joblib.load(SCALER_PATH)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    output_rows = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = build_features_from_suricata(event)

            if not payload:
                continue

            features = payload["features"]

            x_raw = np.array(
                [[features[name] for name in feature_names]],
                dtype=np.float32,
            )

            x_scaled = scaler.transform(
                x_raw
            ).astype(np.float32)

            reconstruction = session.run(
                None,
                {input_name: x_scaled},
            )[0]

            score = float(
                np.mean(
                    (x_scaled - reconstruction) ** 2
                )
            )

            row = {
                "line_number": line_number,
                "flow_id": payload.get("flow_id"),
                "src_ip": payload.get("src_ip"),
                "src_port": payload.get("src_port"),
                "dest_ip": payload.get("dest_ip"),
                "dest_port": payload.get("dest_port"),
                "proto": payload.get("proto"),
                "app_proto": payload.get("app_proto"),
                "direction": payload.get("direction"),
                "anomaly_score": score,
            }

            for name in feature_names:
                row[name] = features[name]

            output_rows.append(row)

    if not output_rows:
        raise ValueError(
            "No valid Suricata flow events were found."
        )

    dataframe = pd.DataFrame(output_rows)

    scores = dataframe["anomaly_score"].to_numpy()

    print("=" * 65)
    print("LIVE FLOW SCORE DISTRIBUTION")
    print("=" * 65)

    print(f"Evaluated flows: {len(scores)}")
    print(f"Minimum:         {np.min(scores):.10f}")
    print(f"Percentile 50:   {np.percentile(scores, 50):.10f}")
    print(f"Percentile 90:   {np.percentile(scores, 90):.10f}")
    print(f"Percentile 95:   {np.percentile(scores, 95):.10f}")
    print(f"Percentile 97:   {np.percentile(scores, 97):.10f}")
    print(f"Percentile 99:   {np.percentile(scores, 99):.10f}")
    print(f"Percentile 99.5: {np.percentile(scores, 99.5):.10f}")
    print(f"Maximum:         {np.max(scores):.10f}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()