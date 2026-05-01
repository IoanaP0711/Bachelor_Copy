#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import onnxruntime as ort
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.realtime.enrichment import (
    benign_background_context,
    explain_traffic_class,
)
from src.realtime.explanations import (
    build_repeat_explanation,
    adjust_final_severity,
    build_explanation_bundle,
    make_display_label_with_reason,
)


# =========================================================
# Active realtime inference pipeline configuration
# =========================================================
# This file defines the single active realtime inference
# pipeline used for the bachelor thesis demo.
# The model, scaler, and threshold paths below are the
# active assets loaded for live inference and interpretation.
# No alternate realtime demo pipeline is used here.
# -------------------------
# Paths
# -------------------------
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/models/ae_features.json")
ONNX_PATH = os.getenv("ONNX_PATH", "data/models/ae.omx")
SCALER_PATH = os.getenv("SCALER_PATH", "data/models/ae_scaler.joblib")
THRESHOLD_JSON = os.getenv("THRESHOLD_JSON", "data/models/ae_threshold_bands.json")

ALERTS_MAX = 200
RECENT_MAX = 300
THROUGHPUT_WINDOW_S = 10

# -------------------------
# Repeated behavior memory
# -------------------------
RepeatKey = Tuple[str, str, str, str, int]

REPEAT_WINDOW_S = 45
REPEAT_MAX_EVENTS_PER_KEY = 50
REPEAT_REVIEW_PREV_COUNT = 1
REPEAT_PERSISTENT_PREV_COUNT = 2

recent_repeat_memory: Dict[RepeatKey, Deque[float]] = defaultdict(deque)

# -------------------------
# Common ports for context
# -------------------------
COMMON_DESKTOP_PORTS = {
    53, 67, 68, 80, 443, 8080, 8443,
    123, 1900, 5353, 5355,
    25, 465, 587, 993, 995, 110, 143,
}

# -------------------------
# Per-request API log
# -------------------------
API_LOG_PATH = os.getenv("API_LOG_PATH", "logs/api_predict.jsonl")


def append_jsonl(path: str, obj: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_feature_cols() -> List[str]:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError(f"{FEATURES_PATH} must be a JSON list of column names.")
    return cols


def load_bands() -> Optional[Dict[str, float]]:
    p = Path(THRESHOLD_JSON)
    if not p.exists():
        return None

    obj = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(obj, dict) and "bands" in obj:
        b = obj["bands"]
        return {
            "ok": float(b["ok"]),
            "warn": float(b["warn"]),
            "crit": float(b["crit"]),
        }

    if isinstance(obj, dict) and "threshold" in obj:
        t = float(obj["threshold"])
        return {"ok": t, "warn": t * 1.5, "crit": t * 2.0}

    if isinstance(obj, (int, float)):
        t = float(obj)
        return {"ok": t, "warn": t * 1.5, "crit": t * 2.0}

    return None


class AutoencoderOnnxScorer:
    def __init__(self, onnx_path: str, scaler_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.scaler = joblib.load(scaler_path)
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        shape = self.sess.get_inputs()[0].shape
        self.expected_dim = shape[1] if isinstance(shape[1], int) else None

    def score_one(self, x_raw: np.ndarray) -> float:
        x_raw = x_raw.reshape(1, -1)
        if self.expected_dim is not None and x_raw.shape[1] != self.expected_dim:
            raise ValueError(
                f"Feature mismatch: got {x_raw.shape[1]}, expected {self.expected_dim}"
            )

        x_scaled = self.scaler.transform(x_raw).astype(np.float32)
        x_hat = self.sess.run([self.output_name], {self.input_name: x_scaled})[0].astype(np.float32)
        mse = float(np.mean((x_scaled - x_hat) ** 2, axis=1)[0])
        return mse


def severity_from(score: float, bands: Optional[Dict[str, float]]) -> str:
    if not bands:
        return "UNKNOWN"

    ok = float(bands["ok"])
    warn = float(bands["warn"])
    crit = float(bands["crit"])

    if score < ok:
        return "OK"
    if score < warn:
        return "WARN"
    if score < crit:
        return "MED"
    return "CRIT"


def make_repeat_key(event: Dict[str, Any]) -> RepeatKey:
    src_ip = str(event.get("src_ip") or "?")
    dest_ip = str(event.get("dest_ip") or "?")
    proto = str(event.get("proto") or "?").upper()
    traffic_class = str(event.get("traffic_class") or "unknown").lower()

    dport_raw = event.get("dest_port")
    try:
        dest_port = int(dport_raw) if dport_raw not in (None, "") else 0
    except (TypeError, ValueError):
        dest_port = 0

    if traffic_class in {"local_discovery", "dns"}:
        dest_port = 0

    return (src_ip, dest_ip, proto, traffic_class, dest_port)


def check_repeated_behavior(
    memory: Dict[RepeatKey, Deque[float]],
    key: RepeatKey,
    now_ts: float,
    window_s: int = REPEAT_WINDOW_S,
    max_events_per_key: int = REPEAT_MAX_EVENTS_PER_KEY,
) -> Dict[str, Any]:
    dq = memory[key]

    while dq and (now_ts - dq[0] > window_s):
        dq.popleft()

    previous_count = len(dq)
    dq.append(now_ts)

    while len(dq) > max_events_per_key:
        dq.popleft()

    current_count = len(dq)

    if previous_count == 0:
        repeat_level = "single"
        suspicion_boost = 0
    elif previous_count < REPEAT_PERSISTENT_PREV_COUNT:
        repeat_level = "repeated"
        suspicion_boost = 1
    else:
        repeat_level = "persistent"
        suspicion_boost = 2

    return {
        "repeat_key": {
            "src_ip": key[0],
            "dest_ip": key[1],
            "proto": key[2],
            "traffic_class": key[3],
            "dest_port": key[4],
        },
        "repeat_window_s": int(window_s),
        "previous_count": int(previous_count),
        "current_count": int(current_count),
        "repeat_level": repeat_level,
        "suspicion_boost": int(suspicion_boost),
        "is_repeated": bool(previous_count > 0),
    }


def escalate_severity_one_level(sev: str) -> str:
    sev = str(sev or "").upper()
    order = ["OK", "WARN", "MED", "CRIT"]
    if sev not in order:
        return sev
    idx = order.index(sev)
    return order[min(idx + 1, len(order) - 1)]


def apply_repeat_review_logic(event: Dict[str, Any]) -> Dict[str, Any]:
    repeat_info = event.get("repeat_info") or {}
    repeat_level = str(repeat_info.get("repeat_level", "single")).lower()

    current_final = str(event.get("final_severity", event.get("severity", "UNKNOWN"))).upper()
    likely_benign = bool(event.get("likely_benign", False))
    traffic_class = str(event.get("traffic_class", "") or "").strip().lower()
    is_anom = bool(event.get("is_anom", False))

    dport_raw = event.get("dest_port")
    try:
        dport = int(dport_raw) if dport_raw not in (None, "") else None
    except (TypeError, ValueError):
        dport = None

    is_unknown = traffic_class in {"", "unknown", "other", "failed"}
    is_uncommon_port = dport is not None and dport not in COMMON_DESKTOP_PORTS

    is_strong_local_discovery = (
        traffic_class == "local_discovery"
        and likely_benign
        and not is_unknown
    )

    event["final_severity_before_repeat"] = current_final

    if not is_anom or repeat_level == "single":
        return event

    # Keep repeated SSDP / local discovery noise at the context-adjusted level.
    # This avoids over-reviewing very common local multicast discovery traffic.
    if repeat_level in {"repeated", "persistent"} and is_strong_local_discovery:
        event["final_severity"] = current_final

        prev_reason = str(event.get("final_severity_reason", "") or "").strip()
        extra = "repeated local discovery traffic kept at benign context-adjusted level"
        if prev_reason:
            event["final_severity_reason"] = f"{prev_reason}; {extra}"
        else:
            event["final_severity_reason"] = extra

        return event

    reasons: List[str] = []
    new_final = current_final

    if repeat_level == "repeated":
        if likely_benign:
            new_final = current_final
            reasons.append(
                "kept at the context-adjusted level because benign-looking behavior repeated but is not yet persistent"
            )
        elif is_unknown or is_uncommon_port:
            new_final = escalate_severity_one_level(current_final)
            reasons.append(
                "escalated because similar unknown or uncommon traffic repeated in the short window"
            )
        elif current_final == "OK":
            new_final = "WARN"
            reasons.append(
                "raised to WARN because similar anomalous traffic repeated in the short window"
            )

    elif repeat_level == "persistent":
        if likely_benign:
            if current_final == "OK":
                new_final = "WARN"
            else:
                new_final = current_final
            reasons.append(
                "kept visible because benign-looking anomalous behavior became persistent"
            )
        elif is_unknown or is_uncommon_port:
            if current_final == "OK":
                new_final = "MED"
            elif current_final == "WARN":
                new_final = "CRIT"
            else:
                new_final = escalate_severity_one_level(current_final)
            reasons.append(
                "escalated aggressively because unknown or uncommon traffic became persistent"
            )
        else:
            new_final = escalate_severity_one_level(current_final)
            reasons.append(
                "escalated because similar anomalous traffic became persistent"
            )

    event["final_severity"] = new_final

    prev_reason = str(event.get("final_severity_reason", "") or "").strip()
    extra_reason = "; ".join(reasons).strip()
    if extra_reason:
        event["final_severity_reason"] = f"{prev_reason}; {extra_reason}".strip("; ").strip()

    # Hard repetition floor, but NOT for strongly benign local discovery.
    if repeat_level in {"repeated", "persistent"} and not is_strong_local_discovery:
        current_final = str(event.get("final_severity", "")).upper()

        if current_final in {"OK", "WARN"}:
            event["final_severity"] = "MED"

            prev_reason = str(event.get("final_severity_reason", "") or "").strip()
            extra = "forced escalation due to repeated anomalous behavior"
            if prev_reason:
                event["final_severity_reason"] = f"{prev_reason}; {extra}"
            else:
                event["final_severity_reason"] = extra

    return event


def score_flow(req: "PredictRequest") -> tuple[np.ndarray, Dict[str, float], float, float]:
    if isinstance(req.features, dict):
        missing = [c for c in feature_cols if c not in req.features]
        if missing:
            raise ValueError(f"missing_features={len(missing)} ex={missing[:8]}")
        x = np.array([req.features[c] for c in feature_cols], dtype=np.float32)
        raw_map = req.features
    else:
        x = np.array(req.features, dtype=np.float32)
        if len(x) != len(feature_cols):
            raise ValueError(f"feature_length_mismatch got={len(x)} expected={len(feature_cols)}")
        raw_map = {feature_cols[i]: float(x[i]) for i in range(len(feature_cols))}

    t0 = time.perf_counter()
    score = ae.score_one(x)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return x, raw_map, float(score), float(infer_ms)


def build_raw_event(
    req: "PredictRequest",
    flow_id: str,
    score: float,
    infer_ms: float,
    total_ms: float,
    cpu: float,
    rss: float,
    now_ts: float,
) -> Dict[str, Any]:
    raw_severity = severity_from(score, bands)
    raw_severity = str(raw_severity).upper()
    raw_model_flag = bool(raw_severity not in ("OK", "UNKNOWN"))

    return {
        "ts_unix": req.ts_unix if req.ts_unix is not None else now_ts,
        "flow_id": flow_id,
        "ae_score": float(score),
        "bands": bands,
        "is_anom": raw_model_flag,
        "raw_model_flag": raw_model_flag,
        "raw_severity": raw_severity,
        "severity": raw_severity,
        "infer_ms": float(infer_ms),
        "total_ms": float(total_ms),
        "cpu_proc_pct": float(cpu),
        "rss_mb": float(rss),
        "throughput_fps": float(throughput_fps()),
        "src_ip": req.src_ip or "",
        "src_port": req.src_port if req.src_port is not None else "",
        "dest_ip": req.dest_ip or "",
        "dest_port": req.dest_port if req.dest_port is not None else "",
        "proto": (req.proto or "").upper(),
        "app_proto": req.app_proto or "",
        "direction": req.direction or "",
    }


def enrich_event_context(event: Dict[str, Any]) -> Dict[str, Any]:
    ctx = benign_background_context(event)
    event["traffic_class"] = ctx["traffic_class"]
    event["likely_benign"] = ctx["likely_benign"]
    event["benign_reason"] = ctx["benign_reason"]
    event["context_tags"] = ctx["context_tags"]

    event["traffic_note"] = explain_traffic_class(
        event["traffic_class"],
        event["raw_severity"],
        event["is_anom"],
    )
    return event


def attach_repeat_context(event: Dict[str, Any]) -> Dict[str, Any]:
    repeat_info = {
        "repeat_key": None,
        "repeat_window_s": REPEAT_WINDOW_S,
        "previous_count": 0,
        "current_count": 0,
        "repeat_level": "single",
        "suspicion_boost": 0,
        "is_repeated": False,
    }

    if event["is_anom"]:
        repeat_key = make_repeat_key(event)
        repeat_info = check_repeated_behavior(
            recent_repeat_memory,
            repeat_key,
            float(event["ts_unix"]),
        )

    event["repeat_info"] = repeat_info
    event["repeat_level"] = repeat_info["repeat_level"]
    event["repeat_count"] = repeat_info["current_count"]
    event["repeat_previous_count"] = repeat_info["previous_count"]
    event["repetition_key"] = repeat_info["repeat_key"]
    event["repeat_window_s"] = repeat_info["repeat_window_s"]
    event["repeat_explanation"] = build_repeat_explanation(event) if event["is_anom"] else ""
    return event


def apply_final_decision_logic(event: Dict[str, Any]) -> Dict[str, Any]:
    event["raw_severity"] = str(event.get("raw_severity", "UNKNOWN")).upper()

    event = adjust_final_severity(event)
    event["final_severity_after_context"] = str(
        event.get("final_severity", event["raw_severity"])
    ).upper()

    event = apply_repeat_review_logic(event)

    event["final_severity"] = str(
        event.get("final_severity", event["final_severity_after_context"])
    ).upper()

    event["severity"] = event["final_severity"]
    return event


def attach_explanations(event: Dict[str, Any]) -> Dict[str, Any]:
    bundle = build_explanation_bundle(event)

    event["summary"] = bundle["summary"]
    event["interpretation"] = bundle["interpretation"]
    event["explanation"] = bundle["explanation"]
    event["adjustment_reason"] = bundle["adjustment_reason"]
    event["possible_explanation"] = bundle.get("possible_explanation", "")
    event["what_to_check"] = bundle.get("what_to_check", "")

    # backward compatibility
    event["short_summary"] = bundle.get("short_summary", bundle["summary"])
    event["full_explanation"] = bundle.get("full_explanation", bundle["explanation"])

    return event


def attach_top_feature_errors(
    event: Dict[str, Any],
    x: np.ndarray,
    raw_map: Dict[str, float],
) -> Dict[str, Any]:
    x_scaled = ae.scaler.transform(x.reshape(1, -1)).astype(np.float32)
    x_hat = ae.sess.run([ae.output_name], {ae.input_name: x_scaled})[0].astype(np.float32)

    per_feat_err = ((x_scaled - x_hat) ** 2)[0]
    top_idx = np.argsort(per_feat_err)[::-1][:5]

    event["top_features"] = [
        {
            "name": feature_cols[i],
            "err": float(per_feat_err[i]),
            "x": float(x_scaled[0, i]),
            "x_hat": float(x_hat[0, i]),
        }
        for i in top_idx
    ]

    event["top_features_raw"] = [
        {"name": feature_cols[i], "raw": float(raw_map.get(feature_cols[i], float("nan")))}
        for i in top_idx
    ]
    return event


def assemble_final_event(event: Dict[str, Any]) -> Dict[str, Any]:
    repeat_info = event.get("repeat_info") or {}

    timing = {
        "infer_ms": float(event.get("infer_ms", 0.0)),
        "total_ms": float(event.get("total_ms", 0.0)),
        "throughput_fps": float(event.get("throughput_fps", 0.0)),
    }

    system = {
        "cpu_proc_pct": float(event.get("cpu_proc_pct", 0.0)),
        "rss_mb": float(event.get("rss_mb", 0.0)),
    }

    model = {
        "name": "autoencoder_onnx",
        "bands": event.get("bands"),
    }

    final_label = event.get("display_label", "REVIEW")
    final_severity = str(event.get("final_severity", event.get("severity", "UNKNOWN"))).upper()

    final_event = {
        # -------------------------
        # Core identity / flow metadata
        # -------------------------
        "ts_unix": float(event.get("ts_unix", time.time())),
        "flow_id": str(event.get("flow_id", "")),
        "src_ip": event.get("src_ip", ""),
        "src_port": event.get("src_port", ""),
        "dest_ip": event.get("dest_ip", ""),
        "dest_port": event.get("dest_port", ""),
        "proto": event.get("proto", ""),
        "app_proto": event.get("app_proto", ""),
        "direction": event.get("direction", ""),

        # -------------------------
        # Raw model evidence
        # -------------------------
        "ae_score": float(event.get("ae_score", 0.0)),
        "raw_severity": str(event.get("raw_severity", "UNKNOWN")).upper(),
        "raw_model_flag": bool(event.get("is_anom", False)),

        # -------------------------
        # Context
        # -------------------------
        "traffic_class": event.get("traffic_class", "unknown"),
        "likely_benign": bool(event.get("likely_benign", False)),
        "benign_reason": event.get("benign_reason", ""),
        "traffic_note": event.get("traffic_note", ""),
        "context_tags": event.get("context_tags", []),

        # -------------------------
        # Repetition
        # -------------------------
        "repeat_count": int(repeat_info.get("current_count", 0)),
        "repeat_previous_count": int(repeat_info.get("previous_count", 0)),
        "repeat_level": repeat_info.get("repeat_level", "single"),
        "repeat_window_s": int(repeat_info.get("repeat_window_s", REPEAT_WINDOW_S)),
        "repetition_key": repeat_info.get("repeat_key"),

        # -------------------------
        # Final decision
        # -------------------------
        "final_label": final_label,
        "final_severity": final_severity,
        "summary": event.get("summary", ""),
        "interpretation": event.get("interpretation", ""),
        "explanation": event.get("explanation", ""),
        "adjustment_reason": (
            event.get("adjustment_reason")
            or event.get("final_severity_reason")
            or ""
        ),
        "possible_explanation": event.get("possible_explanation", ""),
        "what_to_check": event.get("what_to_check", ""),

        # -------------------------
        # Backward compatibility / UI helpers
        # -------------------------
        "display_label": final_label,
        "display_label_reason": event.get("display_label_reason", ""),
        "severity": final_severity,

        # -------------------------
        # Thesis/debug evidence
        # -------------------------
        "top_features": event.get("top_features", []),
        "top_features_raw": event.get("top_features_raw", []),
        "timing": timing,
        "system": system,
        "model": model,
        "debug": {
            "is_anom": bool(event.get("is_anom", False)),
            "severity_before_repeat": event.get("final_severity_before_repeat", ""),
            "severity_after_context": event.get("final_severity_after_context", ""),
            "repeat_explanation": event.get("repeat_explanation", ""),
            "repeat_info": repeat_info,
        },
    }

    return final_event


REQ_TOTAL = Counter("rtids_requests_total", "Total /predict requests", ["status"])
INFER_MS = Histogram(
    "rtids_infer_latency_ms",
    "AE inference latency (ms)",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
)
TOTAL_MS = Histogram(
    "rtids_total_latency_ms",
    "Total request latency (ms)",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
)
CPU_PROC = Gauge("rtids_cpu_process_pct", "Process CPU %")
RSS_MB = Gauge("rtids_rss_mb", "Process RSS memory (MB)")
THROUGHPUT_FPS = Gauge("rtids_throughput_fps", "Requests/sec over rolling window")
ALERTS_BUFFERED = Gauge("rtids_alerts_buffered", "Alerts buffered in memory")

PROC = psutil.Process(os.getpid())
PROC.cpu_percent(None)

_done_ts: Deque[float] = deque()


def proc_stats() -> tuple[float, float]:
    cpu = PROC.cpu_percent(None)
    rss = PROC.memory_info().rss / (1024 * 1024)
    return cpu, rss


def _append_done(ts: float) -> None:
    _done_ts.append(ts)
    cutoff = ts - THROUGHPUT_WINDOW_S
    while _done_ts and _done_ts[0] < cutoff:
        _done_ts.popleft()


def throughput_fps() -> float:
    if len(_done_ts) < 2:
        return 0.0
    span = max(1e-6, _done_ts[-1] - _done_ts[0])
    return len(_done_ts) / span


def update_gauges(alerts_len: int) -> None:
    cpu, rss = proc_stats()
    CPU_PROC.set(cpu)
    RSS_MB.set(rss)
    THROUGHPUT_FPS.set(throughput_fps())
    ALERTS_BUFFERED.set(alerts_len)


app = FastAPI(title="RT-IDS AE Dashboard")

feature_cols = load_feature_cols()
bands = load_bands()
ae = AutoencoderOnnxScorer(ONNX_PATH, SCALER_PATH)

alerts: Deque[Dict[str, Any]] = deque(maxlen=ALERTS_MAX)
recent: Deque[Dict[str, Any]] = deque(maxlen=RECENT_MAX)


class PredictRequest(BaseModel):
    flow_id: str
    features: Dict[str, float]

    src_ip: Optional[str] = None
    src_port: Optional[int] = None
    dest_ip: Optional[str] = None
    dest_port: Optional[int] = None
    proto: Optional[str] = None
    app_proto: Optional[str] = None
    direction: Optional[str] = None
    ts_unix: Optional[float] = None


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RT-IDS Dashboard</title>
  <style>
    * { box-sizing: border-box; }

    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      margin: 24px;
      background: #fafafa;
    }

    h1 {
      margin: 0 0 6px 0;
      font-size: 26px;
      line-height: 1.2;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .sub {
      color:#555;
      margin-bottom: 16px;
      line-height: 1.45;
    }

    .toplinks {
      display: inline-flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .toplinks a {
      font-size: 14px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(160px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }

    .card {
      background: white;
      border: 1px solid #e5e5e5;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      min-width: 0;
    }

    .label {
      color:#666;
      font-size: 12px;
    }

    .value {
      font-size: 18px;
      font-weight: 600;
      margin-top: 4px;
      word-break: break-word;
    }

    .pill {
      display:inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      margin-left: 0;
      font-weight: 700;
      vertical-align: middle;
    }

    .pill.ok { background:#e8f5e9; color:#1b5e20; }
    .pill.benign { background:#e0f7fa; color:#006064; }
    .pill.review { background:#fff8e1; color:#e65100; }
    .pill.critical { background:#ffebee; color:#b71c1c; }

    .controls {
      display:flex;
      gap:10px;
      align-items:center;
      margin: 10px 0 16px 0;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid #ddd;
      background: white;
      padding: 8px 12px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      flex: 0 0 auto;
    }

    button:hover { background: #f3f3f3; }

    input {
      padding: 8px 10px;
      border: 1px solid #ddd;
      border-radius: 10px;
      width: 110px;
      max-width: 100%;
    }

    .table-hint {
      margin: 0 0 10px 0;
      color: #666;
      font-size: 13px;
    }

    .table-wrap {
      overflow-x: auto;
      overflow-y: hidden;
      border: 1px solid #e5e5e5;
      border-radius: 14px;
      background: white;
      -webkit-overflow-scrolling: touch;
      width: 100%;
    }

    table {
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      background:white;
      border: none;
      border-radius: 14px;
      overflow: hidden;
      table-layout: fixed;
    }

    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid #f0f0f0;
      font-size: 13px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }

    th {
      text-align: left;
      background: #fbfbfb;
      color:#444;
      font-size: 13px;
      white-space: nowrap;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    tr:last-child td { border-bottom: none; }

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
    }

    .muted { color:#666; }

    .row-ok       { background: #ffffff; }
    .row-benign   { background: #f8fdfe; }
    .row-review   { background: #fffdf7; }
    .row-critical { background: #fff4f4; }

    .compact {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .center { text-align: center; }

    .clickable-row {
      cursor: pointer;
    }

    .clickable-row:hover {
      filter: brightness(0.985);
    }

    .details-sticky {
      position: sticky;
      right: 0;
      background: inherit;
      z-index: 2;
      box-shadow: -6px 0 8px rgba(0,0,0,0.04);
    }

    th.details-sticky {
      background: #fbfbfb;
      z-index: 3;
    }

    .info-btn {
      border: 1px solid #cfcfcf;
      background: #f8f8f8;
      color: #333;
      border-radius: 999px;
      width: 24px;
      height: 24px;
      padding: 0;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      line-height: 22px;
      text-align: center;
      flex: 0 0 auto;
    }

    .info-btn:hover { background: #ececec; }

    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .badge-blue { background: #e3f2fd; color: #0d47a1; }
    .badge-cyan { background: #e0f7fa; color: #006064; }
    .badge-purple { background: #f3e5f5; color: #6a1b9a; }
    .badge-gray { background: #f3f4f6; color: #444; }
    .badge-ok { background: #e8f5e9; color: #1b5e20; }
    .badge-benign { background: #e0f7fa; color: #006064; }
    .badge-review { background: #fff8e1; color: #e65100; }
    .badge-critical { background: #ffebee; color: #b71c1c; }

    .summary-cell {
      min-width: 220px;
      max-width: 340px;
      white-space: normal;
      word-break: normal;
      overflow-wrap: break-word;
      line-height: 1.3;
    }

    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.35);
      z-index: 9998;
    }

    .modal-backdrop.show { display: block; }

    .modal {
      display: none;
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: min(980px, calc(100vw - 24px));
      height: min(760px, calc(100vh - 24px));
      min-width: 520px;
      min-height: 420px;
      max-width: calc(100vw - 16px);
      max-height: calc(100vh - 16px);
      overflow: auto;
      resize: both;
      background: white;
      border-radius: 16px;
      border: 1px solid #ddd;
      box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      z-index: 9999;
      padding: 18px;
    }

    .modal.show { display: block; }

    .modal-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }

    .modal-title {
      font-size: 18px;
      font-weight: 700;
      margin: 0;
    }

    .modal-close {
      border: 1px solid #ddd;
      background: white;
      border-radius: 10px;
      padding: 6px 10px;
      cursor: pointer;
      font-weight: 700;
    }

    .modal-section {
      border: 1px solid #ececec;
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
      background: #fcfcfc;
    }

    .modal-section h3 {
      margin: 0 0 10px 0;
      font-size: 15px;
    }

    .modal-grid {
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px 12px;
    }

    .modal-label {
      color: #666;
      font-size: 13px;
      font-weight: 600;
    }

    .modal-value {
      font-size: 14px;
      word-break: break-word;
      min-width: 0;
    }

    .modal-text {
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 14px;
      word-break: break-word;
    }

    .feature-list {
      margin: 0;
      padding-left: 18px;
    }

    .feature-list li {
      margin-bottom: 4px;
    }

    @media (max-width: 1200px) {
      .grid {
        grid-template-columns: repeat(3, minmax(180px, 1fr));
      }

      table {
        min-width: 900px;
      }

      .summary-cell {
        min-width: 160px;
        max-width: 220px;
      }
    }

    @media (max-width: 900px) {
      body {
        margin: 16px;
      }

      .grid {
        grid-template-columns: repeat(2, minmax(160px, 1fr));
      }

      .controls {
        align-items: stretch;
      }

      .controls input {
        width: 100%;
        max-width: 140px;
      }

      .modal-grid {
        grid-template-columns: 1fr;
      }

      .modal {
        padding: 16px;
        border-radius: 14px;
      }

      table {
        min-width: 840px;
      }
    }

    @media (max-width: 640px) {
      body {
        margin: 12px;
      }

      h1 {
        font-size: 22px;
      }

      .sub {
        font-size: 14px;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .controls {
        gap: 8px;
      }

      .controls button,
      .controls input {
        width: 100%;
        max-width: none;
      }

      .table-wrap {
        border-radius: 12px;
      }

      table {
        min-width: 760px;
      }

      th, td {
        padding: 9px 8px;
        font-size: 13px;
      }

      .summary-cell {
        min-width: 150px;
        max-width: 190px;
      }

      .modal {
        width: calc(100vw - 12px);
        height: calc(100vh - 12px);
        min-width: 0;
        min-height: 0;
        resize: none;
        padding: 14px;
      }

      .modal-title {
        font-size: 16px;
      }
    }
  </style>
</head>
<body>
  <h1>Real-time IDS Alerts <span class="pill" id="statusPill">…</span></h1>
  <div class="sub">
    Autoencoder anomaly detection with contextual interpretation layer
    <span class="toplinks">
      <a href="/health" target="_blank">/health</a>
      <a href="/alerts" target="_blank">/alerts</a>
      <a href="/recent" target="_blank">/recent</a>
      <a href="/metrics" target="_blank">/metrics</a>
    </span>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label">Bands</div>
      <div class="value" id="thr">—</div>
      <div class="label muted">Raw model bands (OK / WARN / MED / CRIT)</div>
    </div>
    <div class="card">
      <div class="label">Buffered alerts</div>
      <div class="value" id="buf">0</div>
      <div class="label muted">last 200 anomaly alerts kept</div>
    </div>
    <div class="card">
      <div class="label">Throughput</div>
      <div class="value" id="fps">0.0</div>
      <div class="label muted">req/sec (rolling)</div>
    </div>
    <div class="card">
      <div class="label">CPU</div>
      <div class="value" id="cpu">—</div>
      <div class="label muted">process CPU %</div>
    </div>
    <div class="card">
      <div class="label">Memory</div>
      <div class="value" id="rss">—</div>
      <div class="label muted">RSS MB</div>
    </div>
  </div>

  <div class="controls">
    <button id="toggleBtn" onclick="toggle()">Pause</button>
    <button onclick="clearAlerts()">Clear alerts</button>
    <span class="label">Limit:</span>
    <input id="limit" type="number" min="1" max="200" value="50"/>
    <span class="label muted" id="lastUpdate">—</span>
  </div>

  <div class="table-hint">Click a row to open the full alert explanation.</div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width: 150px;">Time</th>
          <th style="width: 220px;">Source</th>
          <th style="width: 220px;">Destination</th>
          <th style="width: 90px;">Proto</th>
          <th style="width: 90px;">Repeat</th>
          <th style="width: 130px;">Final label</th>
          <th style="width: 320px;">Summary</th>
          <th class="details-sticky" style="width: 90px;">Details</th>
        </tr>
      </thead>
      <tbody id="rows">
        <tr><td colspan="8" class="muted">Loading…</td></tr>
      </tbody>
    </table>
  </div>

  <div id="modalBackdrop" class="modal-backdrop" onclick="closeExplanationModal()"></div>

  <div id="explanationModal" class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
    <div class="modal-head">
      <h2 id="modalTitle" class="modal-title">Alert explanation</h2>
      <button class="modal-close" onclick="closeExplanationModal()">Close</button>
    </div>

    <div class="modal-section">
      <h3>Operational decision</h3>
      <div class="modal-grid">
        <div class="modal-label">Flow ID</div>
        <div class="modal-value mono" id="modalFlowId">-</div>

        <div class="modal-label">Final label</div>
        <div class="modal-value" id="modalDisplayLabel">-</div>

        <div class="modal-label">Final label reason</div>
        <div class="modal-value" id="modalDisplayLabelReason">-</div>

        <div class="modal-label">Interpretation</div>
        <div class="modal-value" id="modalInterpretation">-</div>

        <div class="modal-label">Summary</div>
        <div class="modal-value" id="modalSummary">-</div>

        <div class="modal-label">Raw model flag</div>
        <div class="modal-value" id="modalModelFlag">-</div>

        <div class="modal-label">Raw anomaly score</div>
        <div class="modal-value mono" id="modalScore">-</div>

        <div class="modal-label">Raw model severity</div>
        <div class="modal-value" id="modalRawSeverity">-</div>

        <div class="modal-label">Context-adjusted severity</div>
        <div class="modal-value" id="modalFinalSeverity">-</div>

        <div class="modal-label">Adjustment reason</div>
        <div class="modal-value" id="modalAdjustmentReason">-</div>

        <div class="modal-label">Possible explanation</div>
        <div class="modal-value" id="modalPossibleExplanation">-</div>

        <div class="modal-label">What to check</div>
        <div class="modal-value" id="modalWhatToCheck">-</div>
      </div>
    </div>

    <div class="modal-section">
      <h3>Context and repetition</h3>
      <div class="modal-grid">
        <div class="modal-label">Source</div>
        <div class="modal-value mono" id="modalSource">-</div>

        <div class="modal-label">Destination</div>
        <div class="modal-value mono" id="modalDestination">-</div>

        <div class="modal-label">Protocol</div>
        <div class="modal-value mono" id="modalProto">-</div>

        <div class="modal-label">Traffic class</div>
        <div class="modal-value mono" id="modalClass">-</div>

        <div class="modal-label">Traffic note</div>
        <div class="modal-value" id="modalTrafficNote">-</div>

        <div class="modal-label">Likely benign</div>
        <div class="modal-value" id="modalLikelyBenign">-</div>

        <div class="modal-label">Benign reason</div>
        <div class="modal-value" id="modalBenignReason">-</div>

        <div class="modal-label">Context tags</div>
        <div class="modal-value" id="modalContextTags">-</div>

        <div class="modal-label">Repeat level</div>
        <div class="modal-value" id="modalRepeatLevel">-</div>

        <div class="modal-label">Repeat count</div>
        <div class="modal-value" id="modalRepeatCount">-</div>

        <div class="modal-label">Repeat window (s)</div>
        <div class="modal-value" id="modalRepeatWindow">-</div>

        <div class="modal-label">Repetition key</div>
        <div class="modal-value mono" id="modalRepeatKey">-</div>
      </div>
    </div>

    <div class="modal-section">
      <h3>Technical evidence</h3>
      <div class="modal-grid">
        <div class="modal-label">Top contributing features</div>
        <div class="modal-value" id="modalTopFeatures">-</div>

        <div class="modal-label">Inference time (ms)</div>
        <div class="modal-value mono" id="modalInferMs">-</div>

        <div class="modal-label">Total time (ms)</div>
        <div class="modal-value mono" id="modalTotalMs">-</div>

        <div class="modal-label">Throughput (fps)</div>
        <div class="modal-value mono" id="modalThroughput">-</div>

        <div class="modal-label">CPU / RSS</div>
        <div class="modal-value mono" id="modalSystem">-</div>
      </div>
    </div>

    <div class="modal-section">
      <h3>Full explanation</h3>
      <div class="modal-text" id="modalExplanation">-</div>
    </div>
  </div>

<script>
let running = true;
let timer = null;
let latestRows = [];

function setStatusPillLabel(label) {
  const pill = document.getElementById('statusPill');
  const l = String(label || "").toUpperCase();

  if (l === "OK") {
    pill.className = "pill ok";
    pill.textContent = "OK";
  } else if (l === "BENIGN") {
    pill.className = "pill benign";
    pill.textContent = "BENIGN";
  } else if (l === "REVIEW") {
    pill.className = "pill review";
    pill.textContent = "REVIEW";
  } else if (l === "CRITICAL") {
    pill.className = "pill critical";
    pill.textContent = "CRITICAL";
  } else {
    pill.className = "pill";
    pill.textContent = "…";
  }
}

function rowClassFromLabel(label) {
  const l = String(label || "").toUpperCase();
  if (l === "OK") return "row-ok";
  if (l === "BENIGN") return "row-benign";
  if (l === "REVIEW") return "row-review";
  if (l === "CRITICAL") return "row-critical";
  return "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeText(value, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  return escapeHtml(value);
}

function safeNum(value, digits = 0, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  const n = Number(value);
  if (Number.isNaN(n)) return escapeHtml(value);
  return n.toFixed(digits);
}

function badge(text, cls = "") {
  return `<span class="badge ${cls}">${safeText(text)}</span>`;
}

function protoBadge(proto) {
  const p = String(proto ?? "").toUpperCase();
  if (!p) return badge("-", "badge-gray");
  if (p === "TCP") return badge(p, "badge-blue");
  if (p === "UDP") return badge(p, "badge-cyan");
  if (p === "ICMP" || p === "ICMPV6" || p === "IPV6-ICMP") return badge(p, "badge-purple");
  return badge(p, "badge-gray");
}

function displayLabelBadge(label) {
  const l = String(label ?? "").toUpperCase();

  if (l === "OK") return badge("OK", "badge-ok");
  if (l === "BENIGN") return badge("BENIGN", "badge-benign");
  if (l === "REVIEW") return badge("REVIEW", "badge-review");
  if (l === "CRITICAL") return badge("CRITICAL", "badge-critical");

  return badge(l || "-", "badge-gray");
}

function renderTopFeatures(features) {
  if (!Array.isArray(features) || !features.length) return "-";
  const items = features.map(f => {
    const name = safeText(f.name, "?");
    const err = safeNum(f.err, 4, "-");
    return `<li><span class="mono">${name}</span> (err=${err})</li>`;
  }).join("");
  return `<ul class="feature-list">${items}</ul>`;
}

function openExplanationModal(alertObj) {
  const repKey = alertObj.repetition_key
    ? JSON.stringify(alertObj.repetition_key)
    : "-";

  document.getElementById("modalFlowId").textContent = alertObj.flow_id ?? "-";
  document.getElementById("modalSource").textContent =
    `${alertObj.src_ip ?? "-"} : ${alertObj.src_port ?? "-"}`;
  document.getElementById("modalDestination").textContent =
    `${alertObj.dest_ip ?? "-"} : ${alertObj.dest_port ?? "-"}`;
  document.getElementById("modalProto").textContent =
    `${alertObj.proto ?? "-"} / ${alertObj.app_proto ?? "-"}`;
  document.getElementById("modalClass").textContent = alertObj.traffic_class ?? "-";
  document.getElementById("modalTrafficNote").textContent = alertObj.traffic_note ?? "-";
  document.getElementById("modalLikelyBenign").textContent = alertObj.likely_benign ? "YES" : "NO";
  document.getElementById("modalBenignReason").textContent = alertObj.benign_reason ?? "-";
  document.getElementById("modalContextTags").textContent =
    Array.isArray(alertObj.context_tags) && alertObj.context_tags.length
      ? alertObj.context_tags.join(", ")
      : "-";
  document.getElementById("modalRepeatLevel").textContent = alertObj.repeat_level ?? "-";
  document.getElementById("modalRepeatCount").textContent = String(alertObj.repeat_count ?? "-");
  document.getElementById("modalRepeatWindow").textContent = String(alertObj.repeat_window_s ?? "-");
  document.getElementById("modalRepeatKey").textContent = repKey;
  document.getElementById("modalDisplayLabel").textContent =
    alertObj.final_label ?? alertObj.display_label ?? "-";
  document.getElementById("modalDisplayLabelReason").textContent = alertObj.display_label_reason ?? "-";
  document.getElementById("modalInterpretation").textContent = alertObj.interpretation ?? "-";
  document.getElementById("modalSummary").textContent = alertObj.summary ?? "-";
  document.getElementById("modalModelFlag").textContent = alertObj.raw_model_flag ? "YES" : "NO";
  document.getElementById("modalRawSeverity").textContent = alertObj.raw_severity ?? "-";
  document.getElementById("modalFinalSeverity").textContent = alertObj.final_severity ?? alertObj.severity ?? "-";
  document.getElementById("modalAdjustmentReason").textContent = alertObj.adjustment_reason ?? "-";

  const finalLabel = String(alertObj.final_label ?? alertObj.display_label ?? "").toUpperCase();
  const showHints = finalLabel === "REVIEW" || finalLabel === "CRITICAL";

  document.getElementById("modalPossibleExplanation").textContent =
    showHints ? (alertObj.possible_explanation ?? "-") : "-";

  document.getElementById("modalWhatToCheck").textContent =
    showHints ? (alertObj.what_to_check ?? "-") : "-";

  document.getElementById("modalScore").textContent = safeNum(alertObj.ae_score, 6, "-");
  document.getElementById("modalInferMs").textContent = safeNum(alertObj.timing?.infer_ms, 3, "-");
  document.getElementById("modalTotalMs").textContent = safeNum(alertObj.timing?.total_ms, 3, "-");
  document.getElementById("modalThroughput").textContent = safeNum(alertObj.timing?.throughput_fps, 2, "-");
  document.getElementById("modalSystem").textContent =
    `CPU=${safeNum(alertObj.system?.cpu_proc_pct, 1, "-")}% | RSS=${safeNum(alertObj.system?.rss_mb, 1, "-")} MB`;
  document.getElementById("modalExplanation").textContent = alertObj.explanation ?? "-";
  document.getElementById("modalTopFeatures").innerHTML = renderTopFeatures(alertObj.top_features);

  document.getElementById("modalBackdrop").classList.add("show");
  document.getElementById("explanationModal").classList.add("show");
}

function closeExplanationModal() {
  document.getElementById("modalBackdrop").classList.remove("show");
  document.getElementById("explanationModal").classList.remove("show");
}

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    closeExplanationModal();
  }
});

async function fetchText(url) {
  const r = await fetch(url);
  return await r.text();
}

function parsePrometheus(text) {
  const out = {};
  const lines = text.split('\\n');
  for (const line of lines) {
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(' ');
    if (parts.length < 2) continue;
    const name = parts[0];
    const value = parseFloat(parts[1]);
    if (!isNaN(value)) out[name] = value;
  }
  return out;
}

async function refresh() {
  const limit = parseInt(document.getElementById('limit').value || "50", 10);

  const [recentRes, alertsRes] = await Promise.all([
    fetch(`/recent?limit=${limit}`),
    fetch(`/alerts?limit=${limit}`)
  ]);

  const recentData = await recentRes.json();
  const alertsData = await alertsRes.json();

  const b = recentData.bands;
  document.getElementById('thr').textContent =
    b ? `ok=${b.ok.toFixed(6)} warn=${b.warn.toFixed(6)} crit=${b.crit.toFixed(6)}` : "not set";

  document.getElementById('buf').textContent = (alertsData.alerts || []).length;

  const rows = recentData.recent || [];
  latestRows = rows;

  if (!rows.length) {
    setStatusPillLabel("OK");
  } else {
    const topLabel = rows[0].final_label || rows[0].display_label || "OK";
    setStatusPillLabel(topLabel);
  }

  const mText = await fetchText('/metrics');
  const m = parsePrometheus(mText);
  if (m['rtids_cpu_process_pct'] !== undefined) {
    document.getElementById('cpu').textContent = m['rtids_cpu_process_pct'].toFixed(1) + "%";
  }
  if (m['rtids_rss_mb'] !== undefined) {
    document.getElementById('rss').textContent = m['rtids_rss_mb'].toFixed(1) + " MB";
  }
  if (m['rtids_throughput_fps'] !== undefined) {
    document.getElementById('fps').textContent = m['rtids_throughput_fps'].toFixed(2);
  }

  const tbody = document.getElementById('rows');
  tbody.innerHTML = "";

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="muted">No traffic yet.</td></tr>`;
  } else {
    for (let idx = 0; idx < rows.length; idx++) {
      const a = rows[idx];
      const t = new Date((a.ts_unix ?? 0) * 1000)
        .toISOString()
        .replace('T', ' ')
        .replace('Z', 'Z')
        .split('.')[0] + 'Z';

      const source = `${a.src_ip ?? "-"}:${a.src_port ?? "-"}`;
      const destination = `${a.dest_ip ?? "-"}:${a.dest_port ?? "-"}`;
      const proto = a.proto || "-";
      const finalLabel = a.final_label || a.display_label || "-";
      const summary = a.summary || a.interpretation || "-";
      const repeatCount = a.repeat_count ?? 0;

      tbody.innerHTML += `
        <tr
          class="${rowClassFromLabel(finalLabel)} clickable-row"
          onclick="openExplanationModal(latestRows[${idx}])"
          title="Open alert explanation"
        >
          <td class="mono compact">${escapeHtml(t)}</td>
          <td class="mono compact">${escapeHtml(source)}</td>
          <td class="mono compact">${escapeHtml(destination)}</td>
          <td class="center">${protoBadge(proto)}</td>
          <td class="center"><span class="mono">${escapeHtml(String(repeatCount))}</span></td>
          <td class="center">${displayLabelBadge(finalLabel)}</td>
          <td class="summary-cell">${safeText(summary)}</td>
          <td class="center details-sticky">
            <button
              class="info-btn"
              onclick="event.stopPropagation(); openExplanationModal(latestRows[${idx}])"
              title="Show details"
            >i</button>
          </td>
        </tr>
      `;
    }
  }

  document.getElementById('lastUpdate').textContent =
    "Updated: " + new Date().toLocaleTimeString();
}

function toggle() {
  running = !running;
  const btn = document.getElementById('toggleBtn');
  btn.textContent = running ? "Pause" : "Resume";
  if (running) {
    refresh();
    timer = setInterval(refresh, 2000);
  } else {
    clearInterval(timer);
    timer = null;
  }
}

async function clearAlerts() {
  await fetch('/alerts/clear', { method: 'POST' });
  refresh();
}

timer = setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ONNX_PATH,
        "scaler": SCALER_PATH,
        "n_features": len(feature_cols),
        "bands": bands,
        "repeat_window_s": REPEAT_WINDOW_S,
        "display_labels": ["OK", "BENIGN", "REVIEW", "CRITICAL"],
        "log_path": API_LOG_PATH,
        "event_fields": [
            "ae_score",
            "raw_severity",
            "raw_model_flag",
            "traffic_class",
            "likely_benign",
            "benign_reason",
            "repeat_count",
            "repetition_key",
            "final_label",
            "final_severity",
            "summary",
            "interpretation",
            "explanation",
            "adjustment_reason",
            "possible_explanation",
            "what_to_check",
        ],
    }


@app.get("/stats")
def stats():
    update_gauges(len(alerts))
    return {
        "bands": bands,
        "alerts_buffered": len(alerts),
        "recent_buffered": len(recent),
        "cpu_proc_pct": float(CPU_PROC._value.get()),
        "rss_mb": float(RSS_MB._value.get()),
        "throughput_fps": float(THROUGHPUT_FPS._value.get()),
        "repeat_keys_buffered": len(recent_repeat_memory),
    }


@app.get("/metrics")
def metrics():
    update_gauges(len(alerts))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recent")
def get_recent(limit: int = 50):
    items = list(recent)[-limit:][::-1]
    return {"bands": bands, "recent": items}


@app.get("/alerts")
def get_alerts(limit: int = 50):
    update_gauges(len(alerts))
    items = list(alerts)[-limit:][::-1]
    return {"bands": bands, "alerts": items}


@app.post("/alerts/clear")
def clear_alerts():
    alerts.clear()
    recent.clear()
    recent_repeat_memory.clear()
    update_gauges(len(alerts))
    return {"status": "ok", "cleared": True}


@app.post("/predict")
def predict(req: PredictRequest):
    t_total0 = time.perf_counter()
    ts = time.time()
    flow_id = req.flow_id or f"ts_{int(ts * 1000)}"

    try:
        x, raw_map, score, infer_ms = score_flow(req)
        total_ms = (time.perf_counter() - t_total0) * 1000.0

        INFER_MS.observe(infer_ms)
        TOTAL_MS.observe(total_ms)
        REQ_TOTAL.labels("ok").inc()
        _append_done(ts)

        cpu, rss = proc_stats()

        # -------------------------
        # 1. Raw model event
        # -------------------------
        event = build_raw_event(
            req=req,
            flow_id=flow_id,
            score=score,
            infer_ms=infer_ms,
            total_ms=total_ms,
            cpu=cpu,
            rss=rss,
            now_ts=ts,
        )

        # -------------------------
        # 2. Context enrichment
        # -------------------------
        event = enrich_event_context(event)

        # -------------------------
        # 3. Repetition context
        # -------------------------
        event = attach_repeat_context(event)

        # -------------------------
        # 4. Final internal severity decision
        #    raw -> context-adjusted -> repeat-adjusted
        # -------------------------
        event = apply_final_decision_logic(event)

        # -------------------------
        # 5. Final display label
        # -------------------------
        event["display_label"], event["display_label_reason"] = make_display_label_with_reason(event)
        event["final_label"] = event["display_label"]
        event["severity"] = str(event.get("final_severity", event.get("severity", "UNKNOWN"))).upper()

        # -------------------------
        # 6. User-facing explanations
        # -------------------------
        event = attach_explanations(event)

        # -------------------------
        # 7. Thesis/debug evidence
        # -------------------------
        event = attach_top_feature_errors(event, x, raw_map)

        final_event = assemble_final_event(event)

        recent.append(final_event)
        if final_event["raw_model_flag"]:
            alerts.append(final_event)

        append_jsonl(API_LOG_PATH, final_event)
        update_gauges(len(alerts))

        return final_event

    except Exception as e:
        import traceback
        REQ_TOTAL.labels("error").inc()
        print("PREDICT ERROR:", repr(e))
        traceback.print_exc()
        return {"flow_id": flow_id, "error": str(e)}