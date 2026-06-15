from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import secrets
import time
from collections import Counter as CollectionCounter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import onnxruntime as ort
import psutil
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from itsdangerous import BadSignature, URLSafeSerializer
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
    build_recommended_action,
    make_display_label_with_reason,
)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG_PATH = Path(
    os.getenv(
        "RUNTIME_CONFIG_PATH",
        str(PROJECT_ROOT / "config" / "runtime_config.json"),
    )
)

# =========================================================
# Active realtime inference pipeline configuration
# =========================================================
# This file defines the single active realtime inference
# pipeline used for the bachelor thesis.
# The model, scaler, and threshold paths below are the
# active assets loaded for live inference and interpretation.
# No alternate realtime demo pipeline is used here.

FEATURES_PATH = os.getenv("FEATURES_PATH", "data/models/ae_features.json")
ONNX_PATH = os.getenv("ONNX_PATH", "data/models/ae.omx")
SCALER_PATH = os.getenv("SCALER_PATH", "data/models/ae_scaler.joblib")
THRESHOLD_JSON = os.getenv("THRESHOLD_JSON", "data/models/ae_threshold_bands.json")

ALERTS_MAX = 200
RECENT_MAX = 300
THROUGHPUT_WINDOW_S = 10

RepeatKey = Tuple[str, str, str, str, int]

# These values preserve the current behavior when the runtime
# configuration file is missing or invalid.
DEFAULT_REPEAT_WINDOW_S = 45
DEFAULT_REPEAT_CRITICAL_COUNT = 3
DEFAULT_NOTIFICATION_ENABLED = True
DEFAULT_NOTIFICATION_COOLDOWN_S = 300

REPEAT_MAX_EVENTS_PER_KEY = 50
REPEAT_REVIEW_PREV_COUNT = 1

recent_repeat_memory: Dict[RepeatKey, Deque[float]] = defaultdict(deque)

COMMON_DESKTOP_PORTS = {
    53, 67, 68, 80, 443, 8080, 8443,
    123, 1900, 5353, 5355,
    25, 465, 587, 993, 995, 110, 143,
}

API_LOG_PATH = os.getenv("API_LOG_PATH", "logs/api_predict.jsonl")


# =========================================================
# Live/replay operating mode
# =========================================================
# Replay mode is intended for controlled demonstrations and
# validation. It reuses previously saved event records and
# places them in the normal UI buffers. It does not modify
# model inference, thresholds, contextual filtering, or the
# repeat-based decision logic.

IDS_MODE = os.getenv(
    "IDS_MODE",
    "live",
).strip().lower()

if IDS_MODE not in {"live", "replay"}:
    print(
        f"INVALID IDS_MODE {IDS_MODE!r}; falling back to live mode."
    )
    IDS_MODE = "live"

IDS_REPLAY_FILE = os.getenv(
    "IDS_REPLAY_FILE",
    "",
).strip()

try:
    REPLAY_INTERVAL_S = max(
        0.1,
        float(
            os.getenv(
                "IDS_REPLAY_INTERVAL_SECONDS",
                "2.0",
            )
        ),
    )
except (TypeError, ValueError):
    REPLAY_INTERVAL_S = 2.0

REPLAY_LOOP = os.getenv(
    "IDS_REPLAY_LOOP",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

REPLAY_AUTO_START = os.getenv(
    "IDS_REPLAY_AUTO_START",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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



def _copy_thresholds(
    value: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None

    try:
        copied = {
            "ok": float(value["ok"]),
            "warn": float(value["warn"]),
            "crit": float(value["crit"]),
        }
    except (KeyError, TypeError, ValueError):
        return None

    if not all(np.isfinite(number) for number in copied.values()):
        return None

    if not (
        copied["ok"] >= 0
        and copied["ok"] < copied["warn"] < copied["crit"]
    ):
        return None

    return copied


def _default_runtime_config() -> Dict[str, Any]:
    """
    Build defaults from the behavior that existed before runtime_config.json.

    Threshold defaults are taken from the existing model threshold asset,
    while repeat and notification defaults preserve the current application
    values.
    """
    model_thresholds = _copy_thresholds(load_bands())

    # This fallback is used only if the model threshold file is unavailable.
    # Keep these numbers aligned with the active model configuration.
    if model_thresholds is None:
        model_thresholds = {
            "ok": 1.5,
            "warn": 3.0,
            "crit": 8.0,
        }

    return {
        "thresholds": model_thresholds,
        "repeat_logic": {
            "window_seconds": DEFAULT_REPEAT_WINDOW_S,
            "critical_repeat_count": DEFAULT_REPEAT_CRITICAL_COUNT,
        },
        "notifications": {
            "enabled": DEFAULT_NOTIFICATION_ENABLED,
            "cooldown_seconds": DEFAULT_NOTIFICATION_COOLDOWN_S,
        },
    }


def _safe_positive_int(
    value: Any,
    fallback: int,
    *,
    minimum: int = 1,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback

    if parsed < minimum:
        return fallback

    return parsed


def load_runtime_config(
    path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and validate config/runtime_config.json.

    A missing file, invalid JSON, invalid section, or invalid individual
    value falls back safely to the behavior-preserving defaults.
    """
    defaults = _default_runtime_config()
    loaded = {
        "thresholds": dict(defaults["thresholds"]),
        "repeat_logic": dict(defaults["repeat_logic"]),
        "notifications": dict(defaults["notifications"]),
    }

    metadata: Dict[str, Any] = {
        "source": "defaults",
        "path": str(path),
        "warnings": [],
    }

    if not path.is_file():
        metadata["warnings"].append(
            "Runtime config file was not found; defaults are active."
        )
        return loaded, metadata

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        metadata["warnings"].append(
            f"Runtime config could not be read; defaults are active: {exc}"
        )
        return loaded, metadata

    if not isinstance(parsed, dict):
        metadata["warnings"].append(
            "Runtime config root must be a JSON object; defaults are active."
        )
        return loaded, metadata

    metadata["source"] = "file"

    threshold_section = parsed.get("thresholds")
    validated_thresholds = _copy_thresholds(threshold_section)

    if validated_thresholds is not None:
        loaded["thresholds"] = validated_thresholds
    elif threshold_section is not None:
        metadata["warnings"].append(
            "Invalid thresholds section; default thresholds are active."
        )

    repeat_section = parsed.get("repeat_logic")
    if isinstance(repeat_section, dict):
        loaded["repeat_logic"]["window_seconds"] = _safe_positive_int(
            repeat_section.get("window_seconds"),
            loaded["repeat_logic"]["window_seconds"],
        )
        loaded["repeat_logic"]["critical_repeat_count"] = _safe_positive_int(
            repeat_section.get("critical_repeat_count"),
            loaded["repeat_logic"]["critical_repeat_count"],
            minimum=2,
        )
    elif repeat_section is not None:
        metadata["warnings"].append(
            "Invalid repeat_logic section; repeat defaults are active."
        )

    notification_section = parsed.get("notifications")
    if isinstance(notification_section, dict):
        enabled_value = notification_section.get("enabled")
        if isinstance(enabled_value, bool):
            loaded["notifications"]["enabled"] = enabled_value
        elif enabled_value is not None:
            metadata["warnings"].append(
                "notifications.enabled must be true or false; default is active."
            )

        loaded["notifications"]["cooldown_seconds"] = _safe_positive_int(
            notification_section.get("cooldown_seconds"),
            loaded["notifications"]["cooldown_seconds"],
            minimum=0,
        )
    elif notification_section is not None:
        metadata["warnings"].append(
            "Invalid notifications section; notification defaults are active."
        )

    return loaded, metadata


RUNTIME_CONFIG, RUNTIME_CONFIG_META = load_runtime_config(
    RUNTIME_CONFIG_PATH
)

bands = dict(RUNTIME_CONFIG["thresholds"])

REPEAT_WINDOW_S = int(
    RUNTIME_CONFIG["repeat_logic"]["window_seconds"]
)
REPEAT_CRITICAL_COUNT = int(
    RUNTIME_CONFIG["repeat_logic"]["critical_repeat_count"]
)

# check_repeated_behavior compares the number of previous events.
# A configured critical count of 3 therefore means persistent behavior
# begins when the current event is the third matching event.
REPEAT_PERSISTENT_PREV_COUNT = max(
    1,
    REPEAT_CRITICAL_COUNT - 1,
)

NOTIFICATIONS_ENABLED = bool(
    RUNTIME_CONFIG["notifications"]["enabled"]
)
NOTIFICATION_COOLDOWN_S = int(
    RUNTIME_CONFIG["notifications"]["cooldown_seconds"]
)


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

    event["summary"] = bundle.get("summary", "")
    event["interpretation"] = bundle.get("interpretation", "")

    # Compatibility field.
    # This is now intentionally the simple explanation, so old compact popups stay readable.
    event["explanation"] = bundle.get(
        "simple_explanation",
        bundle.get("explanation", ""),
    )

    # New layered explanation fields.
    event["simple_explanation"] = bundle.get(
        "simple_explanation",
        event["explanation"],
    )
    event["analyst_explanation"] = bundle.get(
        "analyst_explanation",
        event["simple_explanation"],
    )
    event["technical_explanation"] = bundle.get(
        "technical_explanation",
        bundle.get("full_explanation", event["simple_explanation"]),
    )

    # Keep old fields for compatibility.
    event["short_summary"] = bundle.get("short_summary", event["summary"])
    event["full_explanation"] = bundle.get(
        "full_explanation",
        event["technical_explanation"],
    )
    event["legacy_explanation"] = bundle.get("legacy_explanation", "")

    event["adjustment_reason"] = bundle.get(
    "adjustment_reason",
    "",
)

    event["possible_explanation"] = bundle.get(
        "possible_explanation",
        "",
    )

    event["what_to_check"] = bundle.get(
        "what_to_check",
        "",
    )

    # The backend normally receives this value from the explanation bundle.
    # The helper is also called as a safe fallback for older or incomplete data.
    event["recommended_action"] = (
        bundle.get("recommended_action")
        or build_recommended_action(event)
    )

    return event

def attach_top_feature_errors(
    event: Dict[str, Any],
    x: np.ndarray,
    raw_map: Dict[str, float],
) -> Dict[str, Any]:
    x_scaled = ae.scaler.transform(
        x.reshape(1, -1)
    ).astype(np.float32)

    x_hat = ae.sess.run(
        [ae.output_name],
        {
            ae.input_name: x_scaled,
        },
    )[0].astype(np.float32)

    per_feat_err = (
        (x_scaled - x_hat) ** 2
    )[0]

    sorted_idx = np.argsort(
        per_feat_err
    )[::-1]

    # Complete contributor list used only by richer views,
    # such as the full alert detail page.
    #
    # The contributor calculation itself is unchanged.
    event["feature_contributors"] = [
        {
            "name": feature_cols[i],
            "err": float(
                per_feat_err[i]
            ),
            "x": float(
                x_scaled[0, i]
            ),
            "x_hat": float(
                x_hat[0, i]
            ),
        }
        for i in sorted_idx
    ]

    # Preserve the existing compact top-five field.
    # The compact popup can continue showing only its
    # current top-three selection from this field.
    top_idx = sorted_idx[:5]

    event["top_features"] = (
        event["feature_contributors"][:5]
    )

    event["top_features_raw"] = [
        {
            "name": feature_cols[i],
            "raw": float(
                raw_map.get(
                    feature_cols[i],
                    float("nan"),
                )
            ),
        }
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
        "ts_unix": float(event.get("ts_unix", time.time())),
        "flow_id": str(event.get("flow_id", "")),
        "src_ip": event.get("src_ip", ""),
        "src_port": event.get("src_port", ""),
        "dest_ip": event.get("dest_ip", ""),
        "dest_port": event.get("dest_port", ""),
        "proto": event.get("proto", ""),
        "app_proto": event.get("app_proto", ""),
        "direction": event.get("direction", ""),

        
        "ae_score": float(event.get("ae_score", 0.0)),
        "raw_severity": str(event.get("raw_severity", "UNKNOWN")).upper(),
        "raw_model_flag": bool(event.get("is_anom", False)),

        
        "traffic_class": event.get("traffic_class", "unknown"),
        "likely_benign": bool(event.get("likely_benign", False)),
        "benign_reason": event.get("benign_reason", ""),
        "traffic_note": event.get("traffic_note", ""),
        "context_tags": event.get("context_tags", []),

        
        "repeat_count": int(repeat_info.get("current_count", 0)),
        "repeat_previous_count": int(repeat_info.get("previous_count", 0)),
        "repeat_level": repeat_info.get("repeat_level", "single"),
        "repeat_window_s": int(repeat_info.get("repeat_window_s", REPEAT_WINDOW_S)),
        "repetition_key": repeat_info.get("repeat_key"),

        
        "final_label": final_label,
        "final_severity": final_severity,
                "summary": event.get("summary", ""),
        "interpretation": event.get("interpretation", ""),

        
        "explanation": event.get("simple_explanation", event.get("explanation", "")),

        
        "simple_explanation": event.get(
            "simple_explanation",
            event.get("explanation", ""),
        ),
        "analyst_explanation": event.get(
            "analyst_explanation",
            event.get("simple_explanation", event.get("explanation", "")),
        ),
        "technical_explanation": event.get(
            "technical_explanation",
            event.get("full_explanation", event.get("explanation", "")),
        ),

        
        "short_summary": event.get("short_summary", event.get("summary", "")),
        "full_explanation": event.get(
            "full_explanation",
            event.get("technical_explanation", event.get("explanation", "")),
        ),
        "legacy_explanation": event.get("legacy_explanation", ""),

        "adjustment_reason": (
            event.get("adjustment_reason")
            or event.get("final_severity_reason")
            or ""
        ),
        "possible_explanation": event.get(
            "possible_explanation",
            "",
        ),
        "what_to_check": event.get(
            "what_to_check",
            "",
        ),
        "recommended_action": (
            event.get("recommended_action")
            or build_recommended_action(event)
        ),

        
        "display_label": final_label,
        "display_label_reason": event.get("display_label_reason", ""),
        "severity": final_severity,

        
        "top_features": event.get(
            "top_features",
            [],
        ),
        "top_features_raw": event.get(
            "top_features_raw",
            [],
        ),
        "feature_contributors": event.get(
            "feature_contributors",
            event.get(
                "top_features",
                [],
            ),
        ),
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

DASHBOARD_USER = os.getenv("IDS_DASHBOARD_USER", "admin")
DASHBOARD_PASSWORD = os.getenv("IDS_DASHBOARD_PASSWORD", "admin")
IDS_API_KEY = os.getenv("IDS_API_KEY", "")
IDS_SESSION_SECRET = os.getenv("IDS_SESSION_SECRET", "dev-session-secret-change-me")

SESSION_COOKIE_NAME = "ids_dashboard_session"
SESSION_SERIALIZER = URLSafeSerializer(
    IDS_SESSION_SECRET,
    salt="ids-dashboard-session",
)

def create_session_token(username: str) -> str:
    return SESSION_SERIALIZER.dumps({"user": username})


def read_session_token(token: str) -> dict | None:
    try:
        data = SESSION_SERIALIZER.loads(token)
        if isinstance(data, dict) and data.get("user") == DASHBOARD_USER:
            return data
    except BadSignature:
        return None
    return None


def require_dashboard_login(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)

    if not token or not read_session_token(token):
        raise HTTPException(status_code=401, detail="Authentication required")

    return True


def require_api_key(x_api_key: str = Header(default="")):
    if not IDS_API_KEY:
        raise HTTPException(status_code=500, detail="API key is not configured")

    if not secrets.compare_digest(x_api_key, IDS_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return True

REPORTS_DIR = PROJECT_ROOT / "reports"
FEEDBACK_HISTORY_PATH = PROJECT_ROOT / "data" / "feedback_history.jsonl"
FEEDBACK_HISTORY_LOCK = Lock()

ALERT_STATUS_PATH = PROJECT_ROOT / "data" / "alert_status.json"
ALERT_STATUS_LOCK = Lock()

ALERT_HUMAN_STATUSES = (
    "New",
    "Seen",
    "Under review",
    "Resolved",
    "False positive",
)

SAVE_JSON_REPORT_COPIES = os.getenv(
    "SAVE_JSON_REPORT_COPIES",
    "1",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

METRICS_TIME_BUCKET_S = 60
METRICS_FILE_MAX_RECORDS = 5000

METRICS_SEARCH_DIRECTORIES = (
    PROJECT_ROOT,
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "metrics",
    PROJECT_ROOT / "validation_results",
)

app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "static"),
    name="static",
)

templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

feature_cols = load_feature_cols()
ae = AutoencoderOnnxScorer(ONNX_PATH, SCALER_PATH)

alerts: Deque[Dict[str, Any]] = deque(maxlen=ALERTS_MAX)
recent: Deque[Dict[str, Any]] = deque(maxlen=RECENT_MAX)


# Replay state is deliberately separate from the detection
# state. The replay worker only inserts saved, already formed
# event records into the existing alerts/recent UI buffers.
REPLAY_STATE_LOCK = Lock()
REPLAY_STOP_EVENT = threading.Event()
REPLAY_THREAD: Optional[threading.Thread] = None

REPLAY_STATE: Dict[str, Any] = {
    "running": False,
    "source_file": None,
    "loaded_events": 0,
    "emitted_events": 0,
    "started_at": None,
    "stopped_at": None,
    "last_error": None,
}


def _replay_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _replay_bool_anomaly(record: Dict[str, Any]) -> bool:
    raw = str(
        record.get("raw_severity")
        or record.get("severity")
        or ""
    ).strip().upper()

    final_label = str(
        record.get("final_label")
        or record.get("display_label")
        or record.get("final_decision")
        or record.get("final_severity")
        or ""
    ).strip().upper()

    return bool(
        record.get("raw_model_flag")
        or record.get("is_anom")
        or raw in {"WARN", "MED", "CRIT"}
        or final_label in {"REVIEW", "CRITICAL"}
    )


def _replay_candidate_files() -> List[Path]:
    candidates: List[Path] = []

    if IDS_REPLAY_FILE:
        configured = Path(IDS_REPLAY_FILE).expanduser()

        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured

        candidates.append(configured)

    candidates.extend(
        [
            PROJECT_ROOT / "data" / "replay_events.jsonl",
            PROJECT_ROOT / "data" / "replay_events.json",
        ]
    )

    preferred_names = (
        "api_predict.jsonl",
        "metrics_flow.jsonl",
        "review_critical_examples.json",
        "alerts.jsonl",
        "eve.json",
    )

    for root in (
        PROJECT_ROOT / "validation_results",
        PROJECT_ROOT / "logs",
    ):
        if not root.is_dir():
            continue

        for name in preferred_names:
            candidates.extend(
                sorted(root.rglob(name))
            )

    unique: List[Path] = []
    seen = set()

    for candidate in candidates:
        key = str(
            candidate.resolve(strict=False)
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

    return unique


def _find_replay_source() -> Optional[Path]:
    for candidate in _replay_candidate_files():
        try:
            if (
                candidate.is_file()
                and candidate.stat().st_size > 0
            ):
                return candidate
        except OSError:
            continue

    return None


def _extract_replay_records(
    value: Any,
) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    if not isinstance(value, dict):
        return []

    for key in (
        "alerts",
        "events",
        "records",
        "results",
        "predictions",
        "examples",
        "data",
    ):
        nested = value.get(key)

        if isinstance(nested, list):
            return [
                item
                for item in nested
                if isinstance(item, dict)
            ]

    return [value]


def _read_replay_records(
    path: Path,
) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                f"Could not read replay JSON: {exc}"
            ) from exc

        return _extract_replay_records(
            parsed
        )

    records: List[Dict[str, Any]] = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as replay_file:
            for line_number, line in enumerate(
                replay_file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        "[replay] Skipping invalid JSON "
                        f"line {line_number} in {path}."
                    )
                    continue

                records.extend(
                    _extract_replay_records(
                        parsed
                    )
                )
    except OSError as exc:
        raise ValueError(
            f"Could not read replay JSONL: {exc}"
        ) from exc

    return records


def _unwrap_replay_record(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    copied = dict(record)

    for key in (
        "event",
        "alert",
        "prediction",
        "result",
        "data",
    ):
        nested = copied.get(key)

        if isinstance(nested, dict):
            copied.pop(key, None)
            copied.update(nested)
            break

    return copied


def _safe_replay_float(
    value: Any,
    default: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default

    return number


def _normalise_replay_event(
    record: Dict[str, Any],
    sequence: int,
    source_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Convert a saved event into the same public event shape used
    by /recent, /alerts, alert details, reports, and explanations.

    Saved final decisions are preserved. Contextual filtering and
    model inference are intentionally not executed again.
    """
    event = _unwrap_replay_record(
        record
    )

    # Ignore aggregate-only metrics records.
    if any(
        key in event
        for key in (
            "severity_distribution",
            "traffic_class_distribution",
            "average_anomaly_score_over_time",
        )
    ) and not any(
        key in event
        for key in (
            "flow_id",
            "src_ip",
            "source_ip",
            "dest_ip",
            "destination_ip",
        )
    ):
        return None

    now_ts = time.time()

    original_timestamp = _first_present(
        event,
        "ts_unix",
        "timestamp",
        "time",
        "created_at",
    )

    original_id = _first_present(
        event,
        "flow_id",
        "alert_id",
        "event_id",
        "id",
    )

    flow_id = (
        f"replay-{sequence}-"
        f"{int(now_ts * 1000)}"
    )

    raw_severity = str(
        _first_present(
            event,
            "raw_severity",
            "raw_model_severity",
            "severity",
        )
        or "UNKNOWN"
    ).upper()

    final_label = _normalise_final_label(
        _first_present(
            event,
            "final_label",
            "display_label",
            "final_decision",
            "final_severity",
            "severity",
        )
        or raw_severity
    )

    if final_label == "UNKNOWN":
        final_label = (
            "CRITICAL"
            if raw_severity == "CRIT"
            else "REVIEW"
            if raw_severity in {"WARN", "MED"}
            else "OK"
        )

    src_ip = _first_present(
        event,
        "src_ip",
        "source_ip",
        "src",
    )

    dest_ip = _first_present(
        event,
        "dest_ip",
        "destination_ip",
        "dst_ip",
        "destination",
    )

    src_port = _first_present(
        event,
        "src_port",
        "source_port",
        "sport",
    )

    dest_port = _first_present(
        event,
        "dest_port",
        "destination_port",
        "dport",
    )

    proto = _first_present(
        event,
        "proto",
        "protocol",
    )

    ae_score = _safe_replay_float(
        _first_present(
            event,
            "ae_score",
            "anomaly_score",
            "score",
            "reconstruction_error",
        ),
        0.0,
    )

    replay_event: Dict[str, Any] = {
        **event,
        "ts_unix": now_ts,
        "flow_id": flow_id,
        "src_ip": str(src_ip or ""),
        "src_port": src_port if src_port is not None else "",
        "dest_ip": str(dest_ip or ""),
        "dest_port": dest_port if dest_port is not None else "",
        "proto": str(proto or "").upper(),
        "app_proto": str(
            _first_present(
                event,
                "app_proto",
                "application",
                "service",
            )
            or ""
        ),
        "direction": str(
            event.get("direction")
            or ""
        ),
        "ae_score": ae_score,
        "bands": event.get("bands") or bands,
        "raw_severity": raw_severity,
        "raw_model_flag": _replay_bool_anomaly(
            {
                **event,
                "raw_severity": raw_severity,
                "final_label": final_label,
            }
        ),
        "final_label": final_label,
        "display_label": final_label,
        "final_severity": str(
            event.get("final_severity")
            or final_label
        ).upper(),
        "severity": str(
            event.get("final_severity")
            or final_label
        ).upper(),
        "traffic_class": str(
            event.get("traffic_class")
            or "unknown"
        ),
        "likely_benign": bool(
            event.get("likely_benign", False)
        ),
        "repeat_count": int(
            _safe_replay_float(
                event.get("repeat_count"),
                0.0,
            )
        ),
        "repeat_previous_count": int(
            _safe_replay_float(
                event.get(
                    "repeat_previous_count"
                ),
                0.0,
            )
        ),
        "repeat_level": str(
            event.get("repeat_level")
            or "single"
        ),
        "repeat_window_s": int(
            _safe_replay_float(
                event.get("repeat_window_s"),
                float(REPEAT_WINDOW_S),
            )
        ),
        "timing": event.get("timing")
        if isinstance(event.get("timing"), dict)
        else {
            "infer_ms": 0.0,
            "total_ms": 0.0,
            "throughput_fps": 0.0,
        },
        "system": event.get("system")
        if isinstance(event.get("system"), dict)
        else {
            "cpu_proc_pct": 0.0,
            "rss_mb": 0.0,
        },
        "model": event.get("model")
        if isinstance(event.get("model"), dict)
        else {
            "name": "saved_replay_event",
            "bands": bands,
        },
        "replay_mode": True,
        "replay_sequence": sequence,
        "replay_original_id": (
            str(original_id)
            if original_id is not None
            else None
        ),
        "replay_original_timestamp": (
            original_timestamp
        ),
        "replay_source_file": str(
            source_path
        ),
    }

    if not replay_event.get("summary"):
        replay_event["summary"] = (
            "Saved event replayed for a controlled "
            "demonstration."
        )

    if not replay_event.get(
        "display_label_reason"
    ):
        replay_event[
            "display_label_reason"
        ] = replay_event.get(
            "adjustment_reason"
        ) or (
            "Previously saved decision preserved "
            "during replay."
        )

    if not replay_event.get(
        "simple_explanation"
    ):
        replay_event["simple_explanation"] = (
            replay_event.get("explanation")
            or (
                "This previously saved event is being "
                "shown in replay mode for demonstration "
                "and validation."
            )
        )

    # Reuse the normal explanation builder only when the saved
    # event does not already contain the richer explanation fields.
    if not (
        replay_event.get("analyst_explanation")
        and replay_event.get(
            "technical_explanation"
        )
    ):
        try:
            replay_event = attach_explanations(
                replay_event
            )
        except Exception as exc:
            print(
                "[replay] Explanation fallback failed:",
                repr(exc),
            )

    replay_event.setdefault(
        "explanation",
        replay_event.get(
            "simple_explanation",
            "",
        ),
    )
    replay_event.setdefault(
        "analyst_explanation",
        replay_event.get(
            "simple_explanation",
            "",
        ),
    )
    replay_event.setdefault(
        "technical_explanation",
        replay_event.get(
            "simple_explanation",
            "",
        ),
    )
    replay_event.setdefault(
        "recommended_action",
        build_recommended_action(
            replay_event
        ),
    )
    replay_event.setdefault(
        "top_features",
        [],
    )
    replay_event.setdefault(
        "top_features_raw",
        [],
    )
    replay_event.setdefault(
        "feature_contributors",
        replay_event.get(
            "top_features",
            [],
        ),
    )

    return replay_event


def _set_replay_state(
    **changes: Any,
) -> None:
    with REPLAY_STATE_LOCK:
        REPLAY_STATE.update(changes)


def _replay_worker(
    records: List[Dict[str, Any]],
    source_path: Path,
) -> None:
    emitted = 0

    _set_replay_state(
        running=True,
        source_file=str(source_path),
        loaded_events=len(records),
        emitted_events=0,
        started_at=_replay_now_iso(),
        stopped_at=None,
        last_error=None,
    )

    try:
        while not REPLAY_STOP_EVENT.is_set():
            usable_in_pass = 0

            for record in records:
                if REPLAY_STOP_EVENT.is_set():
                    break

                replay_event = _normalise_replay_event(
                    record=record,
                    sequence=emitted + 1,
                    source_path=source_path,
                )

                if replay_event is None:
                    continue

                usable_in_pass += 1
                emitted += 1

                recent.append(
                    replay_event
                )

                if replay_event[
                    "raw_model_flag"
                ]:
                    alerts.append(
                        replay_event
                    )

                _append_done(
                    time.time()
                )
                update_gauges(
                    len(alerts)
                )

                _set_replay_state(
                    emitted_events=emitted,
                )

                if REPLAY_STOP_EVENT.wait(
                    REPLAY_INTERVAL_S
                ):
                    break

            if usable_in_pass == 0:
                raise ValueError(
                    "The selected replay file contains "
                    "no usable flow or alert events."
                )

            if (
                REPLAY_STOP_EVENT.is_set()
                or not REPLAY_LOOP
            ):
                break

    except Exception as exc:
        print(
            "[replay] Worker error:",
            repr(exc),
        )

        _set_replay_state(
            last_error=str(exc),
        )

    finally:
        _set_replay_state(
            running=False,
            stopped_at=_replay_now_iso(),
        )


def start_replay() -> Dict[str, Any]:
    global REPLAY_THREAD

    if IDS_MODE != "replay":
        raise HTTPException(
            status_code=409,
            detail=(
                "Replay is unavailable because "
                "IDS_MODE is not replay."
            ),
        )

    with REPLAY_STATE_LOCK:
        if (
            REPLAY_THREAD is not None
            and REPLAY_THREAD.is_alive()
        ):
            return {
                "status": "already_running",
                "mode": IDS_MODE,
                "replay": dict(REPLAY_STATE),
            }

    source_path = _find_replay_source()

    if source_path is None:
        message = (
            "No replay source was found. Add "
            "data/replay_events.jsonl or set "
            "IDS_REPLAY_FILE."
        )

        _set_replay_state(
            running=False,
            source_file=None,
            loaded_events=0,
            last_error=message,
        )

        raise HTTPException(
            status_code=404,
            detail=message,
        )

    records = _read_replay_records(
        source_path
    )

    if not records:
        message = (
            "The replay source contains no JSON "
            "event records."
        )

        _set_replay_state(
            running=False,
            source_file=str(source_path),
            loaded_events=0,
            last_error=message,
        )

        raise HTTPException(
            status_code=400,
            detail=message,
        )

    REPLAY_STOP_EVENT.clear()

    REPLAY_THREAD = threading.Thread(
        target=_replay_worker,
        args=(
            records,
            source_path,
        ),
        name="ids-replay-worker",
        daemon=True,
    )

    REPLAY_THREAD.start()

    return {
        "status": "started",
        "mode": IDS_MODE,
        "source_file": str(source_path),
        "loaded_events": len(records),
    }


def stop_replay() -> Dict[str, Any]:
    global REPLAY_THREAD

    REPLAY_STOP_EVENT.set()

    thread = REPLAY_THREAD

    if (
        thread is not None
        and thread.is_alive()
        and thread is not threading.current_thread()
    ):
        thread.join(timeout=2.0)

    _set_replay_state(
        running=False,
        stopped_at=_replay_now_iso(),
    )

    return {
        "status": "stopped",
        "mode": IDS_MODE,
        "replay": get_replay_state(),
    }


def get_replay_state() -> Dict[str, Any]:
    with REPLAY_STATE_LOCK:
        return dict(REPLAY_STATE)


def runtime_mode_payload() -> Dict[str, Any]:
    return {
        "mode": IDS_MODE,
        "mode_label": (
            "REPLAY MODE"
            if IDS_MODE == "replay"
            else "LIVE MODE"
        ),
        "is_live": IDS_MODE == "live",
        "is_replay": IDS_MODE == "replay",
        "replay_interval_seconds": (
            REPLAY_INTERVAL_S
        ),
        "replay_loop": REPLAY_LOOP,
        "replay": get_replay_state(),
    }

def _first_present(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in record:
            continue

        value = record.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return None


def _safe_metric_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _safe_metric_timestamp(value: Any) -> Optional[float]:
    numeric_value = _safe_metric_float(value)

    if numeric_value is not None:
        return numeric_value

    if not isinstance(value, str):
        return None

    text = value.strip()

    if not text:
        return None

    try:
        normalised = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalised).timestamp()
    except ValueError:
        return None


def _flatten_metrics_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flattened = dict(record)

    for nested_key in ("event", "flow", "data", "metrics"):
        nested_value = record.get(nested_key)

        if isinstance(nested_value, dict):
            flattened.update(nested_value)

    return flattened


def _normalise_final_label(value: Any) -> str:
    label = str(value or "").strip().upper()

    aliases = {
        "CRIT": "CRITICAL",
        "MED": "REVIEW",
        "WARN": "REVIEW",
        "WARNING": "REVIEW",
        "NORMAL": "OK",
    }

    label = aliases.get(label, label)

    if not label:
        return "UNKNOWN"

    return label


def _normalise_traffic_class(value: Any) -> str:
    label = str(value or "").strip().lower()

    if not label:
        return "unknown"

    return label.replace("_", " ")


def _normalise_protocol(value: Any) -> str:
    label = str(value or "").strip().upper()
    return label if label else "UNKNOWN"


def _normalise_destination_port(value: Any) -> str:
    if value is None:
        return "UNKNOWN"

    text = str(value).strip()

    if not text:
        return "UNKNOWN"

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text.upper()


def _merge_distribution(
    target: CollectionCounter,
    value: Any,
    normaliser,
) -> None:
    if isinstance(value, dict):
        for label, count in value.items():
            numeric_count = _safe_metric_float(count)

            if numeric_count is None:
                continue

            normalised_label = normaliser(label)

            if normalised_label:
                target[normalised_label] += int(numeric_count)

        return

    if not isinstance(value, list):
        return

    for item in value:
        if isinstance(item, dict):
            label = _first_present(
                item,
                "label",
                "name",
                "key",
                "severity",
                "final_label",
                "traffic_class",
                "protocol",
                "proto",
                "port",
                "dest_port",
                "destination_port",
            )

            count = _first_present(
                item,
                "count",
                "value",
                "total",
                "flows",
                "flow_count",
            )

            numeric_count = _safe_metric_float(count)

            if label is None or numeric_count is None:
                continue

            normalised_label = normaliser(label)

            if normalised_label:
                target[normalised_label] += int(numeric_count)

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label = item[0]
            numeric_count = _safe_metric_float(item[1])

            if numeric_count is None:
                continue

            normalised_label = normaliser(label)

            if normalised_label:
                target[normalised_label] += int(numeric_count)


def _find_metrics_file(filename: str) -> Optional[Path]:
    for directory in METRICS_SEARCH_DIRECTORIES:
        candidate = directory / filename

        if candidate.is_file():
            return candidate

    return None


def _read_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    buffered_lines: Deque[str] = deque(maxlen=METRICS_FILE_MAX_RECORDS)

    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line:
                    buffered_lines.append(line)
    except OSError:
        return []

    records: List[Dict[str, Any]] = []

    for line in buffered_lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            records.append(parsed)

    return records


def _metrics_source_label(path: Path) -> str:
    try:
        relative_path = path.relative_to(PROJECT_ROOT)
        return f"jsonl:{relative_path.as_posix()}"
    except ValueError:
        return f"jsonl:{path.name}"


def _collect_metrics_records() -> Tuple[List[Dict[str, Any]], str]:
    recent_items = list(recent)

    if recent_items:
        return recent_items, "memory:recent"

    alert_items = list(alerts)

    if alert_items:
        return alert_items, "memory:alerts"

    flow_metrics_path = _find_metrics_file("metrics_flow.jsonl")

    if flow_metrics_path is not None:
        flow_records = _read_metrics_jsonl(flow_metrics_path)

        if flow_records:
            return flow_records, _metrics_source_label(flow_metrics_path)

    summary_metrics_path = _find_metrics_file("metrics_summary.jsonl")

    if summary_metrics_path is not None:
        summary_records = _read_metrics_jsonl(summary_metrics_path)

        if summary_records:
            return summary_records, _metrics_source_label(summary_metrics_path)

    return [], "none"


def _counter_rows(
    counter: CollectionCounter,
    preferred_order: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Tuple[str, int]] = []

    if preferred_order:
        already_added = set()

        for label in preferred_order:
            count = int(counter.get(label, 0))

            if count > 0:
                rows.append((label, count))
                already_added.add(label)

        remaining = [
            (label, int(count))
            for label, count in counter.items()
            if label not in already_added and int(count) > 0
        ]

        remaining.sort(key=lambda item: (-item[1], item[0]))
        rows.extend(remaining)

    else:
        rows = [
            (str(label), int(count))
            for label, count in counter.items()
            if int(count) > 0
        ]

        rows.sort(key=lambda item: (-item[1], item[0]))

    if limit is not None:
        rows = rows[:limit]

    return [
        {
            "label": label,
            "count": count,
        }
        for label, count in rows
    ]


def _build_metrics_ui_summary(
    records: List[Dict[str, Any]],
    source: str,
) -> Dict[str, Any]:
    severity_counts: CollectionCounter = CollectionCounter()
    traffic_class_counts: CollectionCounter = CollectionCounter()
    protocol_counts: CollectionCounter = CollectionCounter()
    destination_port_counts: CollectionCounter = CollectionCounter()

    score_buckets: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {
            "sum": 0.0,
            "weight": 0.0,
        }
    )

    is_summary_file = "metrics_summary.jsonl" in source

    if is_summary_file and records:
        distribution_records = records[-1:]
    else:
        distribution_records = records

    explicit_total = 0

    for raw_record in distribution_records:
        if not isinstance(raw_record, dict):
            continue

        record = _flatten_metrics_record(raw_record)

        total_value = _safe_metric_float(
            _first_present(
                record,
                "total_flows",
                "flow_count",
                "total_events",
                "count",
            )
        )

        if total_value is not None:
            explicit_total = max(explicit_total, int(total_value))

        severity_distribution = _first_present(
            record,
            "severity_distribution",
            "severity_counts",
            "final_label_distribution",
            "label_counts",
        )

        if severity_distribution is not None:
            _merge_distribution(
                severity_counts,
                severity_distribution,
                _normalise_final_label,
            )
        else:
            severity = _first_present(
                record,
                "final_label",
                "display_label",
                "final_severity",
                "severity",
                "raw_severity",
            )

            if severity is not None:
                severity_counts[_normalise_final_label(severity)] += 1

        traffic_distribution = _first_present(
            record,
            "traffic_class_distribution",
            "traffic_class_counts",
        )

        if traffic_distribution is not None:
            _merge_distribution(
                traffic_class_counts,
                traffic_distribution,
                _normalise_traffic_class,
            )
        else:
            traffic_class = _first_present(
                record,
                "traffic_class",
                "class",
            )

            if traffic_class is not None:
                traffic_class_counts[
                    _normalise_traffic_class(traffic_class)
                ] += 1

        protocol_distribution = _first_present(
            record,
            "protocol_distribution",
            "protocol_counts",
            "top_protocols",
        )

        if protocol_distribution is not None:
            _merge_distribution(
                protocol_counts,
                protocol_distribution,
                _normalise_protocol,
            )
        else:
            protocol = _first_present(
                record,
                "proto",
                "protocol",
                "app_proto",
            )

            if protocol is not None:
                protocol_counts[_normalise_protocol(protocol)] += 1

        port_distribution = _first_present(
            record,
            "destination_port_distribution",
            "destination_port_counts",
            "dest_port_counts",
            "top_destination_ports",
        )

        if port_distribution is not None:
            _merge_distribution(
                destination_port_counts,
                port_distribution,
                _normalise_destination_port,
            )
        else:
            destination_port = _first_present(
                record,
                "dest_port",
                "destination_port",
                "dport",
            )

            if destination_port is not None:
                destination_port_counts[
                    _normalise_destination_port(destination_port)
                ] += 1

    def add_score_point(
        timestamp_value: Any,
        score_value: Any,
        weight_value: Any = 1,
    ) -> None:
        timestamp = _safe_metric_timestamp(timestamp_value)
        score = _safe_metric_float(score_value)
        weight = _safe_metric_float(weight_value)

        if timestamp is None or score is None:
            return

        if weight is None or weight <= 0:
            weight = 1.0

        bucket = (
            int(timestamp // METRICS_TIME_BUCKET_S)
            * METRICS_TIME_BUCKET_S
        )

        score_buckets[bucket]["sum"] += score * weight
        score_buckets[bucket]["weight"] += weight

    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue

        record = _flatten_metrics_record(raw_record)

        existing_series = _first_present(
            record,
            "average_anomaly_score_over_time",
            "anomaly_score_over_time",
        )

        if isinstance(existing_series, list):
            for point in existing_series:
                if not isinstance(point, dict):
                    continue

                add_score_point(
                    _first_present(
                        point,
                        "ts_unix",
                        "timestamp",
                        "time",
                        "bucket",
                    ),
                    _first_present(
                        point,
                        "average",
                        "avg",
                        "average_anomaly_score",
                        "anomaly_score",
                        "ae_score",
                        "score",
                    ),
                    _first_present(
                        point,
                        "count",
                        "flow_count",
                        "sample_count",
                    ) or 1,
                )

            continue

        add_score_point(
            _first_present(
                record,
                "ts_unix",
                "timestamp",
                "time",
                "created_at",
            ),
            _first_present(
                record,
                "ae_score",
                "anomaly_score",
                "average_anomaly_score",
                "avg_anomaly_score",
                "score",
            ),
            _first_present(
                record,
                "sample_count",
                "flow_count",
            ) or 1,
        )

    score_series = []

    for bucket_timestamp in sorted(score_buckets):
        bucket_data = score_buckets[bucket_timestamp]
        weight = bucket_data["weight"]

        if weight <= 0:
            continue

        score_series.append(
            {
                "ts_unix": bucket_timestamp,
                "average": bucket_data["sum"] / weight,
                "count": int(weight),
            }
        )

    severity_total = sum(int(value) for value in severity_counts.values())

    if is_summary_file and explicit_total > 0:
        total_flows = explicit_total
    elif severity_total > 0:
        total_flows = severity_total
    elif explicit_total > 0:
        total_flows = explicit_total
    else:
        total_flows = len(records)

    return {
        "generated_at": time.time(),
        "source": source,
        "total_flows": int(total_flows),
        "severity_distribution": _counter_rows(
            severity_counts,
            preferred_order=[
                "OK",
                "BENIGN",
                "REVIEW",
                "CRITICAL",
                "UNKNOWN",
            ],
        ),
        "traffic_class_distribution": _counter_rows(
            traffic_class_counts,
            limit=10,
        ),
        "average_anomaly_score_over_time": score_series,
        "top_protocols": _counter_rows(
            protocol_counts,
            limit=10,
        ),
        "top_destination_ports": _counter_rows(
            destination_port_counts,
            limit=10,
        ),
    }


class AlertStatusUpdate(BaseModel):
    status: str


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


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IDS Dashboard Login</title>
        <link rel="stylesheet" href="/static/dashboard.css">
        <style>
            .login-container {
                max-width: 420px;
                margin: 80px auto;
                padding: 24px;
                border: 1px solid #ddd;
                border-radius: 12px;
                background: white;
            }
            .login-container input {
                width: 100%;
                padding: 10px;
                margin: 8px 0 16px 0;
                box-sizing: border-box;
            }
            .login-container button {
                width: 100%;
                padding: 10px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <main class="login-container">
            <h1>IDS Dashboard Login</h1>
            <form method="post" action="/login">
                <label>Username</label>
                <input type="text" name="username" required>

                <label>Password</label>
                <input type="password" name="password" required>

                <button type="submit">Login</button>
            </form>
        </main>
    </body>
    </html>
    """


@app.post("/login")
async def login_submit(username: str = Form(...), password: str = Form(...)):
    valid_user = secrets.compare_digest(username, DASHBOARD_USER)
    valid_password = secrets.compare_digest(password, DASHBOARD_PASSWORD)

    if not (valid_user and valid_password):
        return HTMLResponse(
            """
            <h1>Login failed</h1>
            <p>Invalid username or password.</p>
            <a href="/login">Try again</a>
            """,
            status_code=401,
        )

    response = RedirectResponse(url="/ui/system", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_token(username),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 4,
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response

@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request},
    )


@app.get("/ui/system", response_class=HTMLResponse)
def ui_system(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_system.html",
        {"request": request},
    )

@app.get("/ui/alerts", response_class=HTMLResponse)
def ui_alerts(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_alerts.html",
        {"request": request},
    )

@app.get("/ui/alerts/{alert_id}", response_class=HTMLResponse)
def ui_alert_detail(
    request: Request,
    alert_id: str,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_alert_detail.html",
        {
            "request": request,
            "alert_id": alert_id,
        },
    )

@app.get("/ui/recent", response_class=HTMLResponse)
def ui_recent(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_recent.html",
        {"request": request},
    )

@app.get("/ui/metrics", response_class=HTMLResponse)
def ui_metrics(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_metrics.html",
        {
            "request": request,
        },
    )

@app.get("/ui/patterns", response_class=HTMLResponse)
def ui_patterns(
    request: Request,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    return templates.TemplateResponse(
        "ui_patterns.html",
        {
            "request": request,
        },
    )


@app.get("/ui/network", response_class=HTMLResponse)
def ui_network(
    request: Request,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    return templates.TemplateResponse(
        "ui_network.html",
        {
            "request": request,
        },
    )

@app.get("/ui/help", response_class=HTMLResponse)
def ui_help(
    request: Request,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    return templates.TemplateResponse(
        "ui_help.html",
        {
            "request": request,
        },
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": IDS_MODE,
        "mode_label": "REPLAY MODE" if IDS_MODE == "replay" else "LIVE MODE",
        "model": ONNX_PATH,
        "scaler": SCALER_PATH,
        "n_features": len(feature_cols),
        "bands": bands,
        "repeat_window_s": REPEAT_WINDOW_S,
        "repeat_critical_count": REPEAT_CRITICAL_COUNT,
        "notifications_enabled": NOTIFICATIONS_ENABLED,
        "notification_cooldown_s": NOTIFICATION_COOLDOWN_S,
        "runtime_config_source": RUNTIME_CONFIG_META["source"],
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
            "simple_explanation",
            "analyst_explanation",
            "technical_explanation",
            "legacy_explanation",
            "short_summary",
            "full_explanation",
            "adjustment_reason",
            "possible_explanation",
            "what_to_check",
            "recommended_action",
        ],
    }


@app.get("/stats")
def stats(
    authorized: bool = Depends(require_dashboard_login),
):
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


@app.get("/runtime-config")
def get_runtime_config(
    authorized: bool = Depends(require_dashboard_login),
):
    """Return the validated, active runtime configuration."""
    return JSONResponse(
        {
            "config": RUNTIME_CONFIG,
            "source": RUNTIME_CONFIG_META["source"],
            "path": RUNTIME_CONFIG_META["path"],
            "warnings": RUNTIME_CONFIG_META["warnings"],
        }
    )




@app.get("/runtime/mode")
def get_runtime_mode(
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    return JSONResponse(
        runtime_mode_payload()
    )


@app.post("/replay/start")
def replay_start(
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    return JSONResponse(
        start_replay()
    )


@app.post("/replay/stop")
def replay_stop(
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    return JSONResponse(
        stop_replay()
    )


@app.get("/metrics")
def metrics(
    authorized: bool = Depends(require_dashboard_login),
):
    update_gauges(len(alerts))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/ui-summary")
def metrics_ui_summary(
    authorized: bool = Depends(require_dashboard_login),
):
    records, source = _collect_metrics_records()
    return JSONResponse(
        _build_metrics_ui_summary(
            records=records,
            source=source,
        )
    )

def _get_alert_detail_id(
    alert: Dict[str, Any],
) -> Optional[str]:
    """
    Return the canonical identifier used by alert-detail URLs.

    The identifier is read-only metadata. It does not change the
    alert, model output, final decision, human status, feedback,
    anomaly score, or blocklist state.
    """
    if not isinstance(alert, dict):
        return None

    candidate_keys = (
        "detail_id",
        "flow_id",
        "alert_id",
        "event_id",
        "uid",
        "id",
        "replay_original_id",
        "ts_unix",
        "timestamp",
    )

    for key in candidate_keys:
        value = alert.get(key)

        if value is None:
            continue

        text_value = str(value).strip()

        if text_value:
            return text_value

    return None


def _attach_alert_detail_id(
    alert: Dict[str, Any],
) -> Dict[str, Any]:
    copied_alert = dict(alert)
    copied_alert["detail_id"] = _get_alert_detail_id(alert)
    return copied_alert


@app.get("/recent")
def get_recent(
    limit: int = 50,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    items = [
        _attach_alert_detail_id(item)
        for item in list(recent)[-limit:][::-1]
        if isinstance(item, dict)
    ]

    return {
        "bands": bands,
        "recent": items,
    }


@app.get("/alerts")
def get_alerts(
    limit: int = 50,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    update_gauges(len(alerts))

    status_store = (
        _read_alert_status_store()
    )

    items = []

    for item in list(alerts)[-limit:][::-1]:
        if not isinstance(item, dict):
            continue

        item_with_status = (
            _attach_human_status(
                item,
                status_store,
            )
        )

        items.append(
            _attach_alert_detail_id(
                item_with_status
            )
        )

    return {
        "bands": bands,
        "alerts": items,
    }

TEMPORAL_CONTEXT_WINDOW_S = 24 * 60 * 60


def _normalise_temporal_text(
    value: Any,
    uppercase: bool = False,
) -> str:
    text = str(value or "").strip()

    if uppercase:
        return text.upper()

    return text


def _normalise_temporal_port(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _temporal_timestamp(
    record: Dict[str, Any],
) -> Optional[float]:
    return _safe_metric_timestamp(
        _first_present(
            record,
            "ts_unix",
            "timestamp",
            "time",
            "created_at",
        )
    )


def _temporal_record_identity(
    record: Dict[str, Any],
) -> Tuple[Any, ...]:
    """
    Create a stable identity so the same event is not counted twice.

    This is important because anomalous events may exist in both
    the alerts and recent-flow buffers.
    """
    flow_id = _normalise_temporal_text(
        record.get("flow_id")
    )

    if flow_id:
        return (
            "flow_id",
            flow_id,
        )

    timestamp = _temporal_timestamp(record)

    rounded_timestamp = (
        round(timestamp, 6)
        if timestamp is not None
        else None
    )

    return (
        "flow",
        _normalise_temporal_text(
            record.get("src_ip")
        ),
        _normalise_temporal_text(
            record.get("dest_ip")
        ),
        _normalise_temporal_port(
            record.get("dest_port")
        ),
        _normalise_temporal_text(
            record.get("proto"),
            uppercase=True,
        ),
        rounded_timestamp,
    )


def _available_repeat_count(
    alert: Dict[str, Any],
) -> Optional[int]:
    """
    Retrieve the repeat count from the available alert format.

    Newer events normally contain repeat_count directly.
    Older events may contain it inside repeat_info or debug.
    """
    repeat_info = alert.get("repeat_info")

    if not isinstance(repeat_info, dict):
        repeat_info = {}

    debug = alert.get("debug")

    if not isinstance(debug, dict):
        debug = {}

    debug_repeat_info = debug.get("repeat_info")

    if not isinstance(debug_repeat_info, dict):
        debug_repeat_info = {}

    candidates = (
        alert.get("repeat_count"),
        repeat_info.get("current_count"),
        debug_repeat_info.get("current_count"),
    )

    for value in candidates:
        if value is None or value == "":
            continue

        try:
            return max(
                0,
                int(float(value)),
            )
        except (TypeError, ValueError):
            continue

    return None


def compute_temporal_context(
    alert: Dict[str, Any],
    history: Any,
) -> Dict[str, Any]:
    """
    Calculate read-only historical information for an alert.

    The function does not modify:
    - anomaly scoring;
    - repeat memory;
    - contextual filtering;
    - final decision logic.

    history may be:
    - a list or deque of event dictionaries; or
    - a dictionary containing records and scope.
    """
    if not isinstance(alert, dict):
        alert = {}

    history_scope = "current buffer"
    history_records: Any = history

    if isinstance(history, dict):
        history_records = history.get(
            "records",
            [],
        )

        history_scope = str(
            history.get("scope")
            or "current buffer"
        ).strip()

    if not isinstance(
        history_records,
        (list, tuple, deque),
    ):
        history_records = []

    unique_records: List[Dict[str, Any]] = []
    seen_identities = set()

    # Also include the selected alert if it is not already
    # present in the supplied history.
    for record in list(history_records) + [alert]:
        if not isinstance(record, dict):
            continue

        identity = _temporal_record_identity(
            record
        )

        if identity in seen_identities:
            continue

        seen_identities.add(identity)
        unique_records.append(record)

    timestamped_records: List[
        Tuple[Dict[str, Any], float]
    ] = []

    for record in unique_records:
        timestamp = _temporal_timestamp(
            record
        )

        if timestamp is not None:
            timestamped_records.append(
                (
                    record,
                    timestamp,
                )
            )

    # When timestamps are available, use only records
    # from the latest 24-hour period.
    if timestamped_records:
        latest_history_timestamp = max(
            timestamp
            for _, timestamp in timestamped_records
        )

        cutoff = (
            latest_history_timestamp
            - TEMPORAL_CONTEXT_WINDOW_S
        )

        records_in_scope = [
            record
            for record, timestamp
            in timestamped_records
            if cutoff
            <= timestamp
            <= latest_history_timestamp
        ]

        time_window = "last 24 hours"

    else:
        records_in_scope = unique_records
        time_window = "available records"

    target_source = _normalise_temporal_text(
        alert.get("src_ip")
    )

    target_destination = _normalise_temporal_text(
        alert.get("dest_ip")
    )

    target_port = _normalise_temporal_port(
        alert.get("dest_port")
    )

    target_protocol = _normalise_temporal_text(
        alert.get("proto"),
        uppercase=True,
    )

    same_source_records: List[
        Dict[str, Any]
    ] = []

    same_flow_records: List[
        Dict[str, Any]
    ] = []

    exact_key_available = bool(
        target_source
        and target_destination
        and target_port is not None
        and target_protocol
    )

    for record in records_in_scope:
        record_source = (
            _normalise_temporal_text(
                record.get("src_ip")
            )
        )

        if (
            target_source
            and record_source == target_source
        ):
            same_source_records.append(
                record
            )

        if not exact_key_available:
            continue

        record_destination = (
            _normalise_temporal_text(
                record.get("dest_ip")
            )
        )

        record_port = (
            _normalise_temporal_port(
                record.get("dest_port")
            )
        )

        record_protocol = (
            _normalise_temporal_text(
                record.get("proto"),
                uppercase=True,
            )
        )

        if (
            record_source == target_source
            and record_destination
            == target_destination
            and record_port == target_port
            and record_protocol
            == target_protocol
        ):
            same_flow_records.append(
                record
            )

    # First/last seen should preferably describe the exact
    # flow pattern. When exact matching is impossible, use
    # observations from the same source IP.
    if same_flow_records:
        seen_records = same_flow_records

        seen_basis = (
            "same source, destination, "
            "destination port, and protocol"
        )

    elif same_source_records:
        seen_records = same_source_records
        seen_basis = "same source IP"

    else:
        seen_records = []
        seen_basis = "no matching records"

    seen_timestamps = [
        timestamp
        for timestamp in (
            _temporal_timestamp(record)
            for record in seen_records
        )
        if timestamp is not None
    ]

    return {
        "same_source_ip_count": len(
            same_source_records
        ),
        "same_flow_count": len(
            same_flow_records
        ),
        "first_seen": (
            min(seen_timestamps)
            if seen_timestamps
            else None
        ),
        "last_seen": (
            max(seen_timestamps)
            if seen_timestamps
            else None
        ),
        "repeat_count": (
            _available_repeat_count(alert)
        ),
        "history_scope": (
            history_scope
            or "current buffer"
        ),
        "time_window": time_window,
        "seen_basis": seen_basis,
        "counts_include_selected_alert": True,
    }


def _collect_temporal_history() -> Dict[str, Any]:
    """
    Prefer the existing JSONL prediction history.

    When that file is unavailable or empty, use only the
    currently available in-memory buffer and label it as
    current buffer.
    """
    history_path = Path(API_LOG_PATH)

    if history_path.is_file():
        log_records = _read_metrics_jsonl(
            history_path
        )

        if log_records:
            return {
                "records": log_records,
                "scope": "historical log",
            }

    buffer_records = list(recent)

    if not buffer_records:
        buffer_records = list(alerts)

    return {
        "records": buffer_records,
        "scope": "current buffer",
    }

def _safe_detail_value(value: Any) -> str:
    if value is None:
        return "-"
    value = str(value).strip()
    return value if value else "-"

# =========================================================
# Repeated Pattern View
# =========================================================
# This section only groups already stored events.
# It does not change anomaly scoring, contextual filtering,
# repeat memory, thresholds, or final decisions.

PATTERN_MIN_EVENTS = 2
PATTERN_HISTORY_LIMIT = 5000

RAW_PATTERN_RANK = {
    "UNKNOWN": 0,
    "OK": 1,
    "WARN": 2,
    "MED": 3,
    "CRIT": 4,
}

FINAL_PATTERN_RANK = {
    "UNKNOWN": 0,
    "OK": 1,
    "BENIGN": 2,
    "REVIEW": 3,
    "CRITICAL": 4,
}


def _pattern_text(
    value: Any,
    *,
    default: str = "unknown",
    upper: bool = False,
    lower: bool = False,
) -> str:
    text = str(
        value or ""
    ).strip()

    if not text:
        text = default

    if upper:
        return text.upper()

    if lower:
        return text.lower()

    return text


def _pattern_port(
    value: Any,
) -> str:
    if value in (
        None,
        "",
    ):
        return "unknown"

    try:
        return str(
            int(
                float(value)
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return (
            str(value)
            .strip()
            .lower()
            or "unknown"
        )


def _pattern_timestamp(
    record: Dict[str, Any],
) -> Optional[float]:
    return _safe_metric_timestamp(
        _first_present(
            record,
            "ts_unix",
            "timestamp",
            "time",
            "created_at",
        )
    )


def _pattern_raw_severity(
    record: Dict[str, Any],
) -> str:
    label = _pattern_text(
        _first_present(
            record,
            "raw_severity",
            "raw_model_severity",
        ),
        default="UNKNOWN",
        upper=True,
    )

    aliases = {
        "WARNING": "WARN",
        "MEDIUM": "MED",
        "CRITICAL": "CRIT",
        "NORMAL": "OK",
    }

    label = aliases.get(
        label,
        label,
    )

    if label not in RAW_PATTERN_RANK:
        return "UNKNOWN"

    return label


def _pattern_final_decision(
    record: Dict[str, Any],
) -> str:
    label = _normalise_final_label(
        _first_present(
            record,
            "final_label",
            "display_label",
            "final_decision",
            "final_severity",
            "severity",
        )
    )

    if label not in FINAL_PATTERN_RANK:
        return "UNKNOWN"

    return label


def _pattern_is_candidate(
    record: Dict[str, Any],
) -> bool:
    """
    Keep the historical view focused on anomaly-related records.

    The live alert buffer already contains raw anomalies. The
    extra checks prevent normal OK records from the complete
    JSONL prediction history from becoming anomaly patterns.
    """
    if bool(
        record.get(
            "raw_model_flag"
        )
    ):
        return True

    debug = record.get(
        "debug"
    )

    if (
        isinstance(
            debug,
            dict,
        )
        and bool(
            debug.get(
                "is_anom"
            )
        )
    ):
        return True

    if _pattern_raw_severity(
        record
    ) in {
        "WARN",
        "MED",
        "CRIT",
    }:
        return True

    repeat_count = (
        _available_repeat_count(
            record
        )
        or 0
    )

    return (
        repeat_count
        >= PATTERN_MIN_EVENTS
    )


def _pattern_identity(
    record: Dict[str, Any],
) -> Tuple[Any, ...]:
    """
    Build an identity used to prevent the same event from
    being counted once from memory and again from JSONL.
    """
    flow_id = str(
        record.get(
            "flow_id"
        )
        or ""
    ).strip()

    if flow_id:
        return (
            "flow_id",
            flow_id,
        )

    return (
        "event",
        _pattern_text(
            _first_present(
                record,
                "src_ip",
                "source_ip",
            )
        ),
        _pattern_text(
            _first_present(
                record,
                "dest_ip",
                "destination_ip",
            )
        ),
        _pattern_port(
            _first_present(
                record,
                "dest_port",
                "destination_port",
            )
        ),
        _pattern_text(
            _first_present(
                record,
                "proto",
                "protocol",
            ),
            upper=True,
        ),
        _pattern_text(
            record.get(
                "traffic_class"
            ),
            lower=True,
        ),
        _pattern_timestamp(
            record
        ),
    )


def _collect_pattern_records(
) -> Tuple[
    List[Dict[str, Any]],
    str,
    bool,
]:
    """
    Start with the current alert buffer.

    The current recent buffer is used as a secondary memory
    source. JSONL prediction history is then included when it
    exists. Duplicate records are removed.
    """
    records: List[
        Dict[str, Any]
    ] = []

    seen = set()

    def append_records(
        source_records: Any,
    ) -> int:
        added = 0

        if not isinstance(
            source_records,
            (
                list,
                tuple,
                deque,
            ),
        ):
            return added

        for record in source_records:
            if not isinstance(
                record,
                dict,
            ):
                continue

            if not _pattern_is_candidate(
                record
            ):
                continue

            identity = (
                _pattern_identity(
                    record
                )
            )

            if identity in seen:
                continue

            seen.add(
                identity
            )

            records.append(
                record
            )

            added += 1

        return added

    # Primary source required by the feature.
    alert_count = append_records(
        list(alerts)
    )

    # Secondary current-memory source.
    recent_count = append_records(
        list(recent)
    )

    history_path = Path(
        API_LOG_PATH
    )

    if (
        not history_path.is_absolute()
        and not history_path.is_file()
    ):
        project_history_path = (
            PROJECT_ROOT
            / history_path
        )

        if project_history_path.is_file():
            history_path = (
                project_history_path
            )

    history_count = 0

    if history_path.is_file():
        history_records = (
            _read_metrics_jsonl(
                history_path
            )
        )

        history_records = (
            history_records[
                -PATTERN_HISTORY_LIMIT:
            ]
        )

        history_count = (
            append_records(
                history_records
            )
        )

    history_included = (
        history_count > 0
    )

    if (
        alert_count
        and history_included
    ):
        source = (
            "current alert buffer "
            "+ JSONL history"
        )

    elif alert_count:
        source = (
            "current alert buffer"
        )

    elif (
        recent_count
        and history_included
    ):
        source = (
            "current recent buffer "
            "+ JSONL history"
        )

    elif recent_count:
        source = (
            "current recent buffer"
        )

    elif history_included:
        source = (
            "JSONL history"
        )

    else:
        source = "none"

    return (
        records,
        source,
        history_included,
    )


def _pattern_reason(
    record: Dict[str, Any],
) -> str:
    debug = record.get(
        "debug"
    )

    if not isinstance(
        debug,
        dict,
    ):
        debug = {}

    reason = _first_present(
        record,
        "short_summary",
        "summary",
        "display_label_reason",
        "adjustment_reason",
        "repeat_explanation",
        "explanation",
        "benign_reason",
        "traffic_note",
    )

    if reason is None:
        reason = debug.get(
            "repeat_explanation"
        )

    text = str(
        reason
        or (
            "Repeated matching "
            "anomalous traffic."
        )
    ).strip()

    return (
        text
        or (
            "Repeated matching "
            "anomalous traffic."
        )
    )


def _build_repeated_patterns(
    records: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        Tuple[
            str,
            str,
            str,
            str,
            str,
        ],
        Dict[str, Any],
    ] = {}

    for record in records:
        source_ip = _pattern_text(
            _first_present(
                record,
                "src_ip",
                "source_ip",
            )
        )

        destination_ip = (
            _pattern_text(
                _first_present(
                    record,
                    "dest_ip",
                    "destination_ip",
                )
            )
        )

        destination_port = (
            _pattern_port(
                _first_present(
                    record,
                    "dest_port",
                    "destination_port",
                )
            )
        )

        protocol = _pattern_text(
            _first_present(
                record,
                "proto",
                "protocol",
            ),
            upper=True,
        )

        traffic_class = (
            _pattern_text(
                record.get(
                    "traffic_class"
                ),
                lower=True,
            )
        )

        key = (
            source_ip,
            destination_ip,
            destination_port,
            protocol,
            traffic_class,
        )

        raw_severity = (
            _pattern_raw_severity(
                record
            )
        )

        final_decision = (
            _pattern_final_decision(
                record
            )
        )

        timestamp = (
            _pattern_timestamp(
                record
            )
        )

        if key not in grouped:
            readable_key = (
                " | ".join(key)
            )

            pattern_hash = (
                hashlib.sha256(
                    readable_key.encode(
                        "utf-8"
                    )
                )
                .hexdigest()[:12]
                .upper()
            )

            grouped[key] = {
                "pattern_id": (
                    f"PAT-{pattern_hash}"
                ),
                "pattern_key": (
                    readable_key
                ),
                "source_ip": (
                    source_ip
                ),
                "destination_ip": (
                    destination_ip
                ),
                "destination_port": (
                    destination_port
                ),
                "protocol": (
                    protocol
                ),
                "traffic_class": (
                    traffic_class
                ),
                "event_count": 0,
                "highest_raw_severity": (
                    "UNKNOWN"
                ),
                "highest_final_decision": (
                    "UNKNOWN"
                ),
                "first_seen": None,
                "last_seen": None,
                "short_reason": (
                    "Repeated matching "
                    "anomalous traffic."
                ),

                # Internal sorting values.
                "_raw_rank": 0,
                "_final_rank": 0,
                "_reason_rank": (
                    -1,
                    -1,
                    -1.0,
                ),
            }

        pattern = grouped[key]

        pattern[
            "event_count"
        ] += 1

        raw_rank = (
            RAW_PATTERN_RANK[
                raw_severity
            ]
        )

        final_rank = (
            FINAL_PATTERN_RANK[
                final_decision
            ]
        )

        if (
            raw_rank
            > pattern["_raw_rank"]
        ):
            pattern[
                "_raw_rank"
            ] = raw_rank

            pattern[
                "highest_raw_severity"
            ] = raw_severity

        if (
            final_rank
            > pattern["_final_rank"]
        ):
            pattern[
                "_final_rank"
            ] = final_rank

            pattern[
                "highest_final_decision"
            ] = final_decision

        if timestamp is not None:
            if (
                pattern["first_seen"]
                is None
                or timestamp
                < pattern["first_seen"]
            ):
                pattern[
                    "first_seen"
                ] = timestamp

            if (
                pattern["last_seen"]
                is None
                or timestamp
                > pattern["last_seen"]
            ):
                pattern[
                    "last_seen"
                ] = timestamp

        reason_rank = (
            final_rank,
            raw_rank,
            (
                timestamp
                if timestamp is not None
                else -1.0
            ),
        )

        if (
            reason_rank
            > pattern["_reason_rank"]
        ):
            pattern[
                "_reason_rank"
            ] = reason_rank

            pattern[
                "short_reason"
            ] = _pattern_reason(
                record
            )

    patterns = [
        pattern
        for pattern
        in grouped.values()
        if (
            pattern["event_count"]
            >= PATTERN_MIN_EVENTS
        )
    ]

    patterns.sort(
        key=lambda pattern: (
            -pattern[
                "event_count"
            ],
            -pattern[
                "_final_rank"
            ],
            -pattern[
                "_raw_rank"
            ],
            -(
                pattern[
                    "last_seen"
                ]
                or 0.0
            ),
        )
    )

    for pattern in patterns:
        pattern.pop(
            "_raw_rank",
            None,
        )

        pattern.pop(
            "_final_rank",
            None,
        )

        pattern.pop(
            "_reason_rank",
            None,
        )

    return patterns


@app.get("/patterns")
def get_patterns(
    limit: int = 200,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    safe_limit = max(
        1,
        min(
            int(limit),
            500,
        ),
    )

    (
        records,
        source,
        history_included,
    ) = _collect_pattern_records()

    patterns = (
        _build_repeated_patterns(
            records
        )
    )

    return JSONResponse(
        {
            "generated_at": (
                time.time()
            ),
            "source": source,
            "history_included": (
                history_included
            ),
            "records_considered": (
                len(records)
            ),
            "minimum_event_count": (
                PATTERN_MIN_EVENTS
            ),
            "pattern_count": (
                len(patterns)
            ),
            "patterns": (
                patterns[
                    :safe_limit
                ]
            ),
        }
    ) 




# =========================================================
# Observed Network Communication Map
# =========================================================
# This section aggregates already stored flow records only.
# It does not perform physical discovery and does not change
# inference, anomaly scoring, contextual filtering, repeat logic,
# alert status, feedback, replay behavior, or reports.

NETWORK_MAP_DEFAULT_MAX_NODES = _safe_positive_int(
    os.getenv("NETWORK_MAP_MAX_NODES"),
    120,
    minimum=2,
)
NETWORK_MAP_DEFAULT_MAX_EDGES = _safe_positive_int(
    os.getenv("NETWORK_MAP_MAX_EDGES"),
    200,
    minimum=1,
)
NETWORK_MAP_HARD_MAX_NODES = 500
NETWORK_MAP_HARD_MAX_EDGES = 1000

NETWORK_MAP_DECISION_RANK = FINAL_PATTERN_RANK
NETWORK_MAP_ALLOWED_DECISIONS = {
    "ALL",
    "OK",
    "BENIGN",
    "REVIEW",
    "CRITICAL",
    "UNKNOWN",
}
NETWORK_MAP_ALLOWED_ADDRESS_TYPES = {
    "ALL",
    "LOOPBACK",
    "LOCAL",
    "EXTERNAL",
    "UNKNOWN",
}


def _network_text(
    value: Any,
    *,
    default: str = "",
    upper: bool = False,
    lower: bool = False,
) -> str:
    text = str(value or "").strip()

    if not text:
        text = default

    if upper:
        return text.upper()

    if lower:
        return text.lower()

    return text


def _network_port(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None

    try:
        port = int(float(value))
    except (TypeError, ValueError):
        return None

    if 0 <= port <= 65535:
        return port

    return None


def _network_timestamp(record: Dict[str, Any]) -> Optional[float]:
    return _safe_metric_timestamp(
        _first_present(
            record,
            "ts_unix",
            "timestamp",
            "time",
            "created_at",
        )
    )


def _network_iso_timestamp(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None

    try:
        return datetime.fromtimestamp(
            float(value),
            tz=timezone.utc,
        ).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _network_generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _network_final_decision(record: Dict[str, Any]) -> str:
    """
    Read the already calculated final decision without modifying it.

    final_label/display_label are preferred. Older records that contain
    only final_severity are translated using the project's existing
    OK/WARN/MED/CRIT -> OK/BENIGN/REVIEW/CRITICAL display meaning.
    """
    explicit = _network_text(
        _first_present(
            record,
            "final_label",
            "display_label",
            "final_decision",
        ),
        upper=True,
    )

    explicit_aliases = {
        "CRIT": "CRITICAL",
        "MED": "REVIEW",
        "WARN": "BENIGN",
        "WARNING": "BENIGN",
        "NORMAL": "OK",
    }
    explicit = explicit_aliases.get(explicit, explicit)

    if explicit in NETWORK_MAP_DECISION_RANK:
        return explicit

    technical = _network_text(
        _first_present(
            record,
            "final_severity",
            "severity",
            "raw_severity",
        ),
        upper=True,
    )

    technical_to_decision = {
        "OK": "OK",
        "NORMAL": "OK",
        "WARN": "BENIGN",
        "WARNING": "BENIGN",
        "MED": "REVIEW",
        "MEDIUM": "REVIEW",
        "REVIEW": "REVIEW",
        "CRIT": "CRITICAL",
        "CRITICAL": "CRITICAL",
        "BENIGN": "BENIGN",
    }

    return technical_to_decision.get(technical, "UNKNOWN")


def _network_is_demo(record: Dict[str, Any]) -> bool:
    """Recognize both explicit demo flags and existing replay metadata."""
    if bool(record.get("is_demo")):
        return True

    if bool(record.get("replay_mode")):
        return True

    if bool(record.get("demo_mode")):
        return True

    mode = _network_text(
        _first_present(
            record,
            "mode",
            "source_mode",
            "runtime_mode",
        ),
        lower=True,
    )

    return mode in {
        "demo",
        "replay",
        "replay_mode",
        "demo_mode",
    }


def _network_address_type(address: str) -> str:
    """Classify an observed IP locally, without any geolocation lookup."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return "unknown"

    if parsed.is_loopback:
        return "loopback"

    if (
        parsed.is_private
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_unspecified
    ):
        return "local"

    if parsed.is_global:
        return "external"

    return "unknown"


def _network_score(record: Dict[str, Any]) -> Optional[float]:
    return _safe_metric_float(
        _first_present(
            record,
            "ae_score",
            "anomaly_score",
            "score",
            "reconstruction_error",
        )
    )


def _network_edge_id(source: str, target: str) -> str:
    material = f"{source}\0{target}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:16]
    return f"edge-{digest}"


def _collect_network_map_records() -> Tuple[List[Dict[str, Any]], str]:
    """
    Prefer the Recent Traffic buffer because it contains all processed
    flows, including OK/BENIGN records. The Alerts buffer is a subset and
    is used only as a fallback when Recent Traffic is empty.
    """
    recent_records = [
        record
        for record in list(recent)
        if isinstance(record, dict)
    ]

    if recent_records:
        return recent_records, "current recent-traffic buffer"

    alert_records = [
        record
        for record in list(alerts)
        if isinstance(record, dict)
    ]

    if alert_records:
        return alert_records, "current alert buffer (Recent Traffic was empty)"

    return [], "none"


def _aggregate_network_edges(
    records: List[Dict[str, Any]],
    *,
    include_demo: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    counters = {
        "records_considered": len(records),
        "records_used": 0,
        "records_without_endpoints": 0,
        "demo_records_excluded": 0,
    }

    for record_index, record in enumerate(records):
        source = _network_text(
            _first_present(
                record,
                "src_ip",
                "source_ip",
                "src",
            )
        )
        target = _network_text(
            _first_present(
                record,
                "dest_ip",
                "destination_ip",
                "dst_ip",
                "destination",
            )
        )

        if not source or not target:
            counters["records_without_endpoints"] += 1
            continue

        is_demo = _network_is_demo(record)

        if is_demo and not include_demo:
            counters["demo_records_excluded"] += 1
            continue

        counters["records_used"] += 1

        key = (source, target)
        decision = _network_final_decision(record)
        decision_rank = NETWORK_MAP_DECISION_RANK.get(decision, 0)
        timestamp = _network_timestamp(record)
        protocol = _network_text(
            _first_present(
                record,
                "proto",
                "protocol",
                "app_proto",
            ),
            default="UNKNOWN",
            upper=True,
        )
        destination_port = _network_port(
            _first_present(
                record,
                "dest_port",
                "destination_port",
                "dport",
            )
        )
        traffic_class = _network_text(
            _first_present(
                record,
                "traffic_class",
                "class",
            ),
            default="unknown",
            lower=True,
        )
        score = _network_score(record)

        if key not in grouped:
            grouped[key] = {
                "id": _network_edge_id(source, target),
                "source": source,
                "target": target,
                "source_address_type": _network_address_type(source),
                "target_address_type": _network_address_type(target),
                "count": 0,
                "protocols": set(),
                "ports": set(),
                "traffic_classes": set(),
                "highest_decision": "UNKNOWN",
                "latest_decision": "UNKNOWN",
                "average_anomaly_score": None,
                "maximum_anomaly_score": None,
                "first_seen": None,
                "last_seen": None,
                "is_demo": False,
                "contains_demo": False,
                "contains_live": False,
                "demo_record_count": 0,
                "live_record_count": 0,
                "_highest_rank": 0,
                "_latest_sort_key": (-1.0, -1),
                "_score_sum": 0.0,
                "_score_count": 0,
            }

        edge = grouped[key]
        edge["count"] += 1
        edge["protocols"].add(protocol)
        edge["traffic_classes"].add(traffic_class)

        if destination_port is not None:
            edge["ports"].add(destination_port)

        if decision_rank > edge["_highest_rank"]:
            edge["_highest_rank"] = decision_rank
            edge["highest_decision"] = decision

        latest_key = (
            timestamp if timestamp is not None else -1.0,
            record_index,
        )

        if latest_key > edge["_latest_sort_key"]:
            edge["_latest_sort_key"] = latest_key
            edge["latest_decision"] = decision

        if timestamp is not None:
            if edge["first_seen"] is None or timestamp < edge["first_seen"]:
                edge["first_seen"] = timestamp

            if edge["last_seen"] is None or timestamp > edge["last_seen"]:
                edge["last_seen"] = timestamp

        if score is not None:
            edge["_score_sum"] += score
            edge["_score_count"] += 1

            if (
                edge["maximum_anomaly_score"] is None
                or score > edge["maximum_anomaly_score"]
            ):
                edge["maximum_anomaly_score"] = score

        if is_demo:
            edge["contains_demo"] = True
            edge["demo_record_count"] += 1
        else:
            edge["contains_live"] = True
            edge["live_record_count"] += 1

    edges: List[Dict[str, Any]] = []

    for edge in grouped.values():
        if edge["_score_count"] > 0:
            edge["average_anomaly_score"] = (
                edge["_score_sum"] / edge["_score_count"]
            )

        edge["is_demo"] = bool(
            edge["contains_demo"]
            and not edge["contains_live"]
        )
        edge["protocols"] = sorted(edge["protocols"])
        edge["ports"] = sorted(edge["ports"])
        edge["traffic_classes"] = sorted(edge["traffic_classes"])
        edge["first_seen"] = _network_iso_timestamp(edge["first_seen"])
        edge["last_seen"] = _network_iso_timestamp(edge["last_seen"])

        edge.pop("_score_sum", None)
        edge.pop("_score_count", None)
        edge.pop("_latest_sort_key", None)

        edges.append(edge)

    return edges, counters


def _network_address_filter_matches(
    edge: Dict[str, Any],
    address_type: str,
) -> bool:
    if address_type == "ALL":
        return True

    expected = address_type.lower()

    return bool(
        edge.get("source_address_type") == expected
        or edge.get("target_address_type") == expected
    )


def _filter_network_edges(
    edges: List[Dict[str, Any]],
    *,
    search: str,
    decision: str,
    address_type: str,
    min_edge_count: int,
) -> List[Dict[str, Any]]:
    search_text = search.strip().lower()
    filtered: List[Dict[str, Any]] = []

    for edge in edges:
        if edge["count"] < min_edge_count:
            continue

        if (
            decision != "ALL"
            and edge["highest_decision"] != decision
        ):
            continue

        if not _network_address_filter_matches(edge, address_type):
            continue

        if search_text and not (
            search_text in edge["source"].lower()
            or search_text in edge["target"].lower()
        ):
            continue

        filtered.append(edge)

    filtered.sort(
        key=lambda edge: (
            edge.get("_highest_rank", 0),
            edge.get("count", 0),
            _safe_metric_timestamp(edge.get("last_seen")) or 0.0,
        ),
        reverse=True,
    )

    return filtered


def _limit_network_edges(
    edges: List[Dict[str, Any]],
    *,
    max_nodes: int,
    max_edges: int,
) -> Tuple[List[Dict[str, Any]], set[str]]:
    selected_edges: List[Dict[str, Any]] = []
    selected_nodes: set[str] = set()

    for edge in edges:
        if len(selected_edges) >= max_edges:
            break

        required_nodes = {
            edge["source"],
            edge["target"],
        } - selected_nodes

        if len(selected_nodes) + len(required_nodes) > max_nodes:
            continue

        selected_edges.append(edge)
        selected_nodes.update(required_nodes)

    return selected_edges, selected_nodes


def _build_network_nodes(
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}

    def ensure_node(address: str) -> Dict[str, Any]:
        if address not in nodes:
            nodes[address] = {
                "id": address,
                "label": address,
                "address_type": _network_address_type(address),
                "flow_count": 0,
                "highest_decision": "UNKNOWN",
                "incoming_count": 0,
                "outgoing_count": 0,
                "incoming_relationships": 0,
                "outgoing_relationships": 0,
                "is_demo": False,
                "contains_demo": False,
                "contains_live": False,
                "demo_flow_count": 0,
                "live_flow_count": 0,
                "_highest_rank": 0,
            }

        return nodes[address]

    for edge in edges:
        source_node = ensure_node(edge["source"])
        target_node = ensure_node(edge["target"])
        count = int(edge.get("count", 0))
        rank = int(edge.get("_highest_rank", 0))
        decision = edge.get("highest_decision", "UNKNOWN")

        source_node["outgoing_count"] += count
        source_node["outgoing_relationships"] += 1
        target_node["incoming_count"] += count
        target_node["incoming_relationships"] += 1

        if edge["source"] == edge["target"]:
            source_node["flow_count"] += count
        else:
            source_node["flow_count"] += count
            target_node["flow_count"] += count

        affected_nodes = (
            (source_node,)
            if edge["source"] == edge["target"]
            else (source_node, target_node)
        )

        for node in affected_nodes:
            if rank > node["_highest_rank"]:
                node["_highest_rank"] = rank
                node["highest_decision"] = decision

            if edge.get("contains_demo"):
                node["contains_demo"] = True
                node["demo_flow_count"] += int(
                    edge.get("demo_record_count", 0)
                )

            if edge.get("contains_live"):
                node["contains_live"] = True
                node["live_flow_count"] += int(
                    edge.get("live_record_count", 0)
                )

    node_list = list(nodes.values())

    for node in node_list:
        node["is_demo"] = bool(
            node["contains_demo"]
            and not node["contains_live"]
        )
        node.pop("_highest_rank", None)

    node_list.sort(
        key=lambda node: (
            NETWORK_MAP_DECISION_RANK.get(
                node["highest_decision"],
                0,
            ),
            node["flow_count"],
            node["id"],
        ),
        reverse=True,
    )

    return node_list


def _clean_network_edge_for_response(
    edge: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = dict(edge)
    cleaned.pop("_highest_rank", None)
    return cleaned


def _build_network_map_payload(
    records: List[Dict[str, Any]],
    *,
    source: str,
    search: str,
    decision: str,
    address_type: str,
    min_edge_count: int,
    include_demo: bool,
    max_nodes: int,
    max_edges: int,
) -> Dict[str, Any]:
    aggregated_edges, counters = _aggregate_network_edges(
        records,
        include_demo=include_demo,
    )

    filtered_edges = _filter_network_edges(
        aggregated_edges,
        search=search,
        decision=decision,
        address_type=address_type,
        min_edge_count=min_edge_count,
    )

    all_filtered_nodes = {
        address
        for edge in filtered_edges
        for address in (
            edge["source"],
            edge["target"],
        )
    }

    selected_edges, selected_node_ids = _limit_network_edges(
        filtered_edges,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )

    nodes = _build_network_nodes(selected_edges)
    cleaned_edges = [
        _clean_network_edge_for_response(edge)
        for edge in selected_edges
    ]

    truncated = bool(
        len(selected_edges) < len(filtered_edges)
        or len(selected_node_ids) < len(all_filtered_nodes)
    )

    truncation_notice = ""

    if truncated:
        truncation_notice = (
            "The map was limited to the most relevant relationships. "
            "REVIEW and CRITICAL decisions were prioritized first, "
            "followed by higher record counts and more recent activity."
        )

    return {
        "scope": "current_buffer",
        "scope_label": "Current buffer",
        "source": source,
        "source_explanation": (
            "The Recent Traffic buffer is preferred because it contains "
            "all processed flows. The Alerts buffer is used only when "
            "Recent Traffic is empty because Alerts contains only an "
            "anomaly-focused subset."
        ),
        "generated_at": _network_generated_at(),
        "records_considered": counters["records_considered"],
        "records_used": counters["records_used"],
        "records_without_endpoints": counters[
            "records_without_endpoints"
        ],
        "demo_records_excluded": counters[
            "demo_records_excluded"
        ],
        "filters": {
            "search": search,
            "decision": decision,
            "address_type": address_type,
            "minimum_edge_count": min_edge_count,
            "include_demo": include_demo,
        },
        "limits": {
            "max_nodes": max_nodes,
            "max_edges": max_edges,
        },
        "total_nodes_before_limit": len(all_filtered_nodes),
        "total_edges_before_limit": len(filtered_edges),
        "returned_node_count": len(nodes),
        "returned_edge_count": len(cleaned_edges),
        "truncated": truncated,
        "truncation_notice": truncation_notice,
        "nodes": nodes,
        "edges": cleaned_edges,
    }


@app.get("/network/map")
def get_network_map(
    search: str = "",
    decision: str = "ALL",
    address_type: str = "ALL",
    min_edge_count: int = 1,
    include_demo: bool = True,
    max_nodes: int = NETWORK_MAP_DEFAULT_MAX_NODES,
    max_edges: int = NETWORK_MAP_DEFAULT_MAX_EDGES,
    authorized: bool = Depends(require_dashboard_login),
):
    safe_search = str(search or "").strip()[:200]

    safe_decision = str(decision or "ALL").strip().upper()
    if safe_decision not in NETWORK_MAP_ALLOWED_DECISIONS:
        safe_decision = "ALL"

    safe_address_type = str(
        address_type or "ALL"
    ).strip().upper()

    address_aliases = {
        "PRIVATE": "LOCAL",
        "PRIVATE/LOCAL": "LOCAL",
        "PUBLIC": "EXTERNAL",
        "PUBLIC/EXTERNAL": "EXTERNAL",
    }
    safe_address_type = address_aliases.get(
        safe_address_type,
        safe_address_type,
    )

    if safe_address_type not in NETWORK_MAP_ALLOWED_ADDRESS_TYPES:
        safe_address_type = "ALL"

    safe_min_edge_count = max(
        1,
        min(int(min_edge_count), 100000),
    )
    safe_max_nodes = max(
        2,
        min(int(max_nodes), NETWORK_MAP_HARD_MAX_NODES),
    )
    safe_max_edges = max(
        1,
        min(int(max_edges), NETWORK_MAP_HARD_MAX_EDGES),
    )

    records, source = _collect_network_map_records()

    return JSONResponse(
        _build_network_map_payload(
            records,
            source=source,
            search=safe_search,
            decision=safe_decision,
            address_type=safe_address_type,
            min_edge_count=safe_min_edge_count,
            include_demo=bool(include_demo),
            max_nodes=safe_max_nodes,
            max_edges=safe_max_edges,
        )
    )

def _find_alert_for_detail(
    alert_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Find an alert using the same identifier exposed to the UI.

    Current memory buffers are searched first. The existing JSONL
    prediction history is used only as a read-only fallback when
    the event has already left the in-memory buffers.
    """
    requested_id = str(
        alert_id or ""
    ).strip()

    if not requested_id:
        return None

    candidate_keys = (
        "detail_id",
        "flow_id",
        "alert_id",
        "event_id",
        "uid",
        "id",
        "replay_original_id",
        "ts_unix",
        "timestamp",
    )

    def match_record(
        record: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(record, dict):
            return None

        flattened = _flatten_metrics_record(record)

        canonical_id = _get_alert_detail_id(
            flattened
        )

        if canonical_id == requested_id:
            return flattened

        for key in candidate_keys:
            value = flattened.get(key)

            if value is None:
                continue

            if str(value).strip() == requested_id:
                return flattened

        return None

    for item in list(alerts) + list(recent):
        matched = match_record(item)

        if matched is not None:
            return matched

    history_path = Path(API_LOG_PATH)

    if not history_path.is_absolute():
        project_history_path = (
            PROJECT_ROOT / history_path
        )

        if project_history_path.is_file():
            history_path = project_history_path

    if history_path.is_file():
        history_records = _read_metrics_jsonl(
            history_path
        )

        for item in reversed(history_records):
            matched = match_record(item)

            if matched is not None:
                return matched

    return None


# =========================================================
# Human alert review status
# =========================================================
# Human status is independent from raw model severity and
# the final context-adjusted decision. Updating this store
# never changes anomaly scoring or decision logic.


def _status_timestamp_local() -> str:
    return (
        datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _normalise_human_status(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()

    aliases = {
        "new": "New",
        "seen": "Seen",
        "under review": "Under review",
        "under_review": "Under review",
        "under-review": "Under review",
        "resolved": "Resolved",
        "false positive": "False positive",
        "false_positive": "False positive",
        "false-positive": "False positive",
    }

    return aliases.get(text)


def _read_alert_status_store_unlocked() -> Dict[str, Dict[str, Any]]:
    if not ALERT_STATUS_PATH.is_file():
        return {}

    try:
        parsed = json.loads(
            ALERT_STATUS_PATH.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if not isinstance(parsed, dict):
        return {}

    cleaned: Dict[str, Dict[str, Any]] = {}

    for alert_id, record in parsed.items():
        if not isinstance(record, dict):
            continue

        normalised_status = _normalise_human_status(
            record.get("status")
        )

        if normalised_status is None:
            continue

        cleaned[str(alert_id)] = {
            "status": normalised_status,
            "updated_at": record.get("updated_at"),
        }

    return cleaned


def _read_alert_status_store() -> Dict[str, Dict[str, Any]]:
    with ALERT_STATUS_LOCK:
        return _read_alert_status_store_unlocked()


def _write_alert_status_store_unlocked(
    store: Dict[str, Dict[str, Any]],
) -> None:
    ALERT_STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = ALERT_STATUS_PATH.with_suffix(
        ".json.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            store,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        ALERT_STATUS_PATH,
    )


def _status_record_from_store(
    alert_id: str,
    store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    requested_id = str(alert_id).strip()
    record = store.get(requested_id)

    if not isinstance(record, dict):
        return {
            "status": "New",
            "updated_at": None,
        }

    return {
        "status": (
            _normalise_human_status(
                record.get("status")
            )
            or "New"
        ),
        "updated_at": record.get("updated_at"),
    }


def _get_alert_status(
    alert_id: str,
) -> Dict[str, Any]:
    store = _read_alert_status_store()

    return _status_record_from_store(
        alert_id,
        store,
    )


def _set_alert_status(
    alert_id: str,
    status: str,
) -> Dict[str, Any]:
    requested_id = str(alert_id).strip()
    normalised_status = _normalise_human_status(
        status
    )

    if normalised_status is None:
        raise ValueError(
            "Invalid human review status."
        )

    record = {
        "status": normalised_status,
        "updated_at": _status_timestamp_local(),
    }

    with ALERT_STATUS_LOCK:
        store = (
            _read_alert_status_store_unlocked()
        )

        store[requested_id] = record

        _write_alert_status_store_unlocked(
            store
        )

    return dict(record)


def _attach_human_status(
    alert: Dict[str, Any],
    status_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    copied_alert = dict(alert)

    alert_id = (
        _get_alert_detail_id(alert)
        or ""
    )

    status_record = _status_record_from_store(
        alert_id,
        status_store,
    )

    copied_alert["human_status"] = (
        status_record["status"]
    )

    copied_alert["human_status_updated_at"] = (
        status_record["updated_at"]
    )

    return copied_alert


@app.get("/alerts/{alert_id}/status")
def get_alert_human_status(
    alert_id: str,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    if _find_alert_for_detail(alert_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    status_record = _get_alert_status(
        alert_id
    )

    return JSONResponse(
        {
            "alert_id": str(alert_id),
            **status_record,
        }
    )


@app.post("/alerts/{alert_id}/status")
def update_alert_human_status(
    alert_id: str,
    payload: AlertStatusUpdate,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    if _find_alert_for_detail(alert_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    normalised_status = _normalise_human_status(
        payload.status
    )

    if normalised_status is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Invalid human review status."
                ),
                "allowed_statuses": list(
                    ALERT_HUMAN_STATUSES
                ),
            },
        )

    try:
        status_record = _set_alert_status(
            alert_id,
            normalised_status,
        )
    except (OSError, ValueError) as exc:
        print(
            "ALERT STATUS SAVE ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The human review status "
                "could not be saved."
            ),
        ) from exc

    return JSONResponse(
        {
            "alert_id": str(alert_id),
            **status_record,
            "message": (
                "Human review status updated."
            ),
        }
    )


# =========================================================
# Analyst feedback collection
# =========================================================
# This feedback history is collected for later human review.
# It is not active learning: records written here do not
# whitelist traffic and do not retrain or update the model.


def _feedback_timestamp_local() -> str:
    """
    Return the current local time as an ISO-8601 timestamp.

    The UTC offset is included so the timestamp remains
    unambiguous, for example:
    2026-06-15T00:00:13+03:00
    """
    return (
        datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _find_false_positive_feedback_unlocked(
    alert_id: str,
) -> Optional[Dict[str, Any]]:
    requested_id = str(alert_id).strip()

    if not requested_id or not FEEDBACK_HISTORY_PATH.is_file():
        return None

    latest_match: Optional[Dict[str, Any]] = None

    try:
        with FEEDBACK_HISTORY_PATH.open(
            "r",
            encoding="utf-8",
        ) as feedback_file:
            for line in feedback_file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # One malformed line must not break the
                    # complete alert detail page.
                    continue

                if not isinstance(record, dict):
                    continue

                if (
                    str(record.get("alert_id", "")).strip()
                    == requested_id
                    and record.get("user_feedback")
                    == "false_positive"
                ):
                    latest_match = record

    except OSError:
        return None

    return latest_match


def _find_false_positive_feedback(
    alert_id: str,
) -> Optional[Dict[str, Any]]:
    with FEEDBACK_HISTORY_LOCK:
        return _find_false_positive_feedback_unlocked(
            alert_id
        )


def _feedback_status_from_record(
    record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "is_false_positive": False,
            "user_feedback": None,
            "timestamp": None,
        }

    return {
        "is_false_positive": True,
        "user_feedback": "false_positive",
        "timestamp": record.get("timestamp"),
    }


def _build_false_positive_record(
    alert_id: str,
    alert: Dict[str, Any],
) -> Dict[str, Any]:
    final_decision = _first_present(
        alert,
        "final_label",
        "display_label",
        "final_severity",
        "severity",
    )

    destination_port = alert.get("dest_port")

    return {
        "alert_id": str(alert_id).strip(),
        "timestamp": _feedback_timestamp_local(),
        "source_ip": str(
            alert.get("src_ip") or ""
        ),
        "destination_ip": str(
            alert.get("dest_ip") or ""
        ),
        "protocol": str(
            alert.get("proto") or ""
        ).upper(),
        "destination_port": (
            destination_port
            if destination_port not in (None, "")
            else ""
        ),
        "traffic_class": str(
            alert.get("traffic_class") or ""
        ),
        "raw_severity": str(
            alert.get("raw_severity") or ""
        ).upper(),
        "final_decision": str(
            final_decision or ""
        ).upper(),
        "user_feedback": "false_positive",
    }


def _normalise_alert_detail(
    alert: Dict[str, Any],
    temporal_context: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    contributor_features = alert.get(
        "feature_contributors",
        alert.get(
            "top_features",
            [],
        ),
    )

    if not isinstance(
        contributor_features,
        list,
    ):
        contributor_features = []

    contributors = []

    for feature in contributor_features:
        if not isinstance(
            feature,
            dict,
        ):
            continue

        contributors.append(
            {
                "feature": _safe_detail_value(
                    feature.get("name")
                ),
                "error": _safe_detail_value(
                    feature.get("err")
                ),
                "scaled_value": _safe_detail_value(
                    feature.get("x")
                ),
                "reconstructed_value": _safe_detail_value(
                    feature.get("x_hat")
                ),
            }
        )

    raw_severity = _safe_detail_value(alert.get("raw_severity"))
    final_severity = _safe_detail_value(
        alert.get("final_label")
        or alert.get("display_label")
        or alert.get("final_severity")
        or alert.get("severity")
    )

    contextual_reason = (
        alert.get("adjustment_reason")
        or alert.get("benign_reason")
        or alert.get("traffic_note")
        or alert.get("display_label_reason")
        or "-"
    )

    simple_explanation = (
        alert.get("simple_explanation")
        or alert.get("explanation")
        or alert.get("summary")
        or ""
    )

    analyst_explanation = (
        alert.get("analyst_explanation")
        or alert.get("possible_explanation")
        or alert.get("what_to_check")
        or simple_explanation
    )

    technical_explanation = (
        alert.get("technical_explanation")
        or alert.get("full_explanation")
        or alert.get("legacy_explanation")
        or alert.get("explanation")
        or simple_explanation
    )
    
    recommended_action = (
    alert.get("recommended_action")
    or build_recommended_action(alert)
    )

    full_explanation_parts = [
        alert.get("summary"),
        alert.get("interpretation"),
        simple_explanation,
        analyst_explanation,
        technical_explanation,
        alert.get("possible_explanation"),
        alert.get("what_to_check"),
    ]

    full_explanation = "\n\n".join(
        str(part).strip()
        for part in full_explanation_parts
        if part is not None and str(part).strip()
    )

    if not isinstance(
        temporal_context,
        dict,
    ):
        temporal_context = (
            compute_temporal_context(
                alert,
                {
                    "records": [alert],
                    "scope": "current buffer",
                },
            )
        )
    
    return {
        "flow_id": _safe_detail_value(alert.get("flow_id")),
        "final_decision": final_severity,
        "raw_model_severity": raw_severity,
        "raw_to_final": f"{raw_severity} → {final_severity}",
        "anomaly_score": _safe_detail_value(alert.get("ae_score")),
        "short_reason": _safe_detail_value(
            alert.get("summary")
            or alert.get("display_label_reason")
            or alert.get("adjustment_reason")
        ),
        "source_ip": _safe_detail_value(alert.get("src_ip")),
        "source_port": _safe_detail_value(alert.get("src_port")),
        "destination_ip": _safe_detail_value(alert.get("dest_ip")),
        "destination_port": _safe_detail_value(alert.get("dest_port")),
        "protocol": _safe_detail_value(alert.get("proto")),
        "traffic_class": _safe_detail_value(alert.get("traffic_class")),
        "repeat_count": _safe_detail_value(
            alert.get("repeat_count")
        ),
        "contextual_reason": _safe_detail_value(
            contextual_reason
        ),
        "recommended_action": _safe_detail_value(
            recommended_action
        ),
        "temporal_context": temporal_context,
        "top_anomaly_contributors": contributors,

        "simple_explanation": _safe_detail_value(simple_explanation),
        "analyst_explanation": _safe_detail_value(analyst_explanation),
        "technical_explanation": _safe_detail_value(technical_explanation),

        "full_explanation": _safe_detail_value(full_explanation),
    }


@app.get("/alerts/{alert_id}/detail")
def get_alert_detail(
    alert_id: str,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    alert = _find_alert_for_detail(
        alert_id
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    temporal_context = (
        compute_temporal_context(
            alert,
            _collect_temporal_history(),
        )
    )

    detail = _normalise_alert_detail(
        alert,
        temporal_context=temporal_context,
    )

    feedback_record = (
        _find_false_positive_feedback(
            alert_id
        )
    )

    detail["feedback"] = (
        _feedback_status_from_record(
            feedback_record
        )
    )

    human_status = _get_alert_status(
        alert_id
    )

    # Preserve compatibility with false-positive feedback
    # saved before the status workflow was introduced.
    if (
        feedback_record is not None
        and human_status["status"] == "New"
        and human_status["updated_at"] is None
    ):
        try:
            human_status = _set_alert_status(
                alert_id,
                "False positive",
            )
        except (OSError, ValueError) as exc:
            print(
                "ALERT STATUS MIGRATION ERROR:",
                repr(exc),
            )

    detail["human_status"] = (
        human_status["status"]
    )

    detail["human_status_updated_at"] = (
        human_status["updated_at"]
    )

    detail["is_demo_alert"] = (
        _blocklist_alert_is_demo(alert)
    )

    detail["blocklist_state"] = (
        _blocklist_public_state(
            alert.get("src_ip")
        )
    )

    return JSONResponse(detail)


@app.post("/alerts/{alert_id}/feedback")
def submit_alert_feedback(
    alert_id: str,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    alert = _find_alert_for_detail(
        alert_id
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    # Analyst feedback collection only. This endpoint does not
    # whitelist the flow, retrain the autoencoder, change model
    # thresholds, or modify the final decision logic.
    with FEEDBACK_HISTORY_LOCK:
        existing_record = (
            _find_false_positive_feedback_unlocked(
                alert_id
            )
        )

        if existing_record is not None:
            try:
                human_status = _set_alert_status(
                    alert_id,
                    "False positive",
                )
            except (OSError, ValueError) as exc:
                print(
                    "ALERT STATUS SAVE ERROR:",
                    repr(exc),
                )

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "The false-positive feedback exists, "
                        "but its human status could not be saved."
                    ),
                ) from exc

            return JSONResponse(
                {
                    "status": "already_marked",
                    "already_marked": True,
                    "message": (
                        "This alert was already marked as "
                        "a false positive."
                    ),
                    "feedback": (
                        _feedback_status_from_record(
                            existing_record
                        )
                    ),
                    "human_status": human_status,
                }
            )

        feedback_record = (
            _build_false_positive_record(
                alert_id=alert_id,
                alert=alert,
            )
        )

        try:
            FEEDBACK_HISTORY_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with FEEDBACK_HISTORY_PATH.open(
                "a",
                encoding="utf-8",
            ) as feedback_file:
                feedback_file.write(
                    json.dumps(
                        feedback_record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                feedback_file.flush()

        except (OSError, ValueError) as exc:
            print(
                "ALERT FEEDBACK SAVE ERROR:",
                repr(exc),
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "The analyst feedback could not be saved."
                ),
            ) from exc

    try:
        human_status = _set_alert_status(
            alert_id,
            "False positive",
        )
    except (OSError, ValueError) as exc:
        print(
            "ALERT STATUS SAVE ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The feedback was recorded, but the "
                "human review status could not be saved."
            ),
        ) from exc

    return JSONResponse(
        {
            "status": "saved",
            "already_marked": False,
            "message": (
                "Alert marked as a false positive."
            ),
            "feedback": (
                _feedback_status_from_record(
                    feedback_record
                )
            ),
            "human_status": human_status,
        }
    )

def _iso_utc_timestamp(value: Any) -> Optional[str]:
    """
    Convert an epoch or ISO timestamp to an ISO-8601 UTC string.

    Missing or invalid values become None so the JSON report
    remains explicit and machine-readable.
    """
    timestamp = _safe_metric_timestamp(value)

    if timestamp is None:
        return None

    try:
        return (
            datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return None


def _safe_report_filename_part(value: Any) -> str:
    text = str(value or "").strip()

    cleaned = "".join(
        character
        if character.isalnum()
        or character in {"-", "_", "."}
        else "_"
        for character in text
    )

    cleaned = cleaned.strip("._")

    return cleaned[:100] or "unknown"


def _report_repeat_window(
    alert: Dict[str, Any],
) -> Optional[int]:
    repeat_info = alert.get("repeat_info")

    if not isinstance(repeat_info, dict):
        repeat_info = {}

    debug = alert.get("debug")

    if not isinstance(debug, dict):
        debug = {}

    debug_repeat_info = debug.get("repeat_info")

    if not isinstance(debug_repeat_info, dict):
        debug_repeat_info = {}

    candidates = (
        alert.get("repeat_window_s"),
        repeat_info.get("repeat_window_s"),
        debug_repeat_info.get("repeat_window_s"),
    )

    for value in candidates:
        if value is None or value == "":
            continue

        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue

    return None


def _build_alert_json_report(
    alert_id: str,
    alert: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a read-only report from an already stored alert.

    This function does not change model scoring, contextual
    filtering, repeat logic, or the final decision.
    """
    temporal_context = compute_temporal_context(
        alert,
        _collect_temporal_history(),
    )

    detail = _normalise_alert_detail(
        alert,
        temporal_context=temporal_context,
    )

    contributors = detail.get(
        "top_anomaly_contributors",
        [],
    )

    if not isinstance(contributors, list):
        contributors = []

    # Keep the five highest stored contributors in the report.
    top_contributors = contributors[:5]

    return {
        "generated_at": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "alert_id": str(alert_id),
        "timestamp": _iso_utc_timestamp(
            _first_present(
                alert,
                "ts_unix",
                "timestamp",
                "time",
            )
        ),
        "source_ip": detail.get(
            "source_ip",
            "-",
        ),
        "destination_ip": detail.get(
            "destination_ip",
            "-",
        ),
        "source_port": detail.get(
            "source_port",
            "-",
        ),
        "destination_port": detail.get(
            "destination_port",
            "-",
        ),
        "protocol": detail.get(
            "protocol",
            "-",
        ),
        "traffic_class": detail.get(
            "traffic_class",
            "-",
        ),
        "raw_model_severity": detail.get(
            "raw_model_severity",
            "-",
        ),
        "final_decision": detail.get(
            "final_decision",
            "-",
        ),
        "anomaly_score": detail.get(
            "anomaly_score",
            "-",
        ),
        "contextual_reason": detail.get(
            "contextual_reason",
            "-",
        ),
        "repeat_count": detail.get(
            "repeat_count",
            "-",
        ),

        # Number of seconds, or null when unavailable.
        "repeat_window": _report_repeat_window(
            alert
        ),

        "top_contributors": top_contributors,

        "simple_explanation": detail.get(
            "simple_explanation",
            "-",
        ),
        "analyst_explanation": detail.get(
            "analyst_explanation",
            "-",
        ),
        "technical_explanation": detail.get(
            "technical_explanation",
            "-",
        ),
        "recommended_action": detail.get(
            "recommended_action",
            "-",
        ),
    }


@app.get("/alerts/{alert_id}/report.json")
def download_alert_json_report(
    alert_id: str,
    authorized: bool = Depends(
        require_dashboard_login
    ),
):
    alert = _find_alert_for_detail(
        alert_id
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    report = _build_alert_json_report(
        alert_id=alert_id,
        alert=alert,
    )

    json_content = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"

    safe_alert_id = _safe_report_filename_part(
        alert_id
    )

    generated_suffix = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    filename = (
        f"ids_alert_{safe_alert_id}_"
        f"{generated_suffix}.json"
    )

    # Saving a local copy is optional. A filesystem error
    # must not prevent the browser download.
    if SAVE_JSON_REPORT_COPIES:
        try:
            REPORTS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                REPORTS_DIR
                / filename
            ).write_text(
                json_content,
                encoding="utf-8",
            )

        except OSError as exc:
            print(
                "JSON REPORT SAVE ERROR:",
                repr(exc),
            )

    return Response(
        content=json_content,
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Cache-Control": "no-store",
        },
    )

@app.post("/alerts/clear")
def clear_alerts(
    authorized: bool = Depends(require_dashboard_login),
):
    alerts.clear()
    recent.clear()
    recent_repeat_memory.clear()
    update_gauges(len(alerts))
    return {"status": "ok", "cleared": True}




@app.on_event("startup")
def start_mode_services() -> None:
    print(
        f"[runtime] IDS mode: {IDS_MODE.upper()}"
    )

    if (
        IDS_MODE == "replay"
        and REPLAY_AUTO_START
    ):
        try:
            result = start_replay()
            print(
                "[replay] Automatic start:",
                result.get("status"),
            )
        except HTTPException as exc:
            print(
                "[replay] Automatic start failed:",
                exc.detail,
            )


@app.on_event("shutdown")
def stop_mode_services() -> None:
    if IDS_MODE == "replay":
        stop_replay()


@app.post("/predict")
def predict(
    req: PredictRequest,
    authorized: bool = Depends(require_api_key),
):
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

        
        event = enrich_event_context(event)

        
        event = attach_repeat_context(event)

        
        event = apply_final_decision_logic(event)

        event["display_label"], event["display_label_reason"] = make_display_label_with_reason(event)
        event["final_label"] = event["display_label"]
        event["severity"] = str(event.get("final_severity", event.get("severity", "UNKNOWN"))).upper()

        event = attach_top_feature_errors(event, x, raw_map)

        event = attach_explanations(event)

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

# =========================================================
# MANUAL IP BLOCKLIST MANAGEMENT
# =========================================================
# Add the constants near the other data-file constants, the Pydantic
# models near AlertStatusUpdate, the helpers after _find_alert_for_detail,
# and the routes with the other authenticated dashboard routes.
#
# This feature stores operator intent only. It does not run iptables,
# nftables, sudo, or any other firewall command.

# ---------- Constants: place near ALERT_STATUS_PATH ----------

BLOCKLIST_PATH = PROJECT_ROOT / "data" / "blocked_ips.json"
BLOCKLIST_HISTORY_PATH = PROJECT_ROOT / "data" / "blocklist_history.jsonl"
BLOCKLIST_LOCK = Lock()

BLOCKLIST_ENFORCEMENT_STATE = "not_enforced"
BLOCKLIST_ENFORCEMENT_MESSAGE = (
    "Stored in the operator blocklist. "
    "Firewall enforcement is not enabled."
)
BLOCKLIST_FIREWALL_ENABLED = False
BLOCKLIST_MAX_REASON_LENGTH = 500


# ---------- Pydantic models: place near AlertStatusUpdate ----------

class BlocklistCreateRequest(BaseModel):
    ip_address: str
    reason: str
    source_alert_id: Optional[str] = None
    confirm_dangerous: bool = False
    update_existing_reason: bool = False


# ---------- Helpers: place after _find_alert_for_detail ----------

def _blocklist_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
    )


def _normalise_ip_address(value: Any) -> Tuple[str, Any]:
    text = str(value or "").strip()

    if not text:
        raise ValueError("IP address is required.")

    try:
        parsed = ipaddress.ip_address(text)
    except ValueError as exc:
        raise ValueError(
            "Enter a valid IPv4 or IPv6 address."
        ) from exc

    return str(parsed), parsed


def _blocklist_address_type(parsed: Any) -> str:
    if isinstance(parsed, ipaddress.IPv4Address):
        return "IPv4"

    if isinstance(parsed, ipaddress.IPv6Address):
        return "IPv6"

    return "Unknown"


def _blocklist_own_addresses() -> set[str]:
    addresses: set[str] = set()

    try:
        interface_map = psutil.net_if_addrs()
    except Exception as exc:
        print(
            "BLOCKLIST ADDRESS DISCOVERY ERROR:",
            repr(exc),
        )
        return addresses

    for interface_addresses in interface_map.values():
        for interface_address in interface_addresses:
            raw_address = str(
                getattr(interface_address, "address", "")
                or ""
            ).strip()

            if not raw_address:
                continue

            # Linux may expose an IPv6 scope suffix such as %eth0.
            address_without_scope = raw_address.split("%", 1)[0]

            try:
                parsed = ipaddress.ip_address(
                    address_without_scope
                )
            except ValueError:
                continue

            addresses.add(str(parsed))

    return addresses


def _blocklist_is_broadcast_like(parsed: Any) -> bool:
    if not isinstance(parsed, ipaddress.IPv4Address):
        return False

    octets = str(parsed).split(".")

    return (
        str(parsed) == "255.255.255.255"
        or (len(octets) == 4 and octets[-1] == "255")
    )


def _blocklist_safety_assessment(
    parsed: Any,
    request: Request,
) -> Dict[str, Any]:
    canonical_ip = str(parsed)
    confirmation_reasons: List[str] = []
    warnings: List[str] = []

    if parsed.is_loopback:
        confirmation_reasons.append(
            "The address is a loopback address used by the local machine."
        )

    if parsed.is_unspecified:
        confirmation_reasons.append(
            "The address is unspecified and does not identify a normal remote host."
        )

    if parsed.is_multicast:
        confirmation_reasons.append(
            "The address is multicast and may represent shared infrastructure traffic."
        )

    if parsed.is_link_local:
        confirmation_reasons.append(
            "The address is link-local and may be required for local network operation."
        )

    if _blocklist_is_broadcast_like(parsed):
        confirmation_reasons.append(
            "The IPv4 address is broadcast-like and may affect many hosts."
        )

    client_host = ""

    if request.client is not None:
        client_host = str(request.client.host or "").strip()

    if client_host:
        client_host_without_scope = client_host.split("%", 1)[0]

        try:
            normalised_client = str(
                ipaddress.ip_address(
                    client_host_without_scope
                )
            )
        except ValueError:
            normalised_client = ""

        if normalised_client == canonical_ip:
            confirmation_reasons.append(
                "This is the current dashboard client address. "
                "Blocking it in a future firewall adapter could disconnect this operator."
            )
    else:
        warnings.append(
            "The current dashboard client address could not be identified."
        )

    own_addresses = _blocklist_own_addresses()

    if canonical_ip in own_addresses:
        confirmation_reasons.append(
            "This address belongs to the machine running the dashboard. "
            "Future firewall enforcement could interrupt the monitored system."
        )

    warnings.append(
        "The application does not reliably identify the default gateway or DNS infrastructure. "
        "Verify the address before adding it."
    )

    warnings.append(
        "Client-address detection may reflect a reverse proxy rather than the original browser."
    )

    return {
        "confirmation_required": bool(confirmation_reasons),
        "confirmation_reasons": confirmation_reasons,
        "warnings": warnings,
        "current_client_identified": bool(client_host),
        "dashboard_own_addresses": sorted(own_addresses),
    }


def _clean_blocklist_record(
    key: str,
    value: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    candidate = value.get("ip_address") or key

    try:
        canonical_ip, parsed = _normalise_ip_address(candidate)
    except ValueError:
        return None

    enforcement_state = str(
        value.get("enforcement_state")
        or BLOCKLIST_ENFORCEMENT_STATE
    ).strip()

    # No firewall adapter exists in this task. Do not trust a file value
    # that incorrectly claims active enforcement.
    if enforcement_state != BLOCKLIST_ENFORCEMENT_STATE:
        enforcement_state = BLOCKLIST_ENFORCEMENT_STATE

    return {
        "ip_address": canonical_ip,
        "address_type": _blocklist_address_type(parsed),
        "reason": str(value.get("reason") or "").strip(),
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
        "source_alert_id": value.get("source_alert_id"),
        "source_final_decision": value.get("source_final_decision"),
        "source_origin": str(
            value.get("source_origin") or "manual"
        ),
        "demo_derived": bool(value.get("demo_derived", False)),
        "enforcement_state": enforcement_state,
        "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
    }


def _read_blocklist_store_unlocked() -> Tuple[
    Dict[str, Dict[str, Any]],
    Optional[str],
    bool,
]:
    if not BLOCKLIST_PATH.is_file():
        return {}, None, False

    try:
        parsed = json.loads(
            BLOCKLIST_PATH.read_text(encoding="utf-8")
        )
    except OSError as exc:
        return (
            {},
            f"The blocklist file could not be read: {exc}",
            False,
        )
    except json.JSONDecodeError as exc:
        return (
            {},
            "The blocklist file contains invalid JSON. "
            f"A preserved backup will be created before the next write: {exc}",
            True,
        )

    if not isinstance(parsed, dict):
        return (
            {},
            "The blocklist JSON root is not an object. "
            "A preserved backup will be created before the next write.",
            True,
        )

    cleaned: Dict[str, Dict[str, Any]] = {}

    for key, value in parsed.items():
        record = _clean_blocklist_record(str(key), value)

        if record is None:
            continue

        cleaned[record["ip_address"]] = record

    return cleaned, None, False


def _preserve_invalid_blocklist_unlocked() -> Optional[Path]:
    if not BLOCKLIST_PATH.is_file():
        return None

    suffix = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    backup_path = BLOCKLIST_PATH.with_name(
        f"{BLOCKLIST_PATH.stem}.invalid-{suffix}.json"
    )

    os.replace(BLOCKLIST_PATH, backup_path)
    return backup_path


def _prepare_blocklist_store_for_write_unlocked() -> Tuple[
    Dict[str, Dict[str, Any]],
    Optional[str],
]:
    store, warning, requires_preservation = (
        _read_blocklist_store_unlocked()
    )

    if not requires_preservation:
        return store, warning

    backup_path = _preserve_invalid_blocklist_unlocked()

    backup_message = (
        f"Invalid blocklist data was preserved as {backup_path.name}."
        if backup_path is not None
        else "Invalid blocklist data was detected."
    )

    return {}, backup_message


def _write_blocklist_store_unlocked(
    store: Dict[str, Dict[str, Any]],
) -> None:
    BLOCKLIST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = BLOCKLIST_PATH.with_suffix(
        ".json.tmp"
    )

    serialisable_store = {
        ip_address: {
            key: value
            for key, value in record.items()
            if key not in {
                "address_type",
                "enforcement_message",
            }
        }
        for ip_address, record in store.items()
    }

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as temporary_file:
        json.dump(
            serialisable_store,
            temporary_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())

    os.replace(temporary_path, BLOCKLIST_PATH)


def _append_blocklist_history_unlocked(
    record: Dict[str, Any],
) -> Optional[str]:
    try:
        BLOCKLIST_HISTORY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with BLOCKLIST_HISTORY_PATH.open(
            "a",
            encoding="utf-8",
        ) as history_file:
            history_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            history_file.flush()
            os.fsync(history_file.fileno())

        return None
    except (OSError, ValueError) as exc:
        print(
            "BLOCKLIST AUDIT WRITE ERROR:",
            repr(exc),
        )
        return (
            "The blocklist was updated, but the optional audit history "
            "could not be written."
        )


def _read_blocklist_store() -> Tuple[
    Dict[str, Dict[str, Any]],
    Optional[str],
]:
    with BLOCKLIST_LOCK:
        store, warning, _ = _read_blocklist_store_unlocked()
        return store, warning


def _blocklist_final_decision(alert: Dict[str, Any]) -> str:
    return str(
        _first_present(
            alert,
            "final_label",
            "display_label",
            "final_decision",
            "final_severity",
            "severity",
        )
        or "UNKNOWN"
    ).strip().upper()


def _blocklist_alert_is_demo(alert: Dict[str, Any]) -> bool:
    return bool(
        alert.get("replay_mode")
        or alert.get("demo_mode")
        or alert.get("is_demo")
        or str(alert.get("source_mode") or "").lower()
        in {"demo", "replay", "demo_mode", "replay_mode"}
    )


def _blocklist_record_for_ip(
    ip_address: Any,
) -> Optional[Dict[str, Any]]:
    try:
        canonical_ip, _ = _normalise_ip_address(ip_address)
    except ValueError:
        return None

    store, _ = _read_blocklist_store()
    record = store.get(canonical_ip)

    return dict(record) if isinstance(record, dict) else None


def _blocklist_public_state(
    ip_address: Any,
) -> Dict[str, Any]:
    record = _blocklist_record_for_ip(ip_address)

    return {
        "is_blocked": record is not None,
        "record": record,
        "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
        "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
        "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
    }


def _blocklist_sorted_records(
    store: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = [dict(record) for record in store.values()]

    records.sort(
        key=lambda record: str(
            record.get("created_at") or ""
        ),
        reverse=True,
    )

    return records


# ---------- UI route: place with the other /ui/... routes ----------

@app.get("/ui/blocklist", response_class=HTMLResponse)
def ui_blocklist(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_blocklist.html",
        {"request": request},
    )


# ---------- API endpoints ----------

@app.get("/blocklist")
def get_blocklist(
    authorized: bool = Depends(require_dashboard_login),
):
    store, storage_warning = _read_blocklist_store()
    records = _blocklist_sorted_records(store)

    return JSONResponse(
        {
            "success": True,
            "total": len(records),
            "records": records,
            "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
            "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
            "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
            "storage_warning": storage_warning,
        }
    )


@app.get("/blocklist/{ip_address}")
def get_blocklist_record(
    ip_address: str,
    authorized: bool = Depends(require_dashboard_login),
):
    try:
        canonical_ip, _ = _normalise_ip_address(ip_address)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    store, storage_warning = _read_blocklist_store()
    record = store.get(canonical_ip)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="IP address is not present in the blocklist.",
        )

    return JSONResponse(
        {
            "success": True,
            "record": record,
            "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
            "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
            "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
            "storage_warning": storage_warning,
        }
    )


@app.post("/blocklist")
def add_blocklist_record(
    payload: BlocklistCreateRequest,
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    try:
        canonical_ip, parsed = _normalise_ip_address(
            payload.ip_address
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    reason = str(payload.reason or "").strip()

    if not reason:
        raise HTTPException(
            status_code=400,
            detail="A reason is required.",
        )

    if len(reason) > BLOCKLIST_MAX_REASON_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                "The reason is too long. "
                f"Use at most {BLOCKLIST_MAX_REASON_LENGTH} characters."
            ),
        )

    source_alert_id = str(
        payload.source_alert_id or ""
    ).strip()
    source_alert: Optional[Dict[str, Any]] = None
    source_final_decision: Optional[str] = None
    source_origin = "manual"
    demo_derived = False

    if source_alert_id:
        source_alert = _find_alert_for_detail(source_alert_id)

        if source_alert is None:
            raise HTTPException(
                status_code=404,
                detail="The associated alert could not be found.",
            )

        alert_source_ip = str(
            source_alert.get("src_ip") or ""
        ).strip()

        try:
            normalised_alert_ip, _ = _normalise_ip_address(
                alert_source_ip
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The associated alert does not contain a valid source IP."
                ),
            ) from exc

        if normalised_alert_ip != canonical_ip:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The submitted IP address does not match the associated "
                    "alert source IP."
                ),
            )

        source_final_decision = _blocklist_final_decision(
            source_alert
        )
        demo_derived = _blocklist_alert_is_demo(source_alert)
        source_origin = (
            "demo-derived" if demo_derived else "live-alert"
        )

    safety = _blocklist_safety_assessment(
        parsed,
        request,
    )

    if (
        safety["confirmation_required"]
        and not payload.confirm_dangerous
    ):
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "confirmation_required": True,
                "message": (
                    "Explicit confirmation is required for this address."
                ),
                "ip_address": canonical_ip,
                "confirmation_reasons": safety[
                    "confirmation_reasons"
                ],
                "warnings": safety["warnings"],
                "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
                "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
                "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
            },
        )

    timestamp = _blocklist_timestamp()

    with BLOCKLIST_LOCK:
        try:
            store, storage_warning = (
                _prepare_blocklist_store_for_write_unlocked()
            )
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The invalid blocklist file could not be preserved. "
                    "No change was made."
                ),
            ) from exc

        existing = store.get(canonical_ip)

        if existing is not None and not payload.update_existing_reason:
            return JSONResponse(
                {
                    "success": True,
                    "already_blocked": True,
                    "updated": False,
                    "message": (
                        "This IP address is already present in the blocklist."
                    ),
                    "record": existing,
                    "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
                    "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
                    "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
                    "warnings": safety["warnings"],
                    "storage_warning": storage_warning,
                }
            )

        previous_record = (
            dict(existing) if isinstance(existing, dict) else None
        )

        created_at = (
            existing.get("created_at")
            if isinstance(existing, dict)
            else timestamp
        ) or timestamp

        record = {
            "ip_address": canonical_ip,
            "address_type": _blocklist_address_type(parsed),
            "reason": reason,
            "created_at": created_at,
            "updated_at": timestamp,
            "source_alert_id": source_alert_id or None,
            "source_final_decision": source_final_decision,
            "source_origin": source_origin,
            "demo_derived": demo_derived,
            "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
            "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
        }

        store[canonical_ip] = record

        try:
            _write_blocklist_store_unlocked(store)
        except (OSError, ValueError) as exc:
            print(
                "BLOCKLIST SAVE ERROR:",
                repr(exc),
            )
            raise HTTPException(
                status_code=500,
                detail="The blocklist could not be saved.",
            ) from exc

        audit_warning = _append_blocklist_history_unlocked(
            {
                "action": "block",
                "ip_address": canonical_ip,
                "timestamp": timestamp,
                "reason": reason,
                "source_alert_id": source_alert_id or None,
                "source_final_decision": source_final_decision,
                "source_origin": source_origin,
                "demo_derived": demo_derived,
                "previous_record": previous_record,
                "enforcement_result": BLOCKLIST_ENFORCEMENT_STATE,
            }
        )

    return JSONResponse(
        {
            "success": True,
            "already_blocked": existing is not None,
            "updated": existing is not None,
            "message": (
                "Blocklist record updated."
                if existing is not None
                else "IP address added to the operator blocklist."
            ),
            "record": record,
            "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
            "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
            "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
            "warnings": safety["warnings"],
            "storage_warning": storage_warning,
            "audit_warning": audit_warning,
        }
    )


@app.delete("/blocklist/{ip_address}")
def delete_blocklist_record(
    ip_address: str,
    authorized: bool = Depends(require_dashboard_login),
):
    try:
        canonical_ip, _ = _normalise_ip_address(ip_address)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    timestamp = _blocklist_timestamp()

    with BLOCKLIST_LOCK:
        try:
            store, storage_warning = (
                _prepare_blocklist_store_for_write_unlocked()
            )
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The invalid blocklist file could not be preserved. "
                    "No change was made."
                ),
            ) from exc

        existing = store.get(canonical_ip)

        if existing is None:
            raise HTTPException(
                status_code=404,
                detail="IP address is not present in the blocklist.",
            )

        previous_record = dict(existing)
        del store[canonical_ip]

        try:
            _write_blocklist_store_unlocked(store)
        except (OSError, ValueError) as exc:
            print(
                "BLOCKLIST DELETE ERROR:",
                repr(exc),
            )
            raise HTTPException(
                status_code=500,
                detail="The blocklist could not be saved.",
            ) from exc

        audit_warning = _append_blocklist_history_unlocked(
            {
                "action": "unblock",
                "ip_address": canonical_ip,
                "timestamp": timestamp,
                "reason": previous_record.get("reason"),
                "source_alert_id": previous_record.get(
                    "source_alert_id"
                ),
                "previous_record": previous_record,
                "enforcement_result": BLOCKLIST_ENFORCEMENT_STATE,
            }
        )

    return JSONResponse(
        {
            "success": True,
            "removed": True,
            "message": "IP address removed from the operator blocklist.",
            "record": previous_record,
            "enforcement_state": BLOCKLIST_ENFORCEMENT_STATE,
            "firewall_enforcement_enabled": BLOCKLIST_FIREWALL_ENABLED,
            "enforcement_message": BLOCKLIST_ENFORCEMENT_MESSAGE,
            "storage_warning": storage_warning,
            "audit_warning": audit_warning,
        }
    )


# ---------- Alert-detail endpoint integration ----------
# In get_alert_detail(), immediately before `return JSONResponse(detail)`, add:
#
#     detail["is_demo_alert"] = _blocklist_alert_is_demo(alert)
#     detail["blocklist_state"] = _blocklist_public_state(
#         alert.get("src_ip")
#     )
#
# This adds read-only blocklist information and does not change the alert.
