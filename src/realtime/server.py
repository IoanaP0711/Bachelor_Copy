#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import onnxruntime as ort
import psutil
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
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
    make_display_label_with_reason,
)
load_dotenv()

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

REPEAT_WINDOW_S = 45
REPEAT_MAX_EVENTS_PER_KEY = 50
REPEAT_REVIEW_PREV_COUNT = 1
REPEAT_PERSISTENT_PREV_COUNT = 2

recent_repeat_memory: Dict[RepeatKey, Deque[float]] = defaultdict(deque)

COMMON_DESKTOP_PORTS = {
    53, 67, 68, 80, 443, 8080, 8443,
    123, 1900, 5353, 5355,
    25, 465, 587, 993, 995, 110, 143,
}

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

    event["summary"] = bundle["summary"]
    event["interpretation"] = bundle["interpretation"]
    event["explanation"] = bundle["explanation"]
    event["adjustment_reason"] = bundle["adjustment_reason"]
    event["possible_explanation"] = bundle.get("possible_explanation", "")
    event["what_to_check"] = bundle.get("what_to_check", "")

    
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
        "explanation": event.get("explanation", ""),
        "adjustment_reason": (
            event.get("adjustment_reason")
            or event.get("final_severity_reason")
            or ""
        ),
        "possible_explanation": event.get("possible_explanation", ""),
        "what_to_check": event.get("what_to_check", ""),

        
        "display_label": final_label,
        "display_label_reason": event.get("display_label_reason", ""),
        "severity": final_severity,

        
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "static"),
    name="static",
)

templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

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

@app.get("/ui/recent", response_class=HTMLResponse)
def ui_recent(
    request: Request,
    authorized: bool = Depends(require_dashboard_login),
):
    return templates.TemplateResponse(
        "ui_recent.html",
        {"request": request},
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


@app.get("/metrics")
def metrics(
    authorized: bool = Depends(require_dashboard_login),
):
    update_gauges(len(alerts))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recent")
def get_recent(
    limit: int = 50,
    authorized: bool = Depends(require_dashboard_login),
):
    items = list(recent)[-limit:][::-1]
    return {"bands": bands, "recent": items}


@app.get("/alerts")
def get_alerts(
    limit: int = 50,
    authorized: bool = Depends(require_dashboard_login),
):
    update_gauges(len(alerts))
    items = list(alerts)[-limit:][::-1]
    return {"bands": bands, "alerts": items}


@app.post("/alerts/clear")
def clear_alerts(
    authorized: bool = Depends(require_dashboard_login),
):
    alerts.clear()
    recent.clear()
    recent_repeat_memory.clear()
    update_gauges(len(alerts))
    return {"status": "ok", "cleared": True}


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

        
        event = attach_explanations(event)

        
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