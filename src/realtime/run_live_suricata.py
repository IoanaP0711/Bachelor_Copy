#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"


def _to_unix_seconds(ts: Any) -> Optional[float]:
    """
    Suricata timestamps are usually ISO8601 strings.
    Returns seconds since epoch.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        s = ts.strip()
        try:
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                return dt.timestamp()

            if len(s) >= 5 and (s[-5] in ["+", "-"]) and s[-2:].isdigit():
                s = s[:-5] + s[-5:-2] + ":" + s[-2:]

            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    return None


def proto_to_num(proto: Any) -> int:
    """
    Convert protocol to a numeric feature.
    Common mappings:
      ICMP=1, TCP=6, UDP=17 (IANA)
    """
    if proto is None:
        return 0
    if isinstance(proto, (int, float)):
        return int(proto)
    if isinstance(proto, str):
        p = proto.strip().lower()
        if p == "icmp":
            return 1
        if p == "tcp":
            return 6
        if p == "udp":
            return 17
    return 0


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def infer_direction(obj: Dict[str, Any], src_ip: Any, dest_ip: Any) -> str:
    """
    Best-effort direction inference for desktop traffic.

    Returns:
    - outbound
    - inbound
    - lateral
    - unknown
    """
    import ipaddress

    def is_private_ip(ip: Any) -> bool:
        if not ip:
            return False
        try:
            return ipaddress.ip_address(str(ip)).is_private
        except Exception:
            return False

    src_private = is_private_ip(src_ip)
    dest_private = is_private_ip(dest_ip)

    if src_private and not dest_private:
        return "outbound"
    if not src_private and dest_private:
        return "inbound"
    if src_private and dest_private:
        return "lateral"

    return "unknown"


def build_features_from_suricata(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Expects Suricata eve.json entries.
    We focus on flow events, because they contain packet/byte counters and duration.
    """

    MIN_DUR_S = 0.01
    FILTER_MDNS = True

    if obj.get("event_type") != "flow":
        return None

    flow = obj.get("flow") or {}

    src_ip = obj.get("src_ip")
    dest_ip = obj.get("dest_ip")
    sport = obj.get("src_port")
    dport = obj.get("dest_port")
    proto_raw = obj.get("proto")
    app_proto = obj.get("app_proto")
    direction = infer_direction(obj, src_ip, dest_ip)

    proto_num = proto_to_num(proto_raw)

    sport_i = safe_int(sport, 0)
    dport_i = safe_int(dport, 0)

    if FILTER_MDNS:
        if dport_i == 5353 or sport_i == 5353 or dest_ip in ("224.0.0.251", "ff02::fb"):
            return None

    pkts_fwd = safe_int(flow.get("pkts_toserver"), 0)
    pkts_rev = safe_int(flow.get("pkts_toclient"), 0)
    bytes_fwd = safe_int(flow.get("bytes_toserver"), 0)
    bytes_rev = safe_int(flow.get("bytes_toclient"), 0)

    age = flow.get("age", None)
    if isinstance(age, (int, float)) and age > 0:
        duration = float(age)
    else:
        start_ts = _to_unix_seconds(flow.get("start"))
        end_ts = _to_unix_seconds(flow.get("end"))
        if start_ts is not None and end_ts is not None and end_ts >= start_ts:
            duration = float(end_ts - start_ts)
        else:
            duration = MIN_DUR_S

    duration = max(MIN_DUR_S, duration)

    total_pkts = pkts_fwd + pkts_rev
    total_bytes = bytes_fwd + bytes_rev

    pkt_rate = float(total_pkts) / float(duration)
    byte_rate = float(total_bytes) / float(duration)

    fwd_rev_ratio = float(bytes_fwd + 1.0) / float(bytes_rev + 1.0)

    features = {
        "pkts_fwd": float(pkts_fwd),
        "pkts_rev": float(pkts_rev),
        "bytes_fwd": float(bytes_fwd),
        "bytes_rev": float(bytes_rev),
        "duration": float(duration),
        "pkt_rate": float(pkt_rate),
        "byte_rate": float(byte_rate),
        "fwd_rev_ratio": float(fwd_rev_ratio),
        "proto": float(proto_num),
    }

    flow_id = obj.get("flow_id")
    if flow_id is None:
        flow_id = f"{src_ip or '?'}:{sport_i}->{dest_ip or '?'}:{dport_i}"

    ts_unix = _to_unix_seconds(obj.get("timestamp"))

    return {
        "flow_id": str(flow_id),
        "features": features,
        "src_ip": src_ip,
        "src_port": sport_i,
        "dest_ip": dest_ip,
        "dest_port": dport_i,
        "proto": proto_raw,
        "app_proto": app_proto,
        "direction": direction,
        "ts_unix": ts_unix,
    }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        payload = build_features_from_suricata(obj)
        if not payload:
            continue

        try:
            r = requests.post(PREDICT_URL, json=payload, timeout=2.0)
            if r.status_code != 200:
                print("[WARN] /predict failed:", r.status_code, r.text[:200], file=sys.stderr)
        except Exception as e:
            print("[WARN] request error:", e, file=sys.stderr)
            time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())