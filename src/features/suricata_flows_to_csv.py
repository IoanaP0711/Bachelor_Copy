#!/usr/bin/env python3
# src/features/suricata_flows_to_csv.py
"""
Read Suricata eve.json (flow events), extract a reproducible 30-field feature vector, and write CSV.

Features (columns), in order:
 1  timestamp                     (str, ISO8601 from event["timestamp"])
 2  flow_id                       (int)
 3  community_id                  (str)
 4  src_ip                        (str)
 5  src_port                      (int)
 6  dest_ip                       (str)
 7  dest_port                     (int)
 8  proto                         (str; e.g., "TCP"/"UDP"/"ICMP"/etc.)
 9  app_proto                     (str; e.g., "http", "dns", ...)
10  flow_start                    (str, ISO8601 from event["flow"]["start"])
11  flow_end                      (str, ISO8601 from event["flow"]["end"])
12  flow_age                      (float seconds; event["flow"]["age"])
13  flow_state                    (str; event["flow"]["state"])
14  flow_reason                   (str; event["flow"]["reason"])
15  pkts_toserver                 (int)
16  pkts_toclient                 (int)
17  bytes_toserver                (int)
18  bytes_toclient                (int)
19  pkts_total                    (int) = 15 + 16
20  bytes_total                   (int) = 17 + 18
21  duration                      (float seconds; parsed from start/end, fallback to flow_age, else 0)
22  bps                           (float; bytes_total / max(duration,1e-9))
23  pps                           (float; pkts_total  / max(duration,1e-9))
24  bytes_ratio_ts_over_tc        (float; bytes_toserver / max(bytes_toclient,1))
25  pkts_ratio_ts_over_tc         (float; pkts_toserver  / max(pkts_toclient,1))
26  avg_bytes_per_pkt_ts          (float; bytes_toserver / max(pkts_toserver,1))
27  avg_bytes_per_pkt_tc          (float; bytes_toclient / max(pkts_toclient,1))
28  is_tcp                        (0/1)
29  is_udp                        (0/1)
30  is_icmp                       (0/1)

Robustness:
- Ignores non-JSON and non-flow lines.
- Missing keys → sensible defaults (0, "", or computed fallbacks).
- Supports reading a single file or a glob (e.g., /var/log/suricata/eve.json*). Handles .gz as well.
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import sys
from datetime import datetime
from glob import glob
from typing import Iterable, Dict, Any, Tuple, Optional

# ---------- Utilities ----------

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO8601 with possible Z; return None if missing/invalid."""
    if not s or not isinstance(s, str):
        return None
    try:
        # Suricata uses e.g. "2025-10-18T01:23:45.678901+0000" sometimes; normalize a few variants
        s2 = s.replace("Z", "+00:00")
        # If timezone like +0000 (no colon), insert colon
        if len(s2) > 5 and (s2[-5] in ['+', '-']) and s2[-3] != ':':
            s2 = s2[:-2] + ":" + s2[-2:]
        return datetime.fromisoformat(s2)
    except Exception:
        return None

def _duration_seconds(start_iso: Optional[str], end_iso: Optional[str], fallback_age: float) -> float:
    dt_start = _iso_to_dt(start_iso)
    dt_end   = _iso_to_dt(end_iso)
    if dt_start and dt_end:
        delta = (dt_end - dt_start).total_seconds()
        # Guard against negative or NaN
        if isinstance(delta, (int, float)) and delta >= 0:
            return float(delta)
    # Fallback to age if valid
    if isinstance(fallback_age, (int, float)) and fallback_age >= 0:
        return float(fallback_age)
    return 0.0

def _div(n: float, d: float) -> float:
    d = d if d not in (None, 0.0) else 0.0
    if d == 0.0:
        return float('inf') if n and n != 0 else 0.0
    return float(n) / float(d)

def _iter_lines(paths: Iterable[str]) -> Iterable[str]:
    """Yield lines from one or more files; handle .gz transparently."""
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            if p.endswith(".gz"):
                with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        yield line
            else:
                with io.open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        yield line
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}", file=sys.stderr)
            continue

# ---------- Core extraction ----------

FEATURE_HEADERS = [
    "timestamp",
    "flow_id",
    "community_id",
    "src_ip",
    "src_port",
    "dest_ip",
    "dest_port",
    "proto",
    "app_proto",
    "flow_start",
    "flow_end",
    "flow_age",
    "flow_state",
    "flow_reason",
    "pkts_toserver",
    "pkts_toclient",
    "bytes_toserver",
    "bytes_toclient",
    "pkts_total",
    "bytes_total",
    "duration",
    "bps",
    "pps",
    "bytes_ratio_ts_over_tc",
    "pkts_ratio_ts_over_tc",
    "avg_bytes_per_pkt_ts",
    "avg_bytes_per_pkt_tc",
    "is_tcp",
    "is_udp",
    "is_icmp",
]

def _event_to_row(ev: Dict[str, Any]) -> Dict[str, Any]:
    # Top-level
    ts            = ev.get("timestamp", "")
    flow_id       = _safe_int(ev.get("flow_id", 0))
    community_id  = ev.get("community_id", "") or ""
    src_ip        = ev.get("src_ip", "") or ""
    src_port      = _safe_int(ev.get("src_port", 0))
    dest_ip       = ev.get("dest_ip", "") or ""
    dest_port     = _safe_int(ev.get("dest_port", 0))

    # Normalize proto to string like TCP/UDP/ICMP if numeric
    proto_val     = ev.get("proto", "")
    if isinstance(proto_val, int):
        proto = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto_val, str(proto_val))
    else:
        proto = str(proto_val or "")

    app_proto     = ev.get("app_proto", "") or ""

    # Flow block (all optional)
    flow          = ev.get("flow", {}) or {}
    flow_start    = flow.get("start", "") or ""
    flow_end      = flow.get("end", "") or ""
    flow_age      = _safe_float(flow.get("age", 0.0))
    flow_state    = flow.get("state", "") or ""
    flow_reason   = flow.get("reason", "") or ""

    pkts_ts       = _safe_int(flow.get("pkts_toserver", 0))
    pkts_tc       = _safe_int(flow.get("pkts_toclient", 0))
    bytes_ts      = _safe_int(flow.get("bytes_toserver", 0))
    bytes_tc      = _safe_int(flow.get("bytes_toclient", 0))

    pkts_total    = pkts_ts + pkts_tc
    bytes_total   = bytes_ts + bytes_tc

    duration      = _duration_seconds(flow_start, flow_end, flow_age)

    # Derived metrics (safe division)
    bps           = _div(bytes_total, duration if duration > 0 else 1e-9)
    pps           = _div(pkts_total,  duration if duration > 0 else 1e-9)
    bytes_ratio   = _div(bytes_ts, max(bytes_tc, 1))
    pkts_ratio    = _div(pkts_ts,   max(pkts_tc, 1))
    avg_bpp_ts    = _div(bytes_ts,  max(pkts_ts, 1))
    avg_bpp_tc    = _div(bytes_tc,  max(pkts_tc, 1))

    proto_upper   = (proto or "").upper()
    is_tcp        = 1 if "TCP"  == proto_upper else 0
    is_udp        = 1 if "UDP"  == proto_upper else 0
    is_icmp       = 1 if "ICMP" == proto_upper else 0

    row = {
        "timestamp": ts,
        "flow_id": flow_id,
        "community_id": community_id,
        "src_ip": src_ip,
        "src_port": src_port,
        "dest_ip": dest_ip,
        "dest_port": dest_port,
        "proto": proto_upper,
        "app_proto": app_proto,
        "flow_start": flow_start,
        "flow_end": flow_end,
        "flow_age": round(flow_age, 6),
        "flow_state": flow_state,
        "flow_reason": flow_reason,
        "pkts_toserver": pkts_ts,
        "pkts_toclient": pkts_tc,
        "bytes_toserver": bytes_ts,
        "bytes_toclient": bytes_tc,
        "pkts_total": pkts_total,
        "bytes_total": bytes_total,
        "duration": round(duration, 9),
        "bps": round(bps, 6) if math.isfinite(bps) else 0.0,
        "pps": round(pps, 6) if math.isfinite(pps) else 0.0,
        "bytes_ratio_ts_over_tc": round(bytes_ratio, 6) if math.isfinite(bytes_ratio) else 0.0,
        "pkts_ratio_ts_over_tc": round(pkts_ratio, 6) if math.isfinite(pkts_ratio) else 0.0,
        "avg_bytes_per_pkt_ts": round(avg_bpp_ts, 6) if math.isfinite(avg_bpp_ts) else 0.0,
        "avg_bytes_per_pkt_tc": round(avg_bpp_tc, 6) if math.isfinite(avg_bpp_tc) else 0.0,
        "is_tcp": is_tcp,
        "is_udp": is_udp,
        "is_icmp": is_icmp,
    }
    return row

# ---------- Main ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Suricata flow features from eve.json into CSV (30 fields)."
    )
    p.add_argument(
        "-i", "--input",
        default="/var/log/suricata/eve.json*",
        help="Input file or glob (supports .gz). Default: /var/log/suricata/eve.json*"
    )
    p.add_argument(
        "-o", "--output",
        default="data/processed/suricata_flows.csv",
        help="Output CSV path. Default: data/processed/suricata_flows.csv"
    )
    return p.parse_args()

def main():
    args = parse_args()
    paths = sorted(glob(args.input)) if any(ch in args.input for ch in "*?[]") else [args.input]

    if not paths:
        print(f"[ERROR] No input files match: {args.input}", file=sys.stderr)
        sys.exit(2)

    # Ensure output dir
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    count_in = 0
    count_out = 0
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FEATURE_HEADERS)
        writer.writeheader()

        for line in _iter_lines(paths):
            count_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                # Skip non-JSON or partial lines
                continue

            if ev.get("event_type") != "flow":
                continue

            try:
                row = _event_to_row(ev)
                writer.writerow(row)
                count_out += 1
            except Exception as e:
                # Be resilient: log and continue
                print(f"[WARN] Failed to convert flow event at input line {count_in}: {e}", file=sys.stderr)
                continue

    print(f"[OK] Wrote {count_out} flow rows to {args.output}")

if __name__ == "__main__":
    main()
