#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeek_to_csv.py
─────────────────────────────────────────────
Convert Zeek conn.log to a Suricata-compatible features CSV (30 columns).

Usage:
  python3 src/features/zeek_to_csv.py --in logs/zeek/conn.log --out data/zeek_conn_features.csv

Outputs:
  A CSV aligned with Suricata’s flow schema:
  pkts_toserver, bytes_toserver, pkts_toclient, bytes_toclient, state, etc.

Author: Ioana Postelnecu (Bachelor Project)
"""

import argparse
import csv
import math
import os
import sys

# --------------- Schema definition (30 columns) -----------------
SCHEMA = [
    "start_ts","end_ts","duration",
    "src_ip","src_port","dst_ip","dst_port",
    "proto","app_proto","state",
    "pkts_toserver","pkts_toclient",
    "bytes_toserver","bytes_toclient",
    "pkts_total","bytes_total",
    "bpp_toserver","bpp_toclient",
    "pps_toserver","pps_toclient",
    "byte_ratio_ts_tc","pkt_ratio_ts_tc",
    "src2dst_rate","dst2src_rate",
    "is_tcp","is_udp","is_icmp","is_ipv6",
    "zeek_uid","zeek_history",
]

NUMERIC_ZERO = {"-", "", None}


# ---------------- Utility helpers -----------------

def nz_float(x, default=0.0):
    """Safe float conversion, returns default if not valid."""
    if x in NUMERIC_ZERO:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)

def nz_int(x, default=0):
    """Safe int conversion."""
    if x in NUMERIC_ZERO:
        return int(default)
    try:
        return int(float(x))
    except Exception:
        return int(default)

def norm_proto(p):
    p = (p or "").lower()
    if p in ("tcp","udp","icmp"):
        return p
    return "other"

def map_state(conn_state, history):
    cs = (conn_state or "").upper()
    hist = history or ""
    if cs == "S0":
        return "attempted"
    if cs == "REJ":
        return "rejected"
    if cs.startswith("RST") or ("R" in hist):
        return "reset"
    if "D" in hist:
        return "established"
    if cs == "SF":
        return "closed"
    return "other"

def bool01(b): return 1 if b else 0

def parse_fields_line(line):
    # Zeek header line: "#fields\tts\tuid\tid.orig_h\t..."
    parts = line.rstrip("\n").split("\t")
    return {name: idx for idx, name in enumerate(parts)}

def get_field(field_map, fields, name, default=""):
    idx = field_map.get(name, None)
    if idx is None or idx >= len(fields):
        return default
    return fields[idx]


# ---------------- Main converter -----------------

def zeek_to_csv(input_path: str, output_path: str):
    """Convert Zeek conn.log to Suricata-compatible CSV."""
    if not os.path.exists(input_path):
        sys.exit(f"❌ Input file not found: {input_path}")

    field_index = None
    rows_out = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8", errors="replace") as f, \
         open(output_path, "w", newline="", encoding="utf-8") as g:
        writer = csv.DictWriter(g, fieldnames=SCHEMA)
        writer.writeheader()

        for raw in f:
            if not raw or raw.startswith("#"):
                if raw.startswith("#fields"):
                    field_index = parse_fields_line(raw.strip())
                continue
            if field_index is None:
                continue

            fields = raw.rstrip("\n").split("\t")

            ts = nz_float(get_field(field_index, fields, "ts", "0"))
            dur = nz_float(get_field(field_index, fields, "duration", "0"))
            uid = get_field(field_index, fields, "uid", "-")

            sh  = get_field(field_index, fields, "id.orig_h", "")
            sp  = nz_int(get_field(field_index, fields, "id.orig_p", "0"))
            dh  = get_field(field_index, fields, "id.resp_h", "")
            dp  = nz_int(get_field(field_index, fields, "id.resp_p", "0"))

            pr  = norm_proto(get_field(field_index, fields, "proto", ""))
            svc = get_field(field_index, fields, "service", "-") or "-"

            hist = get_field(field_index, fields, "history", "-") or "-"
            cs   = get_field(field_index, fields, "conn_state", "-") or "-"

            ob   = nz_float(get_field(field_index, fields, "orig_bytes", "0"))
            rb   = nz_float(get_field(field_index, fields, "resp_bytes", "0"))
            opk  = nz_float(get_field(field_index, fields, "orig_pkts", "0"))
            rpk  = nz_float(get_field(field_index, fields, "resp_pkts", "0"))

            # Derived features
            end_ts = ts + (dur if dur > 0 else 0.0)
            pkts_total  = opk + rpk
            bytes_total = ob + rb

            denom_opk = opk if opk > 0 else 1.0
            denom_rpk = rpk if rpk > 0 else 1.0
            denom_dur = dur if dur > 0 else 1e-9

            bpp_ts = ob / denom_opk
            bpp_tc = rb / denom_rpk
            pps_ts = opk / denom_dur
            pps_tc = rpk / denom_dur
            byte_ratio = ob / (rb if rb > 0 else 1.0)
            pkt_ratio  = opk / (rpk if rpk > 0 else 1.0)
            s2d_rate = ob / denom_dur
            d2s_rate = rb / denom_dur

            is_tcp  = bool01(pr == "tcp")
            is_udp  = bool01(pr == "udp")
            is_icmp = bool01(pr == "icmp")
            is_ipv6 = bool01(":" in sh or ":" in dh)

            state = map_state(cs, hist)

            row = {
                "start_ts": ts,
                "end_ts": end_ts,
                "duration": dur,
                "src_ip": sh,
                "src_port": int(sp),
                "dst_ip": dh,
                "dst_port": int(dp),
                "proto": pr,
                "app_proto": svc,
                "state": state,
                "pkts_toserver": int(opk),
                "pkts_toclient": int(rpk),
                "bytes_toserver": int(ob),
                "bytes_toclient": int(rb),
                "pkts_total": int(pkts_total),
                "bytes_total": int(bytes_total),
                "bpp_toserver": bpp_ts,
                "bpp_toclient": bpp_tc,
                "pps_toserver": pps_ts,
                "pps_toclient": pps_tc,
                "byte_ratio_ts_tc": byte_ratio,
                "pkt_ratio_ts_tc": pkt_ratio,
                "src2dst_rate": s2d_rate,
                "dst2src_rate": d2s_rate,
                "is_tcp": is_tcp,
                "is_udp": is_udp,
                "is_icmp": is_icmp,
                "is_ipv6": is_ipv6,
                "zeek_uid": uid,
                "zeek_history": hist,
            }
            writer.writerow(row)
            rows_out += 1

    print(f"✅ Wrote {rows_out} Zeek flows to {output_path}")


# ---------------- CLI Entry Point -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Zeek conn.log to Suricata-compatible features CSV."
    )
    parser.add_argument("--in", dest="inp", required=True, help="Input Zeek conn.log")
    parser.add_argument("--out", dest="out", required=True, help="Output CSV file")
    args = parser.parse_args()

    zeek_to_csv(args.inp, args.out)
