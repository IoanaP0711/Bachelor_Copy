#!/usr/bin/env python3
from __future__ import annotations

import json
import ipaddress
import socket
import threading
import psutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"
IDS_API_KEY = os.getenv("IDS_API_KEY", "")
IDS_MODE = os.getenv(
    "IDS_MODE",
    "live",
).strip().lower()

PROCESS_CACHE_TTL_SECONDS = 180
PROCESS_POLL_INTERVAL_SECONDS = 0.5

_PROCESS_CACHE_LOCK = threading.Lock()
_PROCESS_CACHE: Dict[tuple, Dict[str, Any]] = {}
_PROCESS_WATCHER_STARTED = False

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


def normalize_socket_ip(value: Any) -> str:
    """
    Normalize IPv4 and IPv6 addresses for socket matching.
    """
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    try:
        return ipaddress.ip_address(text).compressed
    except ValueError:
        return text.lower()


def read_socket_address(address: Any) -> tuple[str, int]:
    """
    Read an address returned by psutil.
    """
    if not address:
        return "", 0

    if hasattr(address, "ip"):
        ip_value = address.ip
        port_value = address.port
    else:
        ip_value = address[0]
        port_value = address[1]

    try:
        port = int(port_value)
    except (TypeError, ValueError):
        port = 0

    return normalize_socket_ip(ip_value), port


def socket_protocol(socket_type: int) -> Optional[str]:
    if socket_type == socket.SOCK_STREAM:
        return "TCP"

    if socket_type == socket.SOCK_DGRAM:
        return "UDP"

    return None


def collect_process_connections() -> None:
    """
    Store active local socket connections together with their process.

    The cache remains available for a short period because Suricata may
    generate the final flow event after the socket has already closed.
    """
    now = time.time()

    current_connections: Dict[
        tuple,
        Dict[str, Any],
    ] = {}

    process_information_by_pid: Dict[
        int,
        Dict[str, Any],
    ] = {}

    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.Error, OSError):
        return

    for connection in connections:
        protocol = socket_protocol(connection.type)

        if protocol is None:
            continue

        if connection.pid is None:
            continue

        local_ip, local_port = read_socket_address(
            connection.laddr
        )

        remote_ip, remote_port = read_socket_address(
            connection.raddr
        )

        if not local_ip or not local_port:
            continue

        # Exact remote endpoint matching is used.
        # Unconnected UDP sockets cannot be attributed reliably.
        if not remote_ip or not remote_port:
            continue

        pid = int(connection.pid)

        if pid not in process_information_by_pid:
            process_name = None
            process_executable = None

            try:
                process = psutil.Process(pid)

                process_name = process.name()

                try:
                    process_executable = process.exe()
                except (
                    psutil.AccessDenied,
                    psutil.NoSuchProcess,
                    psutil.ZombieProcess,
                ):
                    process_executable = None

            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
            ):
                process_name = None
                process_executable = None

            process_information_by_pid[pid] = {
                "process_name": process_name,
                "process_pid": pid,
                "process_exe": process_executable,
            }

        cache_key = (
            protocol,
            local_ip,
            local_port,
            remote_ip,
            remote_port,
        )

        current_connections[cache_key] = {
            **process_information_by_pid[pid],
            "seen_at": now,
        }

    with _PROCESS_CACHE_LOCK:
        _PROCESS_CACHE.update(current_connections)

        expired_keys = [
            key
            for key, value in _PROCESS_CACHE.items()
            if (
                now - float(value.get("seen_at", 0))
                > PROCESS_CACHE_TTL_SECONDS
            )
        ]

        for key in expired_keys:
            _PROCESS_CACHE.pop(key, None)


def process_connection_watcher() -> None:
    while True:
        collect_process_connections()
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)


def start_process_connection_watcher() -> None:
    global _PROCESS_WATCHER_STARTED

    if _PROCESS_WATCHER_STARTED:
        return

    watcher = threading.Thread(
        target=process_connection_watcher,
        name="process-connection-watcher",
        daemon=True,
    )

    watcher.start()

    _PROCESS_WATCHER_STARTED = True


def lookup_process_for_flow(
    src_ip: Any,
    src_port: int,
    dest_ip: Any,
    dest_port: int,
    protocol: Any,
) -> Dict[str, Any]:
    """
    Match a Suricata flow against a cached local socket.
    """
    protocol_name = str(
        protocol or ""
    ).strip().upper()

    normalized_src_ip = normalize_socket_ip(src_ip)
    normalized_dest_ip = normalize_socket_ip(dest_ip)

    source_key = (
        protocol_name,
        normalized_src_ip,
        int(src_port),
        normalized_dest_ip,
        int(dest_port),
    )

    destination_key = (
        protocol_name,
        normalized_dest_ip,
        int(dest_port),
        normalized_src_ip,
        int(src_port),
    )

    with _PROCESS_CACHE_LOCK:
        source_match = _PROCESS_CACHE.get(source_key)
        destination_match = _PROCESS_CACHE.get(
            destination_key
        )

    if source_match:
        return {
            "process_name": source_match.get(
                "process_name"
            ),
            "process_pid": source_match.get(
                "process_pid"
            ),
            "process_exe": source_match.get(
                "process_exe"
            ),
            "process_attribution": (
                "exact_source_socket_match"
            ),
            "process_attribution_confidence": "high",
        }

    if destination_match:
        return {
            "process_name": destination_match.get(
                "process_name"
            ),
            "process_pid": destination_match.get(
                "process_pid"
            ),
            "process_exe": destination_match.get(
                "process_exe"
            ),
            "process_attribution": (
                "exact_destination_socket_match"
            ),
            "process_attribution_confidence": "high",
        }

    return {
        "process_name": None,
        "process_pid": None,
        "process_exe": None,
        "process_attribution": "not_found",
        "process_attribution_confidence": "none",
    }

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

    if pkts_rev > 0:
        fwd_rev_ratio = float(pkts_fwd) / float(pkts_rev)
    else:
        fwd_rev_ratio = float(pkts_fwd)

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

    process_information = lookup_process_for_flow(
        src_ip=src_ip,
        src_port=sport_i,
        dest_ip=dest_ip,
        dest_port=dport_i,
        protocol=proto_raw,
    )

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

        "process_name": process_information.get(
            "process_name"
        ),
        "process_pid": process_information.get(
            "process_pid"
        ),
        "process_exe": process_information.get(
            "process_exe"
        ),
        "process_attribution": process_information.get(
            "process_attribution"
        ),
        "process_attribution_confidence": (
            process_information.get(
                "process_attribution_confidence"
            )
        ),
    }


def main() -> int:

    if IDS_MODE == "replay":
        print(
            "[runner] IDS_MODE=replay; live Suricata input is disabled.",
            file=sys.stderr,
        )
        return 0

    start_process_connection_watcher()

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
            r = requests.post(
                PREDICT_URL,
                json=payload,
                headers={
                    "X-API-Key": IDS_API_KEY,
                },
                timeout=2.0,
            )
            if r.status_code != 200:
                print("[WARN] /predict failed:", r.status_code, r.text[:200], file=sys.stderr)
        except Exception as e:
            print("[WARN] request error:", e, file=sys.stderr)
            time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())