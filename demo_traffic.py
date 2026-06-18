#!/usr/bin/env python3
"""
Controlled traffic generator for a local IDS demonstration.

Safety:
- Active tests are restricted to a private IPv4 target such as your own router
  or another device you control.
- The port sweep is intentionally small and slow.
- No exploit payloads are sent.

Examples:
    python3 demo_traffic.py --list
    python3 demo_traffic.py --case 1
    python3 demo_traffic.py --case 8 --target 192.168.1.1
    python3 demo_traffic.py --all --pause 10
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DemoCase:
    number: int
    name: str
    expected: str
    function: Callable[[str], None]


def default_gateway() -> str:
    try:
        result = subprocess.run(
            ["ip", "-4", "route", "show", "default"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not determine the default gateway.") from exc

    for line in result.stdout.splitlines():
        parts = line.split()
        if "via" in parts:
            candidate = parts[parts.index("via") + 1]
            try:
                ipaddress.ip_address(candidate)
                return candidate
            except ValueError:
                continue

    raise RuntimeError("No IPv4 default gateway was found.")


def require_private_ipv4(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid IP address: {value}") from exc

    if address.version != 4 or not address.is_private:
        raise argparse.ArgumentTypeError(
            "The active-test target must be a private IPv4 address that you control."
        )
    return str(address)


def show_result(message: str) -> None:
    print(f"    Result: {message}")


def normal_dns(_: str) -> None:
    addresses = socket.getaddrinfo("example.com", 443, type=socket.SOCK_STREAM)
    unique = sorted({item[4][0] for item in addresses})
    show_result(f"Resolved example.com to {', '.join(unique[:4])}")


def normal_https(_: str) -> None:
    request = urllib.request.Request(
        "https://example.com/",
        headers={"User-Agent": "Bachelor-IDS-Controlled-Demo/1.0"},
        method="GET",
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=8, context=context) as response:
        response.read(256)
        show_result(f"HTTPS status {response.status}; read 256 bytes")


def ping_gateway(target: str) -> None:
    result = subprocess.run(
        ["ping", "-c", "4", "-W", "1", target],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        show_result(f"Received ICMP replies from {target}")
    else:
        show_result(
            f"Ping completed without replies. Traffic was still generated toward {target}."
        )


def tcp_web_attempt(target: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        code = sock.connect_ex((target, 80))
    show_result(f"TCP connect_ex({target}:80) returned {code}")


def ssdp_multicast(_: str) -> None:
    payload = (
        'M-SEARCH * HTTP/1.1\r\n'
        'HOST: 239.255.255.250:1900\r\n'
        'MAN: "ssdp:discover"\r\n'
        'MX: 1\r\n'
        'ST: ssdp:all\r\n'
        '\r\n'
    ).encode("ascii")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(1.5)
        sock.sendto(payload, ("239.255.255.250", 1900))
        replies = 0
        end = time.time() + 1.5
        while time.time() < end:
            try:
                sock.recvfrom(2048)
                replies += 1
            except socket.timeout:
                break
    show_result(f"Sent one SSDP discovery request; received {replies} replies")


def single_udp_high_port(target: str) -> None:
    payload = b"BACHELOR_IDS_DEMO_SINGLE_UDP"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (target, 65001))
    show_result(f"Sent one UDP datagram to {target}:65001")


def single_tcp_high_port(target: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        code = sock.connect_ex((target, 65000))
    show_result(f"One TCP attempt to {target}:65000 returned {code}")


def repeated_tcp_high_port(target: str) -> None:
    attempts = 8
    results: list[int] = []
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.8)
            results.append(sock.connect_ex((target, 65000)))
        time.sleep(0.25)
    show_result(
        f"Completed {attempts} TCP attempts to the same destination/port; "
        f"return codes={results}"
    )


def small_port_sweep(target: str) -> None:
    ports = [21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445, 3389]
    results: list[str] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.7)
            code = sock.connect_ex((target, port))
        results.append(f"{port}:{'open' if code == 0 else 'closed/filtered'}")
        time.sleep(0.20)
    show_result("Small controlled sweep completed: " + ", ".join(results))


def repeated_dns(_: str) -> None:
    queries = 8
    completed = 0
    for index in range(queries):
        hostname = f"demo-{int(time.time())}-{index}.example.com"
        try:
            socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        except socket.gaierror:
            pass
        completed += 1
        time.sleep(0.25)
    show_result(f"Completed {completed} distinct DNS lookups through the system resolver")


CASES = [
    DemoCase(
        1,
        "Normal DNS lookup",
        "Usually OK or BENIGN. It proves ordinary name-resolution traffic is not treated as a serious incident.",
        normal_dns,
    ),
    DemoCase(
        2,
        "Normal HTTPS request",
        "Usually OK or BENIGN. Destination port 443 and common web context should be recognized as ordinary traffic.",
        normal_https,
    ),
    DemoCase(
        3,
        "ICMP ping to the local gateway",
        "Usually OK/BENIGN, or it may only appear in Suricata if the model pipeline does not transform ICMP flows.",
        ping_gateway,
    ),
    DemoCase(
        4,
        "Single TCP connection attempt to the router web port",
        "Usually OK/BENIGN if port 80 is a normal local service; otherwise a low-confidence REVIEW is acceptable.",
        tcp_web_attempt,
    ),
    DemoCase(
        5,
        "SSDP multicast discovery",
        "Expected BENIGN or low-priority output because 239.255.255.250:1900 is local multicast/service-discovery traffic.",
        ssdp_multicast,
    ),
    DemoCase(
        6,
        "Single UDP datagram to an unusual high port",
        "Expected REVIEW or a higher raw anomaly score, but not necessarily CRITICAL after only one occurrence.",
        single_udp_high_port,
    ),
    DemoCase(
        7,
        "Single TCP attempt to an unusual high port",
        "Expected REVIEW or elevated raw severity. This is the baseline for comparison with case 8.",
        single_tcp_high_port,
    ),
    DemoCase(
        8,
        "Repeated TCP attempts to the same unusual high port",
        "Expected repeat count increase and possible escalation from REVIEW to CRITICAL after the configured threshold.",
        repeated_tcp_high_port,
    ),
    DemoCase(
        9,
        "Small controlled TCP port sweep of the local gateway",
        "Expected REVIEW/CRITICAL. A Suricata scan alert may also appear when the relevant local rule is enabled.",
        small_port_sweep,
    ),
    DemoCase(
        10,
        "Repeated DNS lookups",
        "Expected a repeated pattern, but usually BENIGN/REVIEW rather than CRITICAL because DNS is a known service context.",
        repeated_dns,
    ),
]


def print_cases() -> None:
    for case in CASES:
        print(f"{case.number:>2}. {case.name}")
        print(f"    Expected: {case.expected}")


def run_case(case: DemoCase, target: str) -> None:
    print()
    print("=" * 78)
    print(f"CASE {case.number}: {case.name}")
    print(f"Target: {target}")
    print(f"Expected dashboard behavior: {case.expected}")
    print(f"Start timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        case.function(target)
    except Exception as exc:
        print(f"    Error: {type(exc).__name__}: {exc}")
    print(f"End timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=require_private_ipv4,
        help="Private IPv4 address of your own router or another device you control. "
             "Default: detected IPv4 gateway.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", type=int, choices=range(1, 11))
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    parser.add_argument(
        "--pause",
        type=float,
        default=10.0,
        help="Seconds between cases when --all is used. Default: 10.",
    )
    args = parser.parse_args()

    if args.list:
        print_cases()
        return 0

    try:
        target = args.target or default_gateway()
        target = require_private_ipv4(target)
    except (RuntimeError, argparse.ArgumentTypeError) as exc:
        print(f"Target error: {exc}", file=sys.stderr)
        return 2

    if args.case:
        run_case(CASES[args.case - 1], target)
        return 0

    if args.all:
        for index, case in enumerate(CASES):
            run_case(case, target)
            if index != len(CASES) - 1:
                time.sleep(max(args.pause, 0))
        return 0

    print_cases()
    print()
    print("Run one case with: python3 demo_traffic.py --case NUMBER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
