from __future__ import annotations

import ipaddress
from typing import Any, Dict, List, Optional, Set


# ============================================================
# Safe helpers
# ============================================================

def _safe_lower(value: Any) -> str:
    """Return a lowercase stripped string, or empty string on failure."""
    if value is None:
        return ""
    try:
        return str(value).strip().lower()
    except Exception:
        return ""


def _safe_int(value: Any) -> Optional[int]:
    """Convert to int if possible, otherwise return None."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(alert: Dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty value found among candidate keys."""
    for key in keys:
        value = alert.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_port(alert: Dict[str, Any], canonical_key: str, alias_key: str) -> Optional[int]:
    """Read ports safely from either canonical or alias field names."""
    return _safe_int(_first_present(alert, canonical_key, alias_key))


def _port_set(src_port: Optional[int], dest_port: Optional[int]) -> Set[int]:
    return {p for p in (src_port, dest_port) if p is not None}


def _parse_ip(ip: Any):
    if not ip:
        return None
    try:
        return ipaddress.ip_address(str(ip))
    except Exception:
        return None


def _is_private_ip(ip: Any) -> bool:
    parsed = _parse_ip(ip)
    return bool(parsed and parsed.is_private)


def _is_multicast_ip(ip: Any) -> bool:
    parsed = _parse_ip(ip)
    return bool(parsed and parsed.is_multicast)


def _is_loopback_ip(ip: Any) -> bool:
    parsed = _parse_ip(ip)
    return bool(parsed and parsed.is_loopback)


def _is_link_local_ip(ip: Any) -> bool:
    parsed = _parse_ip(ip)
    return bool(parsed and parsed.is_link_local)


def _is_unspecified_ip(ip: Any) -> bool:
    parsed = _parse_ip(ip)
    return bool(parsed and parsed.is_unspecified)


def _is_localish_ip(ip: Any) -> bool:
    """
    Treat private, loopback, and link-local as localish desktop/home-network IPs.
    """
    return (
        _is_private_ip(ip)
        or _is_loopback_ip(ip)
        or _is_link_local_ip(ip)
    )


def _is_ipv4_broadcast(ip: Any) -> bool:
    """
    Detect limited broadcast. This is common for LAN discovery/service traffic.
    """
    try:
        return str(ip).strip() == "255.255.255.255"
    except Exception:
        return False


def _normalize_direction(direction: str, src_ip: Any, dest_ip: Any) -> str:
    """
    Best-effort direction normalization.
    """
    direction = _safe_lower(direction)
    if direction in {"outbound", "egress"}:
        return "outbound"
    if direction in {"inbound", "ingress"}:
        return "inbound"
    if direction in {"internal", "local", "lan", "lateral"}:
        return "internal"

    src_local = _is_localish_ip(src_ip)
    dest_local = _is_localish_ip(dest_ip)

    if src_local and not dest_local:
        return "outbound"
    if not src_local and dest_local:
        return "inbound"
    if src_local and dest_local:
        return "internal"
    return ""


# ============================================================
# Main traffic classification
# ============================================================

def classify_traffic(alert: Dict[str, Any]) -> str:
    """
    Classify runtime traffic into a small explainable class set.

    Returns one of:
    - dns
    - web_http
    - web_https
    - local_discovery
    - time_sync
    - streaming
    - chat_messaging
    - software_update
    - remote_access
    - background_service
    - unknown
    """
    proto = _safe_lower(alert.get("proto"))
    app_proto = _safe_lower(alert.get("app_proto"))
    src_ip = _first_present(alert, "src_ip")
    dest_ip = _first_present(alert, "dest_ip")
    src_port = _extract_port(alert, "src_port", "sport")
    dest_port = _extract_port(alert, "dest_port", "dport")
    direction = _normalize_direction(_safe_lower(alert.get("direction")), src_ip, dest_ip)

    ports = _port_set(src_port, dest_port)

    # DNS
    if app_proto == "dns" or 53 in ports:
        return "dns"

    # NTP / time sync
    if app_proto == "ntp" or 123 in ports:
        return "time_sync"

    # ICMP / ICMPv6 / IPv6-ICMP
    # Treat as local discovery/control traffic by default on desktop/home networks.
    if "icmp" in proto:
        return "local_discovery"

    # Local discovery / LAN chatter
    if app_proto in {"mdns", "ssdp", "dhcp", "llmnr", "nbns", "snmp"}:
        return "local_discovery"

    local_discovery_ports = {
        67, 68,      # DHCP
        137, 138,    # NetBIOS
        161,         # SNMP
        1900,        # SSDP
        5353,        # mDNS
        5355,        # LLMNR
    }

    if any(p in local_discovery_ports for p in ports):
        return "local_discovery"

    if _is_multicast_ip(dest_ip) or _is_ipv4_broadcast(dest_ip):
        return "local_discovery"

    if (
        proto == "udp"
        and _is_localish_ip(src_ip)
        and _is_localish_ip(dest_ip)
        and app_proto in {"", "unknown", "failed"}
    ):
        return "local_discovery"

    # Remote access
    if app_proto in {"ssh", "rdp"} or any(p in {22, 3389, 5900, 5938} for p in ports):
        return "remote_access"

    # Plain HTTP
    if app_proto == "http" or dest_port == 80:
        return "web_http"

    # HTTPS / TLS / QUIC
    if app_proto in {"tls", "ssl", "https", "http2", "quic"}:
        return "web_https"

    if dest_port in {443, 8443}:
        return "web_https"

    if proto == "udp" and dest_port == 443:
        return "web_https"

    # Messaging / calls
    if app_proto in {"stun", "turn", "xmpp"} or any(p in {3478, 3479, 3480, 5222, 5223, 5349} for p in ports):
        return "chat_messaging"

    # Streaming / media / nearby-device media behavior
    if app_proto in {"rtsp", "rtmp"} or any(p in {554, 1935} for p in ports):
        return "streaming"

    # Internal/local UDP media-ish behavior:
    # this can include local audio/media/casting/nearby-device traffic patterns
    if (
        proto == "udp"
        and direction in {"internal", "outbound"}
        and _is_localish_ip(src_ip)
        and (
            _is_localish_ip(dest_ip)
            or _is_multicast_ip(dest_ip)
            or _is_ipv4_broadcast(dest_ip)
        )
        and app_proto in {"", "unknown", "failed"}
    ):
        return "streaming"

    # Software update / background service
    if (
        proto == "tcp"
        and direction == "outbound"
        and dest_port in {443, 8443}
        and app_proto in {"", "unknown", "failed"}
    ):
        return "software_update"

    if (
        proto in {"tcp", "udp"}
        and dest_port is not None
        and dest_port >= 1024
        and direction in {"outbound", "internal"}
    ):
        return "background_service"

    return "unknown"


def explain_traffic_class(traffic_class: str, severity: str, is_anom: bool) -> str:
    """
    Return a short human-readable explanation for the runtime traffic class.
    This is used by server.py for the dashboard note.
    """
    traffic_class = _safe_lower(traffic_class)
    severity = _safe_lower(severity)

    if traffic_class == "dns":
        return (
            "DNS name-resolution traffic. Normally expected on laptops and desktops."
            if not is_anom
            else "DNS traffic with an unusual pattern compared to learned normal behavior."
        )

    if traffic_class == "web_http":
        return (
            "Plain HTTP web traffic."
            if not is_anom
            else "HTTP web traffic that looks unusual compared to normal behavior."
        )

    if traffic_class == "web_https":
        return (
            "Encrypted web or API traffic over HTTPS/TLS/QUIC."
            if not is_anom
            else "Encrypted web or app traffic that deviates from learned normal behavior."
        )

    if traffic_class == "local_discovery":
        return (
            "Local network discovery or LAN service traffic such as mDNS, SSDP, DHCP, ICMPv6, broadcast, multicast, or nearby-device discovery."
            if not is_anom
            else "Local discovery or LAN chatter that looks unusual in volume or pattern."
        )

    if traffic_class == "time_sync":
        return (
            "System time synchronization traffic."
            if not is_anom
            else "Time synchronization traffic with unusual timing or rate."
        )

    if traffic_class == "streaming":
        return (
            "Likely media, real-time audio/video, casting, or nearby-device communication traffic."
            if not is_anom
            else "Streaming, media, or nearby-device traffic with an unusual pattern."
        )

    if traffic_class == "chat_messaging":
        return (
            "Likely messaging, voice, video call, or chat support traffic."
            if not is_anom
            else "Chat or calling-related traffic that looks unusual."
        )

    if traffic_class == "software_update":
        return (
            "Likely software update or package retrieval traffic."
            if not is_anom
            else "Update-related traffic that deviates from learned normal behavior."
        )

    if traffic_class == "remote_access":
        return (
            "Remote access traffic such as SSH, RDP, or VNC."
            if not is_anom
            else "Remote access traffic with an unusual pattern."
        )

    if traffic_class == "background_service":
        return (
            "Likely normal background application or system service traffic."
            if not is_anom
            else "Background service traffic that looks unusual."
        )

    if severity == "crit":
        return "Unknown traffic class with high anomaly severity; manual inspection is recommended."
    if severity in {"warn", "med"}:
        return "Unknown traffic class with moderate anomaly severity; review may be needed."

    return "Unknown traffic class."


# ============================================================
# Benign background filtering
# ============================================================

def benign_background_context(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runtime-only contextual filter for noisy but often non-malicious traffic.

    Returns:
    {
        "traffic_class": str,
        "likely_benign": bool,
        "benign_reason": str,
        "context_tags": [str, ...],
    }

    Design principles:
    - Conservative, rule-based, explainable.
    - Use metadata only.
    - Prefer downgrade rather than blanket suppression for DNS and HTTPS.
    - Safe fallback for missing data.
    """
    proto = _safe_lower(alert.get("proto"))
    app_proto = _safe_lower(alert.get("app_proto"))
    src_ip = _first_present(alert, "src_ip")
    dest_ip = _first_present(alert, "dest_ip")
    src_port = _extract_port(alert, "src_port", "sport")
    dest_port = _extract_port(alert, "dest_port", "dport")
    direction = _normalize_direction(_safe_lower(alert.get("direction")), src_ip, dest_ip)

    traffic_class = classify_traffic(alert)
    tags: List[str] = []

    src_private = _is_private_ip(src_ip)
    dest_private = _is_private_ip(dest_ip)
    src_localish = _is_localish_ip(src_ip)
    dest_localish = _is_localish_ip(dest_ip)
    src_loopback = _is_loopback_ip(src_ip)
    dest_loopback = _is_loopback_ip(dest_ip)
    dest_multicast = _is_multicast_ip(dest_ip)
    dest_broadcast = _is_ipv4_broadcast(dest_ip)
    dest_link_local = _is_link_local_ip(dest_ip)

    ports = _port_set(src_port, dest_port)

    # -------------------------
    # Context tags first
    # -------------------------
    if proto:
        tags.append(proto)

    if app_proto and app_proto not in {"unknown", "failed"}:
        tags.append(app_proto)

    if direction:
        tags.append(direction)

    if src_loopback or dest_loopback:
        tags.append("loopback")

    if dest_multicast:
        tags.append("multicast")

    if dest_broadcast:
        tags.append("broadcast")

    if src_private and dest_private:
        tags.append("private_lan")
    elif src_localish and dest_localish:
        tags.append("local_network")
    elif direction == "outbound" and not dest_localish:
        tags.append("internet_outbound")

    if dest_link_local:
        tags.append("link_local")

    if traffic_class:
        tags.append(traffic_class)

    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    # 1. Loopback traffic
    if src_loopback or dest_loopback:
        return {
            "traffic_class": "background_service" if traffic_class == "unknown" else traffic_class,
            "likely_benign": True,
            "benign_reason": "Loopback traffic stays on the local machine and is commonly generated by desktop applications and local services.",
            "context_tags": tags,
        }

    # 2. Broadcast traffic
    if dest_broadcast:
        return {
            "traffic_class": "local_discovery" if traffic_class == "unknown" else traffic_class,
            "likely_benign": True,
            "benign_reason": "Broadcast traffic is common on local networks for discovery and service coordination, but should still be reviewed if it becomes frequent or unusual.",
            "context_tags": tags,
        }

    # 3. Explicit SSDP multicast traffic
    if proto == "udp" and dest_ip == "239.255.255.250" and 1900 in ports:
        return {
            "traffic_class": "local_discovery",
            "likely_benign": True,
            "benign_reason": "This is SSDP multicast discovery traffic on 239.255.255.250:1900, which is very common for local device discovery, casting, and nearby-device communication.",
            "context_tags": tags,
        }

    # 3.1. Multicast traffic
    if dest_multicast:
        return {
            "traffic_class": "local_discovery" if traffic_class == "unknown" else traffic_class,
            "likely_benign": True,
            "benign_reason": "Multicast traffic is common for local discovery protocols, casting, and nearby-device communication on home or student networks.",
            "context_tags": tags,
        }

    # 4. Explicit local discovery protocols, including ICMPv6 / IPv6-ICMP
    if traffic_class == "local_discovery":
        return {
            "traffic_class": traffic_class,
            "likely_benign": True,
            "benign_reason": "This traffic matches common local discovery or LAN control behavior such as mDNS, SSDP, DHCP, LLMNR, NetBIOS, ICMPv6, broadcast, multicast, or nearby-device discovery.",
            "context_tags": tags,
        }

    # 5. NTP / time sync
    if traffic_class == "time_sync":
        return {
            "traffic_class": traffic_class,
            "likely_benign": True,
            "benign_reason": "Time synchronization traffic is normal background operating-system behavior.",
            "context_tags": tags,
        }

    # 6. DNS
    if traffic_class == "dns":
        common_dns_shape = (
            proto in {"udp", "tcp"}
            and 53 in ports
            and direction in {"outbound", "internal", ""}
        )

        if common_dns_shape:
            return {
                "traffic_class": traffic_class,
                "likely_benign": True,
                "benign_reason": "This looks like common DNS name-resolution traffic. It should usually be downgraded rather than fully suppressed unless repetition, volume, or other context becomes suspicious.",
                "context_tags": tags,
            }

        return {
            "traffic_class": traffic_class,
            "likely_benign": False,
            "benign_reason": "DNS-related traffic was detected, but the pattern is not typical enough to treat as clearly benign.",
            "context_tags": tags,
        }

    # 7. Common encrypted web traffic
    if traffic_class == "web_https":
        common_https_shape = (
            dest_port in {443, 8443}
            and direction in {"outbound", ""}
            and not dest_multicast
            and not dest_broadcast
        )

        if common_https_shape:
            return {
                "traffic_class": traffic_class,
                "likely_benign": True,
                "benign_reason": "This looks like common encrypted web or API traffic over HTTPS/TLS/QUIC. It is often benign, but should usually be downgraded rather than completely hidden.",
                "context_tags": tags,
            }

        return {
            "traffic_class": traffic_class,
            "likely_benign": False,
            "benign_reason": "Encrypted web-like traffic was detected, but the context is not typical enough to classify as clearly benign.",
            "context_tags": tags,
        }

    # 8. Plain HTTP
    if traffic_class == "web_http":
        return {
            "traffic_class": traffic_class,
            "likely_benign": False,
            "benign_reason": "Plain HTTP traffic may be normal, but it is less safe to automatically downgrade aggressively.",
            "context_tags": tags,
        }

    # 9. Streaming / nearby-device / local media behavior
    if traffic_class == "streaming":
        return {
            "traffic_class": traffic_class,
            "likely_benign": True,
            "benign_reason": "This looks like media, real-time audio/video, casting, or nearby-device communication traffic. That can be normal during music playback, calls, wireless peripherals, or local device use, but should still remain visible if it becomes repetitive or highly unusual.",
            "context_tags": tags,
        }

    # 10. Private-LAN UDP chatter with weak app labels
    if (
        proto == "udp"
        and src_localish
        and dest_localish
        and app_proto in {"", "unknown", "failed"}
    ):
        return {
            "traffic_class": "background_service" if traffic_class == "unknown" else traffic_class,
            "likely_benign": True,
            "benign_reason": "Local UDP traffic between nearby/private addresses is often benign background chatter on home networks, though repeated unusual behavior should still be reviewed.",
            "context_tags": tags,
        }

    # 11. Conservative fallback
    return {
        "traffic_class": traffic_class,
        "likely_benign": False,
        "benign_reason": "No strong benign-background rule matched, so this traffic should remain visible for review.",
        "context_tags": tags,
    }


# ============================================================
# Optional convenience wrapper
# ============================================================

def enrich_alert_context(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper to attach benign-background interpretation fields
    to an alert dictionary without changing model features or model score.

    Returned keys added:
    - traffic_class
    - likely_benign
    - benign_reason
    - context_tags
    """
    result = benign_background_context(alert)

    enriched = dict(alert)
    enriched["traffic_class"] = result["traffic_class"]
    enriched["likely_benign"] = result["likely_benign"]
    enriched["benign_reason"] = result["benign_reason"]
    enriched["context_tags"] = result["context_tags"]
    return enriched