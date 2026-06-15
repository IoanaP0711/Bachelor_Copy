from __future__ import annotations

from typing import Any, Dict, List, Optional


COMMON_DESKTOP_PORTS = {
    53,        # DNS
    67, 68,    # DHCP
    80, 443, 8080, 8443,   # web
    123,       # NTP
    1900,      # SSDP
    5353,      # mDNS
    5355,      # LLMNR
    25, 465, 587, 993, 995, 110, 143,   # mail
}

COMMON_DESKTOP_CLASSES = {
    "dns",
    "web_http",
    "web_https",
    "local_discovery",
    "streaming",
    "chat_messaging",
    "software_update",
    "time_sync",
    "background_app",
    "development_tool",
}

STRONGLY_BENIGN_CLASSES = {
    "dns",
    "web_http",
    "web_https",
    "local_discovery",
    "time_sync",
}

SOFT_BENIGN_CLASSES = {
    "streaming",
    "chat_messaging",
    "software_update",
    "background_app",
    "development_tool",
}

UNKNOWN_CLASSES = {"unknown", "other", "failed", ""}

TRAFFIC_CLASS_TEMPLATES = {
    "dns": "This looks like DNS traffic, which is commonly used for name resolution.",
    "web_http": "This looks like normal web traffic, which is common during everyday browsing or app use.",
    "web_https": "This looks like encrypted web or app traffic, which is very common on normal systems.",
    "local_discovery": "This looks like local discovery traffic, which is common on home and office networks.",
    "streaming": "This looks like streaming, media, or nearby-device communication traffic, which can be bursty during normal use.",
    "chat_messaging": "This looks like messaging or background communication traffic, which may be expected.",
    "software_update": "This looks like software update traffic, which can appear larger or burstier than usual.",
    "time_sync": "This looks like time synchronization traffic, which is normally expected.",
    "background_app": "This looks like background application traffic, such as sync or telemetry.",
    "development_tool": "This looks like developer-tool or local-service traffic, which can seem unusual but still be legitimate.",
    "remote_access": "This looks like remote access traffic. That may be normal, but it deserves attention if it was not started intentionally.",
    "file_transfer": "This looks like file transfer traffic, which can create larger flows than ordinary browsing.",
    "unknown": "This traffic does not clearly match a common desktop category.",
    "other": "This traffic does not clearly match a known desktop category.",
}


def _safe_lower(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        s = str(value).strip().lower()
        return s or default
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_ip(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        s = str(value).strip()
        return s or "unknown"
    except Exception:
        return "unknown"


def _normalize_severity(value: Any) -> str:
    s = _safe_lower(value, "ok")
    if s in {"crit", "critical"}:
        return "crit"
    if s in {"med", "medium"}:
        return "med"
    if s in {"warn", "warning"}:
        return "warn"
    return "ok"


SEV_TO_NUM = {
    "ok": 0,
    "warn": 1,
    "med": 2,
    "crit": 3,
}

NUM_TO_SEV = {
    0: "ok",
    1: "warn",
    2: "med",
    3: "crit",
}


def _severity_display(sev: str) -> str:
    sev = _normalize_severity(sev)
    return {
        "ok": "OK",
        "warn": "WARN",
        "med": "MED",
        "crit": "CRIT",
    }[sev]


def _normalize_display_label(value: Any) -> str:
    s = _safe_lower(value, "review")
    if s == "critical":
        return "critical"
    if s == "benign":
        return "benign"
    if s == "ok":
        return "ok"
    return "review"


def _display_label_text(label: str) -> str:
    label = _normalize_display_label(label)
    return {
        "ok": "OK",
        "benign": "BENIGN",
        "review": "REVIEW",
        "critical": "CRITICAL",
    }[label]


def _class_phrase(traffic_class: str) -> str:
    return TRAFFIC_CLASS_TEMPLATES.get(
        traffic_class,
        "This traffic does not clearly match a common desktop category.",
    )


def _score_phrase(raw_severity: str, score: float) -> str:
    raw_severity = _normalize_severity(raw_severity)

    if raw_severity == "ok":
        return "The model score stayed inside the learned normal range."
    if raw_severity == "warn":
        return "The model score was slightly above the learned normal range."
    if raw_severity == "med":
        return "The model score was clearly above the learned normal range."
    if raw_severity == "crit":
        return "The model score was far above the learned normal range."

    if score >= 100:
        return "The model score was extremely high."
    if score >= 10:
        return "The model score was clearly elevated."
    if score >= 3:
        return "The model score was somewhat elevated."
    return "The model score should be reviewed."


def _port_phrase(dport: Optional[int], common_desktop: bool) -> str:
    if dport is None:
        return "The destination port is not available."
    if common_desktop or dport in COMMON_DESKTOP_PORTS:
        return f"The destination port is {dport}, which is common in routine desktop traffic."
    return f"The destination port is {dport}, which is less typical for routine desktop traffic."


def _proto_phrase(proto: str, app_proto: str) -> str:
    proto_text = proto.upper() if proto not in {"", "unknown"} else "unknown protocol"
    app_text = app_proto.upper() if app_proto not in {"", "unknown", "failed"} else ""
    if app_text and app_text != proto_text:
        return f"Protocol context: {proto_text} / {app_text}."
    return f"Protocol context: {proto_text}."


def _is_common_desktop_traffic(
    traffic_class: str,
    protocol: str,
    dport: Optional[int],
    alert: Dict[str, Any],
) -> bool:
    explicit = alert.get("is_common_desktop_traffic")
    if explicit is not None:
        return bool(explicit)

    if traffic_class in COMMON_DESKTOP_CLASSES:
        return True

    if dport in COMMON_DESKTOP_PORTS:
        return True

    if protocol in {"dns", "http", "https", "tls"}:
        return True

    return False


def _repeat_info(alert: Dict[str, Any]) -> Dict[str, Any]:
    obj = alert.get("repeat_info")
    if isinstance(obj, dict):
        return obj
    return {}


def _has_repeated_behavior(alert: Dict[str, Any]) -> bool:
    info = _repeat_info(alert)

    if info:
        is_repeated = info.get("is_repeated")
        if is_repeated is not None:
            return bool(is_repeated)

        previous_count = info.get("previous_count")
        if previous_count is not None:
            try:
                return int(previous_count) > 0
            except (TypeError, ValueError):
                pass

    repeat_count = alert.get("repeat_count")
    if repeat_count is not None:
        try:
            return int(repeat_count) > 1
        except (TypeError, ValueError):
            pass

    return False


def _repeat_level(alert: Dict[str, Any]) -> str:
    level = alert.get("repeat_level")
    if level is not None:
        return _safe_lower(level, "single")

    info = _repeat_info(alert)
    level = info.get("repeat_level")
    if level is not None:
        return _safe_lower(level, "single")

    if _has_repeated_behavior(alert):
        return "repeated"

    return "single"


def _repeat_previous_count(alert: Dict[str, Any]) -> int:
    value = alert.get("repeat_previous_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    info = _repeat_info(alert)
    value = info.get("previous_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    repeat_count = alert.get("repeat_count")
    if repeat_count is not None:
        try:
            rc = int(repeat_count)
            return max(0, rc - 1)
        except (TypeError, ValueError):
            pass

    return 0


def _repeat_current_count(alert: Dict[str, Any]) -> int:
    value = alert.get("repeat_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    info = _repeat_info(alert)
    value = info.get("current_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    previous = _repeat_previous_count(alert)
    return previous + 1 if previous > 0 else 1


def _repeat_window_s(alert: Dict[str, Any]) -> int:
    info = _repeat_info(alert)
    value = info.get("repeat_window_s", alert.get("repeat_window_s", 45))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 45


def build_repeat_explanation(alert: Dict[str, Any]) -> str:
    is_anom = bool(alert.get("raw_model_flag", alert.get("is_anom", False)))
    if not is_anom:
        return "No repeated anomalous pattern was considered because this event is not marked as anomalous."

    level = _repeat_level(alert)
    current_count = _repeat_current_count(alert)
    window_s = _repeat_window_s(alert)

    key = alert.get("repetition_key")
    if not isinstance(key, dict):
        info = _repeat_info(alert)
        key = info.get("repeat_key") if isinstance(info.get("repeat_key"), dict) else {}

    dest_ip = _clean_ip(key.get("dest_ip", alert.get("dest_ip", "?")))
    dest_port = key.get("dest_port", alert.get("dest_port", "?"))
    proto = key.get("proto", alert.get("proto", alert.get("app_proto", "?")))

    target = f"{dest_ip}:{dest_port}/{str(proto).upper()}"

    if level == "single":
        return (
            f"This event appears isolated. Only {current_count} matching anomalous flow "
            f"toward {target} was seen in the last {window_s} seconds."
        )

    if level == "repeated":
        return (
            f"Similar anomalous behavior was seen {current_count} times toward {target} "
            f"in the last {window_s} seconds. Repetition makes it more worth checking."
        )

    return (
        f"Similar anomalous behavior appears persistent. {current_count} matching anomalous flows "
        f"toward {target} were seen in the last {window_s} seconds. Persistent repetition is more concerning "
        f"than a one-off deviation."
    )


def adjust_final_severity(alert: Dict[str, Any]) -> Dict[str, Any]:
    raw_severity = _normalize_severity(alert.get("raw_severity", alert.get("severity")))
    raw_num = SEV_TO_NUM[raw_severity]
    final_num = raw_num

    score = _to_float(
        alert.get("ae_score", alert.get("anomaly_score", alert.get("score"))),
        default=0.0,
    )

    traffic_class = _safe_lower(alert.get("traffic_class", alert.get("class")), "unknown")
    protocol = _safe_lower(
        alert.get("protocol", alert.get("proto", alert.get("app_proto"))),
        "unknown",
    )
    dport = _to_int(alert.get("dest_port", alert.get("dport")))
    repeat_level = _repeat_level(alert)

    is_strong_local_discovery = (
        traffic_class == "local_discovery"
        and bool(alert.get("likely_benign", False))
    )

    common_desktop = _is_common_desktop_traffic(
        traffic_class=traffic_class,
        protocol=protocol,
        dport=dport,
        alert=alert,
    )

    is_unknown = traffic_class in UNKNOWN_CLASSES
    is_strongly_benign = traffic_class in STRONGLY_BENIGN_CLASSES
    is_soft_benign = traffic_class in SOFT_BENIGN_CLASSES

    reasons: List[str] = []

    if common_desktop and not is_unknown:
        if is_strongly_benign:
            final_num = max(0, final_num - 2)
            reasons.append("traffic matches a common benign pattern")
        elif is_soft_benign:
            final_num = max(0, final_num - 1)
            reasons.append("common benign desktop traffic")
        else:
            final_num = max(0, final_num - 1)
            reasons.append("known desktop protocol")

    if is_unknown:
        final_num = max(final_num, raw_num - 1)
        if final_num < raw_num:
            reasons.append("unknown traffic kept visible")

    if repeat_level == "persistent" and not is_strong_local_discovery:
        final_num = min(3, final_num + 1)
        reasons.append("persistent repeated behavior")

    final_num = max(0, min(3, final_num))
    final_severity = NUM_TO_SEV[final_num]

    alert["raw_severity"] = raw_severity
    alert["final_severity"] = final_severity
    alert["adjustment_reasons"] = reasons
    return alert


def make_display_label_with_reason(alert: Dict[str, Any]) -> tuple[str, str]:
    raw_sev = _normalize_severity(alert.get("raw_severity", "ok"))
    final_sev = _normalize_severity(alert.get("final_severity", "ok"))
    reasons = list(alert.get("adjustment_reasons", []))
    repeat_level = _repeat_level(alert)

    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    likely_benign = bool(alert.get("likely_benign", False))

    is_strong_local_discovery = (
        traffic_class == "local_discovery" and likely_benign
    )

    if raw_sev == "crit" and final_sev in {"ok", "warn"} and not is_strong_local_discovery:
        final_sev = "med"
        reasons.append("high raw anomaly score kept visible for review")

    alert["final_severity"] = final_sev
    alert["severity"] = final_sev

    if final_sev == "ok":
        label = "OK"
    elif final_sev == "warn":
        label = "BENIGN"
    elif final_sev == "med":
        label = "REVIEW"
    else:
        label = "CRITICAL"

    alert["final_label"] = label
    alert["display_label"] = label
    alert["adjustment_reasons"] = reasons

    if label == "OK":
        reason = "The raw model score did not cross the anomaly threshold."
    elif label == "BENIGN":
        reason = (
            f"Model flagged the flow ({raw_sev.upper()}), but context suggests likely benign "
            f"{alert.get('traffic_class', 'traffic')}."
        )
    elif label == "REVIEW":
        reason = (
            f"Unusual flow kept for review ({raw_sev.upper()} raw, {final_sev.upper()} adjusted)."
        )
    else:
        reason = (
            f"Critical anomaly detected ({raw_sev.upper()} raw, {final_sev.upper()} adjusted), "
            "requires immediate attention."
        )

    if reasons:
        reason += " Reason: " + ", ".join(reasons) + "."

    alert["display_label_reason"] = reason
    return label, reason


def make_display_label(alert: Dict[str, Any]) -> str:
    label, _reason = make_display_label_with_reason(alert)
    return label


def _adjustment_reason_text(alert: Dict[str, Any]) -> str:
    raw_severity = _normalize_severity(alert.get("raw_severity"))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))

    reasons = list(alert.get("adjustment_reasons", []))
    extra = str(alert.get("final_severity_reason", "") or "").strip()

    if extra:
        reasons.append(extra)

    seen = set()
    clean_reasons = []
    for r in reasons:
        rr = str(r).strip()
        if rr and rr not in seen:
            seen.add(rr)
            clean_reasons.append(rr)

    if raw_severity == final_severity:
        if clean_reasons:
            return (
                "Technical severity remained at the same visible level after contextual checks: "
                + "; ".join(clean_reasons) + "."
            )
        return "Technical severity remained unchanged after contextual checks."

    if SEV_TO_NUM[final_severity] < SEV_TO_NUM[raw_severity]:
        if clean_reasons:
            return (
                "Technical severity was reduced after contextual checks: "
                + "; ".join(clean_reasons) + "."
            )
        return "Technical severity was reduced after contextual checks."

    if SEV_TO_NUM[final_severity] > SEV_TO_NUM[raw_severity]:
        if clean_reasons:
            return (
                "Technical severity was increased after contextual checks: "
                + "; ".join(clean_reasons) + "."
            )
        return "Technical severity was increased after contextual checks."

    if clean_reasons:
        return (
            "Technical severity was kept visible after contextual checks: "
            + "; ".join(clean_reasons) + "."
        )

    return "Technical severity was kept visible after contextual checks."


def _display_reason_text(alert: Dict[str, Any]) -> str:
    reason = str(alert.get("display_label_reason", "") or "").strip()
    if not reason:
        return "No extra dashboard-label reason was recorded."
    return reason


def _final_label_interpretation(
    label: str,
    raw_severity: str,
    final_severity: str,
    traffic_class: str,
    repeat_level: str,
) -> str:
    label = _normalize_display_label(label)
    raw_text = _severity_display(raw_severity)
    final_text = _severity_display(final_severity)

    if label == "ok":
        return "The model did not detect an operationally meaningful anomaly for this flow."

    if label == "benign":
        msg = (
            f"The model flagged this flow at raw severity {raw_text}, but the surrounding context "
            f"looks consistent with normal desktop activity, so it is treated as BENIGN."
        )
        if repeat_level in {"repeated", "persistent"}:
            msg += " Even so, repeated behavior should still be watched."
        return msg

    if label == "review":
        msg = (
            f"The flow was flagged by the model and is not clearly normal, so it remains under REVIEW. "
            f"The adjusted technical severity is {final_text}."
        )
        if traffic_class in UNKNOWN_CLASSES:
            msg += " It is especially worth checking because the traffic category is unknown."
        if repeat_level == "repeated":
            msg += " Similar behavior was seen again recently."
        elif repeat_level == "persistent":
            msg += " Similar behavior was persistent in the short-term window."
        return msg

    return (
        f"The flow remains CRITICAL after combining the raw anomaly signal with contextual checks. "
        f"The adjusted technical severity is {final_text}."
    )


def _build_summary(alert: Dict[str, Any]) -> str:
    raw_severity = _normalize_severity(alert.get("raw_severity"))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    label = _normalize_display_label(
        alert.get("final_label", alert.get("display_label", "review"))
    )
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    repeat_level = _repeat_level(alert)

    raw_text = _severity_display(raw_severity)
    final_text = _severity_display(final_severity)

    parts: List[str] = []

    if label == "ok":
        parts.append("No operationally relevant anomaly detected.")
    elif label == "benign":
        parts.append(f"Model flagged the flow ({raw_text}), but context suggests likely benign {traffic_class} traffic.")
    elif label == "review":
        parts.append(f"Unusual flow kept for review ({raw_text} raw")
        if final_text != raw_text:
            parts[-1] += f", {final_text} adjusted"
        else:
            parts[-1] += ", still visible"
        parts[-1] += ")."
    else:
        parts.append(f"High-priority anomalous flow ({raw_text} raw")
        if final_text != raw_text:
            parts[-1] += f", {final_text} adjusted"
        parts[-1] += ")."

    if repeat_level == "repeated":
        parts.append("Seen again recently.")
    elif repeat_level == "persistent":
        parts.append("Persistent short-term repetition observed.")

    text = " ".join(parts).strip()
    if len(text) > 220:
        return text[:217].rstrip() + "..."
    return text


def _build_modal_explanation(alert: Dict[str, Any]) -> str:
    score = _to_float(alert.get("ae_score", alert.get("anomaly_score", alert.get("score"))), 0.0)
    raw_severity = _normalize_severity(alert.get("raw_severity"))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    label = _normalize_display_label(
        alert.get("final_label", alert.get("display_label", "review"))
    )
    proto = _safe_lower(alert.get("proto", alert.get("protocol")), "unknown")
    app_proto = _safe_lower(alert.get("app_proto"), "unknown")
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    src_ip = _clean_ip(alert.get("src_ip"))
    dest_ip = _clean_ip(alert.get("dest_ip"))
    src_port = _to_int(alert.get("src_port", alert.get("sport")))
    dest_port = _to_int(alert.get("dest_port", alert.get("dport")))
    likely_benign = bool(alert.get("likely_benign", False))

    common_desktop = _is_common_desktop_traffic(
        traffic_class=traffic_class,
        protocol=proto,
        dport=dest_port,
        alert=alert,
    )

    flow_text = f"Flow: {src_ip}"
    if src_port is not None:
        flow_text += f":{src_port}"
    flow_text += f" -> {dest_ip}"
    if dest_port is not None:
        flow_text += f":{dest_port}"
    flow_text += "."

    parts: List[str] = []

    parts.append(
        f"The autoencoder assigned anomaly score {score:.6f}, which maps to raw severity {_severity_display(raw_severity)}. "
        f"{_score_phrase(raw_severity, score)}"
    )
    parts.append(flow_text)
    parts.append(_proto_phrase(proto, app_proto))
    parts.append(f"Traffic class: {traffic_class}. {_class_phrase(traffic_class)}")
    parts.append(_port_phrase(dest_port, common_desktop))

    if likely_benign:
        parts.append(
            "The traffic also matches indicators of likely benign behavior. "
            "That does not mean the flow is guaranteed safe; it means the surrounding context looks consistent with normal use."
        )

    parts.append(
        f"After contextual analysis, the adjusted technical severity is {_severity_display(final_severity)} "
        f"and the dashboard label is {_display_label_text(label)}."
    )
    parts.append(_adjustment_reason_text(alert))
    parts.append(f"Dashboard label reason: {_display_reason_text(alert)}")
    parts.append(build_repeat_explanation(alert))

    if label == "critical":
        parts.append(
            "This should be treated as a high-priority event for investigation. "
            "Even so, the system is detecting unusual behavior, not proving malware by itself."
        )
    elif label == "review":
        parts.append(
            "This event is worth analyst review because it is unusual and not clearly explained away by context. "
            "It should be interpreted as suspicious or uncommon behavior, not as confirmed compromise."
        )
    elif label == "benign":
        parts.append(
            "This event was kept visible for transparency, but the dashboard interpretation is that it is more likely normal background or user activity than a direct threat."
        )
    else:
        parts.append("Overall, this event is treated as operationally normal.")

    return " ".join(p.strip() for p in parts if p and p.strip())


def _possible_explanation_text(alert: Dict[str, Any]) -> str:
    label = _normalize_display_label(
        alert.get("final_label", alert.get("display_label", "review"))
    )
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    repeat_level = _repeat_level(alert)

    if label not in {"review", "critical"}:
        return ""

    if traffic_class == "local_discovery":
        text = (
            "This may be related to local network discovery, multicast, casting, "
            "or nearby-device communication. The traffic type itself is often normal, "
            "but the observed pattern was unusual enough to remain visible for review."
        )
    elif traffic_class == "dns":
        text = (
            "This may be related to name resolution from a browser, app startup, "
            "background refresh, or another normal service, but the observed pattern "
            "was unusual enough to remain visible for review."
        )
    elif traffic_class == "web_https":
        text = (
            "This may be related to normal encrypted web or app traffic such as browsing, "
            "streaming, sync, or updates, but the observed pattern was unusual enough "
            "to remain visible for review."
        )
    elif traffic_class == "streaming":
        text = (
            "This may be related to media playback, real-time audio/video, casting, "
            "or nearby-device communication such as speakers, headphones, or wireless media devices, "
            "but the observed pattern was unusual enough to remain visible for review."
        )
    elif traffic_class == "chat_messaging":
        text = (
            "This may be related to chat, voice, or video-call traffic, "
            "but the observed pattern was unusual enough to remain visible for review."
        )
    elif traffic_class == "remote_access":
        text = (
            "This may be related to intentional remote access activity, "
            "but remote access should remain visible when it appears unusual."
        )
    elif traffic_class == "software_update":
        text = (
            "This may be related to software updates or package retrieval, "
            "but the observed pattern was unusual enough to remain visible for review."
        )
    elif traffic_class == "background_app":
        text = (
            "This may be related to normal background application activity, "
            "but the observed pattern was unusual enough to remain visible for review."
        )
    elif traffic_class == "development_tool":
        text = (
            "This may be related to developer tooling, local services, or test activity, "
            "but the observed pattern was unusual enough to remain visible for review."
        )
    else:
        text = (
            "The traffic does not clearly match a strongly explainable benign category, "
            "so it remains visible for review."
        )

    if repeat_level == "repeated":
        text += " Similar behavior was seen again recently."
    elif repeat_level == "persistent":
        text += " Similar behavior was persistent in the short-term window."

    return text


def _what_to_check_text(alert: Dict[str, Any]) -> str:
    label = _normalize_display_label(
        alert.get("final_label", alert.get("display_label", "review"))
    )
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))

    if label not in {"review", "critical"}:
        return ""

    if traffic_class == "local_discovery":
        return (
            "Check whether a smart device, TV, speaker, phone, printer, casting app, "
            "or another nearby device was active at that time."
        )

    if traffic_class == "dns":
        return (
            "Check which browser tabs, apps, or background services were started or refreshed "
            "around that time."
        )

    if traffic_class == "web_https":
        return (
            "Check whether browsing, streaming, cloud sync, updates, messaging apps, "
            "or another encrypted application was active at that time."
        )

    if traffic_class == "streaming":
        return (
            "Check whether music, video, calls, casting, wireless speakers, headphones, "
            "or another media-related device or app was active at that time."
        )

    if traffic_class == "chat_messaging":
        return (
            "Check whether a chat, voice-call, or video-call application was active at that time."
        )

    if traffic_class == "remote_access":
        return (
            "Check whether SSH, RDP, VNC, remote support, or another remote-access tool "
            "was intentionally used at that time."
        )

    if traffic_class == "software_update":
        return (
            "Check whether the operating system, browser, package manager, or another application "
            "was updating at that time."
        )

    if traffic_class in {"background_app", "development_tool"}:
        return (
            "Check which background applications, agents, sync tools, containers, developer tools, "
            "or local services were active at that time."
        )

    return (
        "Check which application, service, device, or network activity was active at that time, "
        "and whether similar behavior repeats."
    )

def _prepare_layered_explanation_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(alert, dict):
        alert = {}

    if "final_severity" not in alert:
        alert = adjust_final_severity(alert)

    display_label = alert.get("final_label", alert.get("display_label"))
    if not display_label or not alert.get("display_label_reason"):
        display_label, display_reason = make_display_label_with_reason(alert)
        alert["display_label"] = display_label
        alert["final_label"] = display_label
        alert["display_label_reason"] = display_reason

    return alert


def _human_traffic_class(traffic_class: str) -> str:
    traffic_class = _safe_lower(traffic_class, "unknown")
    return {
        "dns": "name lookup activity",
        "web_http": "normal web activity",
        "web_https": "encrypted web or application activity",
        "local_discovery": "local network discovery activity",
        "streaming": "media, streaming, or nearby-device activity",
        "chat_messaging": "chat, call, or messaging activity",
        "software_update": "software update activity",
        "time_sync": "time synchronization activity",
        "background_app": "background application activity",
        "development_tool": "developer tool or local service activity",
        "remote_access": "remote access activity",
        "file_transfer": "file transfer activity",
        "unknown": "network activity that is not clearly recognized",
        "other": "network activity that is not clearly recognized",
        "failed": "network activity that could not be classified clearly",
        "": "network activity that is not clearly recognized",
    }.get(traffic_class, traffic_class.replace("_", " ") + " activity")


def _flow_text(alert: Dict[str, Any], include_ports: bool = True) -> str:
    src_ip = _clean_ip(alert.get("src_ip"))
    dest_ip = _clean_ip(alert.get("dest_ip"))
    src_port = _to_int(alert.get("src_port", alert.get("sport")))
    dest_port = _to_int(alert.get("dest_port", alert.get("dport")))

    src = src_ip
    dst = dest_ip

    if include_ports and src_port is not None:
        src += f":{src_port}"
    if include_ports and dest_port is not None:
        dst += f":{dest_port}"

    return f"{src} -> {dst}"


def _context_changed_raw_output(alert: Dict[str, Any]) -> bool:
    raw_severity = _normalize_severity(alert.get("raw_severity", alert.get("severity")))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    return raw_severity != final_severity


def _context_change_simple_text(alert: Dict[str, Any]) -> str:
    if not _context_changed_raw_output(alert):
        return ""

    raw_severity = _normalize_severity(alert.get("raw_severity", alert.get("severity")))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))

    if SEV_TO_NUM[final_severity] < SEV_TO_NUM[raw_severity]:
        return (
            "The first automatic warning was lowered after the system checked the surrounding context "
            "and found signs of normal activity."
        )

    return (
        "The first automatic warning was raised after the system checked the surrounding context "
        "and found that the behavior repeated or looked more concerning."
    )


def _context_change_analyst_text(alert: Dict[str, Any]) -> str:
    raw_severity = _normalize_severity(alert.get("raw_severity", alert.get("severity")))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    raw_text = _severity_display(raw_severity)
    final_text = _severity_display(final_severity)

    if _context_changed_raw_output(alert):
        reasons = list(alert.get("adjustment_reasons", []))
        reason_text = f" Reason: {', '.join(str(r) for r in reasons)}." if reasons else ""
        return (
            f"Context changed the raw model output from {raw_text} to {final_text} "
            f"before the final dashboard decision was shown.{reason_text}"
        )

    return f"Context did not change the raw model output; it remained {raw_text}."


def _context_change_technical_text(alert: Dict[str, Any]) -> str:
    raw_severity = _normalize_severity(alert.get("raw_severity", alert.get("severity")))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    raw_text = _severity_display(raw_severity)
    final_text = _severity_display(final_severity)

    if _context_changed_raw_output(alert):
        return (
            f"Contextual filtering changed the raw model severity from {raw_text} to {final_text}. "
            f"{_adjustment_reason_text(alert)}"
        )

    return (
        f"Contextual filtering did not change the raw model severity; it stayed {raw_text}. "
        f"{_adjustment_reason_text(alert)}"
    )


def _format_score(score: float) -> str:
    return f"{score:.6f}"


def _format_repetition_for_analyst(alert: Dict[str, Any]) -> str:
    repeat_level = _repeat_level(alert)
    current_count = _repeat_current_count(alert)
    window_s = _repeat_window_s(alert)

    if repeat_level == "single":
        return f"No repeated behavior was observed in the last {window_s} seconds."

    if repeat_level == "repeated":
        return f"Repeated behavior was observed: {current_count} similar flows in the last {window_s} seconds."

    return f"Persistent repeated behavior was observed: {current_count} similar flows in the last {window_s} seconds."


def _format_contributors(alert: Dict[str, Any], limit: int = 5) -> str:
    candidates = (
        alert.get("top_features"),
        alert.get("top_contributors"),
        alert.get("top_anomaly_contributors"),
        alert.get("contributors"),
        alert.get("feature_contributions"),
        alert.get("contribution_summary"),
    )

    contributors: List[tuple[str, Any]] = []

    for obj in candidates:
        if not obj:
            continue

        if isinstance(obj, dict):
            items = list(obj.items())

            def sort_key(item: tuple[str, Any]) -> float:
                try:
                    return abs(float(item[1]))
                except (TypeError, ValueError):
                    return 0.0

            for key, value in sorted(items, key=sort_key, reverse=True)[:limit]:
                contributors.append((str(key), value))
            break

        if isinstance(obj, list):
            for item in obj[:limit]:
                if isinstance(item, dict):
                    name = (
                        item.get("feature")
                        or item.get("name")
                        or item.get("key")
                        or item.get("column")
                        or "unknown_feature"
                    )
                    value = (
                        item.get("err")
                        if "err" in item
                        else item.get(
                            "value",
                            item.get("contribution", item.get("score", item.get("error", ""))),
                        )
                    )
                    contributors.append((str(name), value))
                elif isinstance(item, (list, tuple)) and item:
                    name = str(item[0])
                    value = item[1] if len(item) > 1 else ""
                    contributors.append((name, value))
                else:
                    contributors.append((str(item), ""))
            break

    if not contributors:
        return "No contributor list was recorded for this alert."

    formatted = []
    for name, value in contributors[:limit]:
        if value == "" or value is None:
            formatted.append(name)
        else:
            try:
                formatted.append(f"{name}={float(value):.6f}")
            except (TypeError, ValueError):
                formatted.append(f"{name}={value}")

    return ", ".join(formatted)


def build_simple_explanation(alert: Dict[str, Any]) -> str:
    alert = _prepare_layered_explanation_alert(alert)

    label = _normalize_display_label(alert.get("final_label", alert.get("display_label", "review")))
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    repeat_level = _repeat_level(alert)

    activity_text = _human_traffic_class(traffic_class)
    context_text = _context_change_simple_text(alert)

    parts: List[str] = []

    if label == "ok":
        parts.append(
            "This network activity looks normal. The system did not find behavior that needs attention."
        )
    elif label == "benign":
        parts.append(
            "This network activity looked unusual at first, but it matches a type of activity that is usually normal."
        )
    elif label == "review":
        parts.append(
            "This network activity is unusual and should be checked. It is not proof of an attack, but it should not be ignored."
        )
    else:
        parts.append(
            "This network activity is highly unusual and should be checked as soon as possible. It may indicate risky behavior, but the alert alone is not proof of malware."
        )

    parts.append(f"It looks related to {activity_text}.")

    if repeat_level == "repeated":
        parts.append("Similar activity was seen again recently.")
    elif repeat_level == "persistent":
        parts.append("Similar activity kept appearing in a short period of time.")

    if context_text:
        parts.append(context_text)

    return " ".join(p.strip() for p in parts if p and p.strip())


def build_analyst_explanation(alert: Dict[str, Any]) -> str:
    alert = _prepare_layered_explanation_alert(alert)

    label = _normalize_display_label(alert.get("final_label", alert.get("display_label", "review")))
    score = _to_float(alert.get("ae_score", alert.get("anomaly_score", alert.get("score"))), 0.0)
    proto = _safe_lower(alert.get("proto", alert.get("protocol")), "unknown")
    app_proto = _safe_lower(alert.get("app_proto"), "unknown")
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))

    proto_text = proto.upper() if proto not in {"", "unknown"} else "unknown protocol"
    app_text = app_proto.upper() if app_proto not in {"", "unknown", "failed"} else ""
    protocol_context = f"{proto_text} / {app_text}" if app_text and app_text != proto_text else proto_text

    parts: List[str] = [
        f"Final decision: {_display_label_text(label)}.",
        f"Flow: {_flow_text(alert, include_ports=True)}.",
        f"Protocol: {protocol_context}. Traffic class: {traffic_class}.",
        f"Anomaly score: {_format_score(score)}.",
        _format_repetition_for_analyst(alert),
        _context_change_analyst_text(alert),
    ]

    possible = _possible_explanation_text(alert)
    what_to_check = _what_to_check_text(alert)

    if possible:
        parts.append(f"Possible explanation: {possible}")
    if what_to_check:
        parts.append(f"What to check: {what_to_check}")

    return " ".join(p.strip() for p in parts if p and p.strip())


def build_technical_explanation(alert: Dict[str, Any]) -> str:
    alert = _prepare_layered_explanation_alert(alert)

    label = _normalize_display_label(alert.get("final_label", alert.get("display_label", "review")))
    raw_severity = _normalize_severity(alert.get("raw_severity", alert.get("severity")))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    score = _to_float(alert.get("ae_score", alert.get("anomaly_score", alert.get("score"))), 0.0)
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    likely_benign = bool(alert.get("likely_benign", False))
    raw_flag = bool(alert.get("raw_model_flag", alert.get("is_anom", False)))

    parts: List[str] = [
        f"Final decision: {_display_label_text(label)}. Adjusted technical severity: {_severity_display(final_severity)}.",
        (
            f"The autoencoder reconstruction error is used as the anomaly score. "
            f"For this flow, anomaly score={_format_score(score)}, raw_model_flag={raw_flag}, "
            f"and raw model severity={_severity_display(raw_severity)}."
        ),
        f"Flow: {_flow_text(alert, include_ports=True)}.",
        f"Traffic class: {traffic_class}. likely_benign={likely_benign}.",
        _context_change_technical_text(alert),
        f"Repeat logic: {build_repeat_explanation(alert)}",
        f"Top anomaly contributors: {_format_contributors(alert, limit=5)}.",
        f"Dashboard label reason: {_display_reason_text(alert)}",
    ]

    return " ".join(p.strip() for p in parts if p and p.strip())

def build_recommended_action(alert: Dict[str, Any]) -> str:
    """
    Return a deterministic manual check for the operator.

    This helper only explains what the operator may verify next.
    It does not modify the anomaly score, raw severity, contextual filtering,
    repeat logic, or final decision.

    Technical terms include short plain-language explanations so that the
    recommendation remains understandable to both non-technical users and
    experienced analysts.
    """
    if not isinstance(alert, dict):
        alert = {}

    explicit_final_decision = (
        alert.get("final_label")
        or alert.get("display_label")
        or alert.get("final_decision")
    )

    final_technical_severity_value = alert.get(
        "final_severity",
        alert.get("severity"),
    )

    # Prefer the final operational decision shown in the interface.
    if explicit_final_decision not in (None, ""):
        final_decision = _normalize_display_label(
            explicit_final_decision
        )

    # Older or incomplete alerts may contain only the final technical
    # severity. Convert that value to its dashboard decision.
    elif final_technical_severity_value not in (None, ""):
        final_decision = {
            "ok": "ok",
            "warn": "benign",
            "med": "review",
            "crit": "critical",
        }[
            _normalize_severity(
                final_technical_severity_value
            )
        ]

    # REVIEW is a cautious fallback when the final result is missing.
    else:
        final_decision = "review"

    recommendation_by_decision = {
        "ok": (
            "No immediate action required."
        ),
        "benign": (
            "Check only if this traffic was unexpected or does not match "
            "the usual activity on the home or office network."
        ),
        "review": (
            "Verify the source IP, meaning the device that sent the traffic; "
            "the destination port, meaning the contacted service; and whether "
            "similar events repeat."
        ),
        "critical": (
            "Inspect the source host, meaning the device that sent the traffic, "
            "and review repeated connections that follow the same pattern."
        ),
    }

    recommended_check = recommendation_by_decision[
        final_decision
    ]

    raw_severity_value = (
        alert.get("raw_severity")
        or alert.get("raw_model_severity")
    )

    # Context adjustment must be detected by comparing the raw technical
    # severity with the final technical severity. The dashboard label uses a
    # different vocabulary and should not be used for this comparison.
    if (
        raw_severity_value not in (None, "")
        and final_technical_severity_value not in (None, "")
    ):
        raw_severity = _normalize_severity(
            raw_severity_value
        )

        final_technical_severity = _normalize_severity(
            final_technical_severity_value
        )

        if raw_severity != final_technical_severity:
            raw_text = _severity_display(
                raw_severity
            )

            final_technical_text = _severity_display(
                final_technical_severity
            )

            final_decision_text = _display_label_text(
                final_decision
            )

            recommended_check += (
                " Context changed the final result: the model first assigned "
                f"raw severity {raw_text}, the context-adjusted technical "
                f"severity became {final_technical_text}, and the final "
                f"decision shown to the operator is {final_decision_text}."
            )

    return recommended_check

def build_explanation_bundle(alert: Dict[str, Any]) -> Dict[str, str]:
    if "final_severity" not in alert:
        alert = adjust_final_severity(alert)

    display_label = alert.get("final_label", alert.get("display_label"))
    if not display_label:
        display_label, display_reason = make_display_label_with_reason(alert)
        alert["display_label"] = display_label
        alert["final_label"] = display_label
        alert["display_label_reason"] = display_reason
    elif not alert.get("display_label_reason"):
        _, display_reason = make_display_label_with_reason(alert)
        alert["display_label_reason"] = display_reason

    raw_severity = _normalize_severity(alert.get("raw_severity"))
    final_severity = _normalize_severity(alert.get("final_severity", raw_severity))
    traffic_class = _safe_lower(alert.get("traffic_class", "unknown"))
    repeat_level = _repeat_level(alert)

    summary = _build_summary(alert)
    interpretation = _final_label_interpretation(
        label=alert.get("final_label", alert.get("display_label", "review")),
        raw_severity=raw_severity,
        final_severity=final_severity,
        traffic_class=traffic_class,
        repeat_level=repeat_level,
    )

    simple_explanation = build_simple_explanation(alert)
    analyst_explanation = build_analyst_explanation(alert)
    technical_explanation = build_technical_explanation(alert)

    legacy_explanation = _build_modal_explanation(alert)

    adjustment_reason = _adjustment_reason_text(alert)
    possible_explanation = _possible_explanation_text(alert)
    what_to_check = _what_to_check_text(alert)
    recommended_action = build_recommended_action(alert)
    return {
        "summary": summary,
        "interpretation": interpretation,

        
        "explanation": simple_explanation,
        "short_summary": summary,
        "full_explanation": technical_explanation,

        
        "simple_explanation": simple_explanation,
        "analyst_explanation": analyst_explanation,
        "technical_explanation": technical_explanation,

        
        "adjustment_reason": adjustment_reason,
        "possible_explanation": possible_explanation,
        "what_to_check": what_to_check,
        "recommended_action": recommended_action,

        
        "legacy_explanation": legacy_explanation,
    }

def build_explanation(alert: Dict[str, Any]) -> str:
    return build_explanation_bundle(alert)["explanation"]