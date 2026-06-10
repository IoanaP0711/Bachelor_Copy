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

function safeNum(value, digits = 2, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  const n = Number(value);
  if (Number.isNaN(n)) return escapeHtml(value);
  return n.toFixed(digits);
}

function formatTime(tsUnix) {
  if (!tsUnix) return "-";
  try {
    return new Date(tsUnix * 1000)
      .toISOString()
      .replace("T", " ")
      .split(".")[0] + "Z";
  } catch {
    return "-";
  }
}

function badge(text, cls = "") {
  return `<span class="badge ${cls}">${safeText(text)}</span>`;
}

function protoBadge(proto) {
  const p = String(proto ?? "").toUpperCase();

  if (!p) return badge("-", "badge-gray");
  if (p === "TCP") return badge(p, "badge-blue");
  if (p === "UDP") return badge(p, "badge-cyan");
  if (p === "ICMP" || p === "ICMPV6" || p === "IPV6-ICMP") {
    return badge(p, "badge-purple");
  }

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

function rowClassFromLabel(label) {
  const l = String(label || "").toUpperCase();

  if (l === "OK") return "row-ok";
  if (l === "BENIGN") return "row-benign";
  if (l === "REVIEW") return "row-review";
  if (l === "CRITICAL") return "row-critical";

  return "";
}

function setText(id, value, fallback = "-") {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
}

function setHtml(id, html) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = html;
}

function setLastUpdate() {
  setText("lastUpdate", "Updated: " + new Date().toLocaleTimeString());
}

function shortTableReason(alertObj) {
  const label = String(
    alertObj.final_label ||
    alertObj.display_label ||
    alertObj.severity ||
    ""
  ).toUpperCase();

  const trafficClass = String(alertObj.traffic_class || "").toLowerCase();
  const repeatLevel = String(alertObj.repeat_level || "").toLowerCase();
  const repeatCount = Number(alertObj.repeat_count || 0);
  const likelyBenign = Boolean(alertObj.likely_benign);

  const isRepeated =
    repeatCount > 0 ||
    repeatLevel === "repeated" ||
    repeatLevel === "persistent";

  const reasonText = String(
    alertObj.display_label_reason ||
    alertObj.adjustment_reason ||
    alertObj.summary ||
    alertObj.interpretation ||
    alertObj.explanation ||
    ""
  ).toLowerCase();

  if (trafficClass === "dns" && likelyBenign) {
    return "Internal DNS traffic";
  }

  if (trafficClass === "local_discovery") {
    return "Local discovery traffic";
  }

  if (
    repeatLevel === "persistent" ||
    reasonText.includes("persistent") ||
    reasonText.includes("forced escalation due to repeated anomalous behavior")
  ) {
    return "Persistent anomaly";
  }

  if (label === "OK" && isRepeated) {
    return "Repeated normal traffic";
  }

  if (label === "OK") {
    return "Normal traffic";
  }

  if (label === "BENIGN") {
    return "Likely benign";
  }

  if (label === "REVIEW") {
    return "Needs review";
  }

  if (label === "CRITICAL" || label === "CRIT") {
    return "Critical anomaly";
  }

  return "Needs review";
}