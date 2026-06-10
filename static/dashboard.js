let running = true;
let timer = null;
let latestRows = [];

function setStatusPillLabel(label) {
  const pill = document.getElementById("statusPill");
  const l = String(label || "").toUpperCase();

  if (l === "OK") {
    pill.className = "pill ok";
    pill.textContent = "OK";
  } else if (l === "BENIGN") {
    pill.className = "pill benign";
    pill.textContent = "BENIGN";
  } else if (l === "REVIEW") {
    pill.className = "pill review";
    pill.textContent = "REVIEW";
  } else if (l === "CRITICAL") {
    pill.className = "pill critical";
    pill.textContent = "CRITICAL";
  } else {
    pill.className = "pill";
    pill.textContent = "…";
  }
}

function rowClassFromLabel(label) {
  const l = String(label || "").toUpperCase();
  if (l === "OK") return "row-ok";
  if (l === "BENIGN") return "row-benign";
  if (l === "REVIEW") return "row-review";
  if (l === "CRITICAL") return "row-critical";
  return "";
}

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

function safeNum(value, digits = 0, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  const n = Number(value);
  if (Number.isNaN(n)) return escapeHtml(value);
  return n.toFixed(digits);
}

function badge(text, cls = "") {
  return `<span class="badge ${cls}">${safeText(text)}</span>`;
}

function protoBadge(proto) {
  const p = String(proto ?? "").toUpperCase();
  if (!p) return badge("-", "badge-gray");
  if (p === "TCP") return badge(p, "badge-blue");
  if (p === "UDP") return badge(p, "badge-cyan");
  if (p === "ICMP" || p === "ICMPV6" || p === "IPV6-ICMP") return badge(p, "badge-purple");
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

function renderTopFeatures(features) {
  const helperText = `
    <div class="helper-text">
      Reconstruction error measures how different the original feature was from the value reconstructed by the autoencoder.
      Higher values contributed more to the anomaly score.
    </div>
    <div class="feature-title">Top anomaly contributors:</div>
  `;

  if (!Array.isArray(features) || !features.length) {
    return `
      ${helperText}
      <div class="muted">No feature contribution data available.</div>
    `;
  }

  const items = features.map(f => {
    const name = safeText(f.name, "?");
    const err = safeNum(f.err, 4, "-");

    return `
      <li>
        <span class="mono">${name}</span>
        — reconstruction error:
        <span class="mono">${err}</span>
      </li>
    `;
  }).join("");

  return `
    ${helperText}
    <ul class="feature-list">${items}</ul>
  `;
}

function openExplanationModal(alertObj) {
  const repKey = alertObj.repetition_key
    ? JSON.stringify(alertObj.repetition_key)
    : "-";

  const finalLabel =
    alertObj.final_label ?? alertObj.display_label ?? alertObj.severity ?? "-";

  const finalLabelUpper = String(finalLabel || "").toUpperCase();
  const showHints = finalLabelUpper === "REVIEW" || finalLabelUpper === "CRITICAL";

  document.getElementById("modalFlowId").textContent = alertObj.flow_id ?? "-";

  document.getElementById("modalDisplayLabel").textContent = finalLabel;
  document.getElementById("modalShortReason").textContent = shortTableReason(alertObj);
  document.getElementById("modalDisplayLabelReason").textContent =
    alertObj.display_label_reason ?? "-";

  document.getElementById("modalInterpretation").textContent =
    alertObj.interpretation ?? "-";

  document.getElementById("modalSummary").textContent =
    alertObj.summary ?? "-";

  document.getElementById("modalModelFlag").textContent =
    alertObj.raw_model_flag ? "YES" : "NO";

  document.getElementById("modalScore").textContent =
    safeNum(alertObj.ae_score, 6, "-");

  document.getElementById("modalRawSeverity").textContent =
    alertObj.raw_severity ?? "-";

  document.getElementById("modalFinalSeverity").textContent =
    alertObj.final_severity ?? alertObj.severity ?? "-";

  document.getElementById("modalAdjustmentReason").textContent =
    alertObj.adjustment_reason ?? "-";

  document.getElementById("modalPossibleExplanation").textContent =
    showHints ? (alertObj.possible_explanation ?? "-") : "-";

  document.getElementById("modalWhatToCheck").textContent =
    showHints ? (alertObj.what_to_check ?? "-") : "-";

  document.getElementById("modalSource").textContent =
    `${alertObj.src_ip ?? "-"} : ${alertObj.src_port ?? "-"}`;

  document.getElementById("modalDestination").textContent =
    `${alertObj.dest_ip ?? "-"} : ${alertObj.dest_port ?? "-"}`;

  document.getElementById("modalProto").textContent =
    `${alertObj.proto ?? "-"} / ${alertObj.app_proto ?? "-"}`;

  document.getElementById("modalClass").textContent =
    alertObj.traffic_class ?? "-";

  document.getElementById("modalContextTags").textContent =
    Array.isArray(alertObj.context_tags) && alertObj.context_tags.length
      ? alertObj.context_tags.join(", ")
      : "-";

  document.getElementById("modalRepeatLevel").textContent =
    alertObj.repeat_level ?? "-";

  document.getElementById("modalRepeatCount").textContent =
    String(alertObj.repeat_count ?? "-");

  document.getElementById("modalRepeatWindow").textContent =
    `${alertObj.repeat_window_s ?? "-"} s`;

  document.getElementById("modalTrafficNote").textContent =
    alertObj.traffic_note ?? "-";

  document.getElementById("modalLikelyBenign").textContent =
    alertObj.likely_benign ? "YES" : "NO";

  document.getElementById("modalBenignReason").textContent =
    alertObj.benign_reason ?? "-";

  document.getElementById("modalRepeatKey").textContent = repKey;

  document.getElementById("modalTopFeatures").innerHTML =
    renderTopFeatures(alertObj.top_features);

  document.getElementById("modalInferMs").textContent =
    `${safeNum(alertObj.timing?.infer_ms, 3, "-")} ms`;

  document.getElementById("modalTotalMs").textContent =
    `${safeNum(alertObj.timing?.total_ms, 3, "-")} ms`;

  document.getElementById("modalThroughput").textContent =
    `${safeNum(alertObj.timing?.throughput_fps, 2, "-")} fps`;

  document.getElementById("modalSystem").textContent =
    `CPU=${safeNum(alertObj.system?.cpu_proc_pct, 1, "-")}% | RSS=${safeNum(alertObj.system?.rss_mb, 1, "-")} MB`;

  document.getElementById("modalExplanation").textContent =
    alertObj.explanation ?? "-";

  document.getElementById("modalBackdrop").classList.add("show");
  document.getElementById("explanationModal").classList.add("show");
}

function closeExplanationModal() {
  document.getElementById("modalBackdrop").classList.remove("show");
  document.getElementById("explanationModal").classList.remove("show");
}

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    closeExplanationModal();
  }
});

async function fetchText(url) {
  const r = await fetch(url);
  return await r.text();
}

function parsePrometheus(text) {
  const out = {};
  const lines = text.split("\n");

  for (const line of lines) {
    if (!line || line.startsWith("#")) continue;

    const parts = line.split(" ");
    if (parts.length < 2) continue;

    const name = parts[0];
    const value = parseFloat(parts[1]);

    if (!isNaN(value)) {
      out[name] = value;
    }
  }

  return out;
}

async function refresh() {
  const limit = parseInt(document.getElementById("limit").value || "50", 10);

  const [recentRes, alertsRes] = await Promise.all([
    fetch(`/recent?limit=${limit}`),
    fetch(`/alerts?limit=${limit}`)
  ]);

  const recentData = await recentRes.json();
  const alertsData = await alertsRes.json();

  const b = recentData.bands;
  document.getElementById("thr").textContent =
    b ? `ok=${b.ok.toFixed(6)} warn=${b.warn.toFixed(6)} crit=${b.crit.toFixed(6)}` : "not set";

  document.getElementById("buf").textContent = (alertsData.alerts || []).length;

  const rows = recentData.recent || [];
  latestRows = rows;

  if (!rows.length) {
    setStatusPillLabel("OK");
  } else {
    const topLabel = rows[0].final_label || rows[0].display_label || "OK";
    setStatusPillLabel(topLabel);
  }

  const mText = await fetchText("/metrics");
  const m = parsePrometheus(mText);

  if (m["rtids_cpu_process_pct"] !== undefined) {
    document.getElementById("cpu").textContent =
      m["rtids_cpu_process_pct"].toFixed(1) + "%";
  }

  if (m["rtids_rss_mb"] !== undefined) {
    document.getElementById("rss").textContent =
      m["rtids_rss_mb"].toFixed(1) + " MB";
  }

  if (m["rtids_throughput_fps"] !== undefined) {
    document.getElementById("fps").textContent =
      m["rtids_throughput_fps"].toFixed(2);
  }

  const tbody = document.getElementById("rows");
  tbody.innerHTML = "";

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="muted">No traffic yet.</td></tr>`;
  } else {
    for (let idx = 0; idx < rows.length; idx++) {
      const a = rows[idx];

      const t = new Date((a.ts_unix ?? 0) * 1000)
        .toISOString()
        .replace("T", " ")
        .replace("Z", "Z")
        .split(".")[0] + "Z";

      const source = `${a.src_ip ?? "-"}:${a.src_port ?? "-"}`;
      const destination = `${a.dest_ip ?? "-"}:${a.dest_port ?? "-"}`;
      const proto = a.proto || "-";
      const finalLabel = a.final_label || a.display_label || "-";
      const summary = shortTableReason(a);
      const repeatCount = a.repeat_count ?? 0;

      tbody.innerHTML += `
        <tr
          class="${rowClassFromLabel(finalLabel)} clickable-row"
          onclick="openExplanationModal(latestRows[${idx}])"
          title="Open alert explanation"
        >
          <td class="mono compact">${escapeHtml(t)}</td>
          <td class="mono compact">${escapeHtml(source)}</td>
          <td class="mono compact">${escapeHtml(destination)}</td>
          <td class="center">${protoBadge(proto)}</td>
          <td class="center"><span class="mono">${escapeHtml(String(repeatCount))}</span></td>
          <td class="center">${displayLabelBadge(finalLabel)}</td>
          <td class="summary-cell">${safeText(summary)}</td>
          <td class="center details-sticky">
            <button
              class="info-btn"
              onclick="event.stopPropagation(); openExplanationModal(latestRows[${idx}])"
              title="Show details"
            >i</button>
          </td>
        </tr>
      `;
    }
  }

  document.getElementById("lastUpdate").textContent =
    "Updated: " + new Date().toLocaleTimeString();
}

function toggle() {
  running = !running;

  const btn = document.getElementById("toggleBtn");
  btn.textContent = running ? "Pause" : "Resume";

  if (running) {
    refresh();
    timer = setInterval(refresh, 2000);
  } else {
    clearInterval(timer);
    timer = null;
  }
}

async function clearAlerts() {
  await fetch("/alerts/clear", { method: "POST" });
  refresh();
}

timer = setInterval(refresh, 2000);
refresh();