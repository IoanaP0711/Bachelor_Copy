let running = true;
let timer = null;
let latestImportantAlerts = [];

const DASHBOARD_REFRESH_MS = 2000;
const DASHBOARD_RECENT_LIMIT = 300;
const DASHBOARD_ALERT_LIMIT = 200;
const IMPORTANT_ALERT_LIMIT = 5;
const DASHBOARD_DETAIL_PAGE_EXISTS = true;

function dashboardEscapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function normaliseFinalDecision(value) {
  const v = String(value || "").trim().toUpperCase();

  if (v === "CRITICAL" || v === "CRIT") return "CRITICAL";
  if (v === "REVIEW" || v === "WARN" || v === "MED") return "REVIEW";
  if (v === "BENIGN") return "BENIGN";
  if (v === "OK") return "OK";

  return "OK";
}

function getDashboardFinalDecision(alertObj) {
  if (typeof getPreviewFinalDecision === "function") {
    return normaliseFinalDecision(getPreviewFinalDecision(alertObj));
  }

  return normaliseFinalDecision(
    alertObj?.final_label ||
    alertObj?.display_label ||
    alertObj?.final_severity ||
    alertObj?.severity ||
    "OK"
  );
}

function getDashboardRawSeverity(alertObj) {
  if (typeof getPreviewRawSeverity === "function") {
    return String(getPreviewRawSeverity(alertObj) || "UNKNOWN").toUpperCase();
  }

  return String(alertObj?.raw_severity || alertObj?.severity || "UNKNOWN").toUpperCase();
}

function setStatusPillLabel(label) {
  const pill = document.getElementById("statusPill");
  if (!pill) return;

  const l = normaliseFinalDecision(label);

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

function dashboardBadge(value) {
  const v = String(value || "UNKNOWN").trim().toUpperCase();

  if (typeof renderSeverityBadge === "function") {
    return renderSeverityBadge(v);
  }

  const css = v === "CRIT" ? "critical" : v.toLowerCase();
  const safeCss = ["ok", "benign", "review", "critical", "warn", "med"].includes(css)
    ? css
    : "muted";

  return `<span class="badge badge-${safeCss}">${dashboardEscapeHtml(v)}</span>`;
}

function dashboardRawFinalBadge(raw, finalDecision) {
  if (typeof renderRawFinalBadge === "function") {
    return renderRawFinalBadge(raw, finalDecision);
  }

  const changed = String(raw || "").toUpperCase() !== String(finalDecision || "").toUpperCase();

  return `
    <span class="raw-final-field">
      <span class="raw-final-values">
        ${dashboardBadge(raw)}
        <span class="raw-final-arrow" aria-hidden="true">→</span>
        ${dashboardBadge(finalDecision)}
      </span>
      ${changed ? `<span class="context-adjusted-badge">Changed by context</span>` : ""}
    </span>
  `;
}

function dashboardRowClass(label) {
  if (typeof rowClassFromLabel === "function") {
    return rowClassFromLabel(label);
  }

  const l = normaliseFinalDecision(label).toLowerCase();
  return `row-${l}`;
}

function dashboardProtoBadge(proto) {
  const p = String(proto || "-").toUpperCase();

  if (typeof protoBadge === "function") {
    return protoBadge(p);
  }

  return `<span class="badge badge-gray">${dashboardEscapeHtml(p)}</span>`;
}

function dashboardShortReason(alertObj) {
  if (typeof shortTableReason === "function") {
    return shortTableReason(alertObj);
  }

  return (
    alertObj?.short_summary ||
    alertObj?.summary ||
    alertObj?.display_label_reason ||
    alertObj?.adjustment_reason ||
    alertObj?.simple_explanation ||
    alertObj?.explanation ||
    "-"
  );
}

function getDashboardTimestampMs(value) {
  const raw =
    value && typeof value === "object"
      ? value.ts_unix ?? value.timestamp ?? value.ts ?? value.time ?? null
      : value;

  if (raw === null || raw === undefined || raw === "") return 0;

  const numeric = Number(raw);

  if (Number.isFinite(numeric)) {
    if (numeric <= 0) return 0;

    // Backend usually sends seconds. If it is already milliseconds, keep it.
    return numeric > 1000000000000 ? numeric : numeric * 1000;
  }

  const parsed = Date.parse(String(raw));
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatDashboardTime(value) {
  const ms = getDashboardTimestampMs(value);
  if (!ms) return "-";

  return new Date(ms).toLocaleString();
}

function formatNumber(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

async function fetchJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }

  return await response.json();
}

function computeDecisionCounts(recentRows) {
  const counts = {
    OK: 0,
    BENIGN: 0,
    REVIEW: 0,
    CRITICAL: 0,
  };

  for (const row of recentRows) {
    const label = getDashboardFinalDecision(row);
    counts[label] = (counts[label] || 0) + 1;
  }

  return counts;
}

function computeCurrentSecurityState(counts, recentRows) {
  if (!recentRows.length) {
    return {
      label: "OK",
      detail: "No traffic events received yet.",
    };
  }

  if (counts.CRITICAL > 0) {
    return {
      label: "CRITICAL",
      detail: `${counts.CRITICAL} critical final decision(s) in the recent buffer.`,
    };
  }

  if (counts.REVIEW > 0) {
    return {
      label: "REVIEW",
      detail: `${counts.REVIEW} review final decision(s) in the recent buffer.`,
    };
  }

  if (counts.BENIGN > 0 && counts.OK === 0) {
    return {
      label: "BENIGN",
      detail: "Recent traffic is context-adjusted as benign.",
    };
  }

  return {
    label: "OK",
    detail: "No REVIEW or CRITICAL final decisions in the recent buffer.",
  };
}

function updateDecisionCounts(counts) {
  setText("countOk", String(counts.OK));
  setText("countBenign", String(counts.BENIGN));
  setText("countReview", String(counts.REVIEW));
  setText("countCritical", String(counts.CRITICAL));
}

function updateSystemSummary(stats) {
  setText("systemStatus", "ONLINE");

  const bands = stats?.bands;
  const systemDetail = bands
    ? `Raw bands loaded: OK < ${formatNumber(bands.ok, 6)}, WARN < ${formatNumber(bands.warn, 6)}, CRIT ≥ ${formatNumber(bands.crit, 6)}`
    : "Raw model bands not available.";

  setText("systemStatusDetail", systemDetail);
  setText("bufferedAlerts", String(stats?.alerts_buffered ?? 0));
  setText("throughput", `${formatNumber(stats?.throughput_fps, 2)} req/s`);
  setText("cpu", `${formatNumber(stats?.cpu_proc_pct, 1)}%`);
  setText("memory", `${formatNumber(stats?.rss_mb, 1)} MB`);
}

function updateCurrentState(state) {
  setText("currentSecurityState", state.label);
  setText("currentSecurityDetail", state.detail);
  setStatusPillLabel(state.label);

  const badge = document.getElementById("currentSecurityBadge");
  if (badge) {
    badge.className = `overview-state-badge state-${state.label.toLowerCase()}`;
    badge.textContent = state.label;
  }
}

function dashboardEndpoint(ip, port) {
  const safeIp =
    ip === null || ip === undefined || ip === ""
      ? "-"
      : String(ip);

  if (port === null || port === undefined || port === "") {
    return safeIp;
  }

  return `${safeIp}:${port}`;
}

function dashboardDetailsHref(alertObj) {
  let stableId = "";

  if (typeof getAlertStableId === "function") {
    stableId = getAlertStableId(alertObj);
  } else {
    stableId =
      alertObj?.alert_id ||
      alertObj?.id ||
      alertObj?.flow_id ||
      alertObj?.uid ||
      alertObj?.event_id ||
      alertObj?.ts_unix ||
      alertObj?.timestamp ||
      "";
  }

  if (
    !DASHBOARD_DETAIL_PAGE_EXISTS ||
    !stableId ||
    stableId === "-"
  ) {
    return "";
  }

  return `/ui/alerts/${encodeURIComponent(String(stableId))}`;
}

function renderImportantAlerts(alertRows) {
  const container = document.getElementById("importantAlertsList");
  if (!container) return;

  latestImportantAlerts = alertRows
    .filter((alertObj) => {
      const label = getDashboardFinalDecision(alertObj);
      return label === "REVIEW" || label === "CRITICAL";
    })
    .sort((a, b) => getDashboardTimestampMs(b) - getDashboardTimestampMs(a))
    .slice(0, IMPORTANT_ALERT_LIMIT);

  setText("importantAlertCount", String(latestImportantAlerts.length));

  if (!latestImportantAlerts.length) {
    container.innerHTML = `
      <div class="important-empty-state">
        No important alerts currently detected.
      </div>
    `;
    return;
  }

  container.innerHTML = latestImportantAlerts.map((alertObj, index) => {
    const finalDecision = getDashboardFinalDecision(alertObj);
    const time = formatDashboardTime(alertObj);
    const source = dashboardEndpoint(alertObj.src_ip, alertObj.src_port);
    const destination = dashboardEndpoint(alertObj.dest_ip, alertObj.dest_port);
    const reason = dashboardShortReason(alertObj);
    const detailsHref = dashboardDetailsHref(alertObj);

    const detailsButton = detailsHref
      ? `
        <a
          class="important-details-btn"
          href="${dashboardEscapeHtml(detailsHref)}"
          onclick="event.stopPropagation()"
        >
          Full details
        </a>
      `
      : "";

    return `
      <article
        class="important-alert-card ${dashboardRowClass(finalDecision)}"
        onclick="openDashboardImportantAlert(${index})"
        title="Open compact alert preview"
      >
        <div class="important-alert-main">
          <div class="important-alert-topline">
            ${dashboardBadge(finalDecision)}
            <span class="mono important-alert-time">${dashboardEscapeHtml(time)}</span>
          </div>

          <div class="important-alert-flow mono">
            ${dashboardEscapeHtml(source)} → ${dashboardEscapeHtml(destination)}
          </div>

          <p class="important-alert-reason">
            ${dashboardEscapeHtml(reason)}
          </p>
        </div>

        <div class="important-alert-actions">
          <button
            class="overview-small-btn"
            type="button"
            onclick="event.stopPropagation(); openDashboardImportantAlert(${index})"
          >
            Quick preview
          </button>

          ${detailsButton}
        </div>
      </article>
    `;
  }).join("");
}

function openDashboardImportantAlert(index) {
  const alertObj = latestImportantAlerts[index];

  if (!alertObj) return;

  if (typeof openAlertPreviewModal === "function") {
    openAlertPreviewModal(alertObj);
  } else {
    console.error("openAlertPreviewModal is not available. Check ui_common.js loading.");
  }
}

function openExplanationModal(alertObj) {
  if (typeof openAlertPreviewModal === "function") {
    openAlertPreviewModal(alertObj);
  } else {
    console.error("openAlertPreviewModal is not available. Check ui_common.js loading.");
  }
}

function closeExplanationModal() {
  if (typeof closeAlertPreviewModal === "function") {
    closeAlertPreviewModal();
  }
}

async function refresh() {
  try {
    const [statsData, recentData, alertsData] = await Promise.all([
      fetchJson("/stats"),
      fetchJson(`/recent?limit=${DASHBOARD_RECENT_LIMIT}`),
      fetchJson(`/alerts?limit=${DASHBOARD_ALERT_LIMIT}`),
    ]);

    const recentRows = Array.isArray(recentData?.recent) ? recentData.recent : [];
    const alertRows = Array.isArray(alertsData?.alerts) ? alertsData.alerts : [];
    const counts = computeDecisionCounts(recentRows);
    const state = computeCurrentSecurityState(counts, recentRows);

    updateSystemSummary(statsData);
    updateDecisionCounts(counts);
    updateCurrentState(state);
    renderImportantAlerts(alertRows);

    setText("lastUpdated", new Date().toLocaleTimeString());
    setText("dashboardError", "");
  } catch (error) {
    console.error("Dashboard refresh failed:", error);

    setText("systemStatus", "OFFLINE");
    setText("systemStatusDetail", "Dashboard data could not be loaded.");
    setText("dashboardError", "Dashboard refresh failed. Check that the backend is running and that you are still logged in.");
    setStatusPillLabel("OK");
  }
}

function toggle() {
  running = !running;

  const btn = document.getElementById("toggleBtn");
  if (btn) btn.textContent = running ? "Pause" : "Resume";

  if (running) {
    refresh();
    timer = setInterval(refresh, DASHBOARD_REFRESH_MS);
  } else {
    clearInterval(timer);
    timer = null;
  }
}

async function clearAlerts() {
  await fetch("/alerts/clear", { method: "POST" });
  refresh();
}

window.openDashboardImportantAlert = openDashboardImportantAlert;
window.openExplanationModal = openExplanationModal;
window.closeExplanationModal = closeExplanationModal;
window.toggle = toggle;
window.clearAlerts = clearAlerts;

timer = setInterval(refresh, DASHBOARD_REFRESH_MS);
refresh();