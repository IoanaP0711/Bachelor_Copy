let alertsRows = [];
let activeAlertFilter = "ALL";
let alertsRefreshTimer = null;

function getFinalLabel(alertObj) {
  return String(
    alertObj.final_label ||
    alertObj.display_label ||
    alertObj.severity ||
    ""
  ).toUpperCase();
}

function setAlertFilter(label) {
  activeAlertFilter = String(label || "ALL").toUpperCase();

  document.querySelectorAll(".filter-btn").forEach((btn) => {
    const btnFilter = String(btn.dataset.filter || "ALL").toUpperCase();
    btn.classList.toggle("active", btnFilter === activeAlertFilter);
  });

  renderAlertsTable();
}

function getFilteredAlerts() {
  if (activeAlertFilter === "ALL") {
    return alertsRows;
  }

  return alertsRows.filter((alertObj) => {
    return getFinalLabel(alertObj) === activeAlertFilter;
  });
}

function renderAlertsTable() {
  const tbody = document.getElementById("rows");
  const rows = getFilteredAlerts();

  tbody.innerHTML = "";

  if (!alertsRows.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="muted">No alerts stored yet.</td>
      </tr>
    `;
    return;
  }

  if (!rows.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="muted">
          No alerts match the selected filter: ${safeText(activeAlertFilter)}.
        </td>
      </tr>
    `;
    return;
  }

  for (const a of rows) {
    const t = new Date((a.ts_unix ?? 0) * 1000)
      .toISOString()
      .replace("T", " ")
      .replace("Z", "Z")
      .split(".")[0] + "Z";

    const source = `${a.src_ip ?? "-"}:${a.src_port ?? "-"}`;
    const destination = `${a.dest_ip ?? "-"}:${a.dest_port ?? "-"}`;
    const proto = a.proto || "-";
    const trafficClass = a.traffic_class || "-";
    const repeatCount = a.repeat_count ?? 0;
    const finalLabel = getFinalLabel(a) || "-";
    const rawSeverity = a.raw_severity || "-";
    const score = safeNum(a.ae_score, 6, "-");

    /*
      The alerts page stays more explanatory than the main dashboard.
      This does not change backend data, inference, or severity logic.
    */
    const summary = a.summary || a.interpretation || "-";

    tbody.innerHTML += `
      <tr class="${rowClassFromLabel(finalLabel)}">
        <td class="mono compact">${escapeHtml(t)}</td>
        <td class="mono compact">${escapeHtml(source)}</td>
        <td class="mono compact">${escapeHtml(destination)}</td>
        <td class="center">${protoBadge(proto)}</td>
        <td class="compact">${safeText(trafficClass)}</td>
        <td class="center"><span class="mono">${safeText(repeatCount)}</span></td>
        <td class="center">${displayLabelBadge(finalLabel)}</td>
        <td class="center">${safeText(rawSeverity)}</td>
        <td class="mono">${safeText(score)}</td>
        <td class="summary-cell">${safeText(summary)}</td>
      </tr>
    `;
  }
}

async function loadAlerts() {
  const limit = parseInt(document.getElementById("limit").value || "50", 10);

  try {
    const res = await fetch(`/alerts?limit=${limit}`);
    const data = await res.json();

    alertsRows = data.alerts || [];
    renderAlertsTable();

    document.getElementById("lastUpdate").textContent =
      "Updated: " + new Date().toLocaleTimeString();
  } catch (err) {
    const tbody = document.getElementById("rows");
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="muted">
          Could not load alerts.
        </td>
      </tr>
    `;

    document.getElementById("lastUpdate").textContent =
      "Load failed: " + new Date().toLocaleTimeString();
  }
}

async function clearAlertsUi() {
  await fetch("/alerts/clear", { method: "POST" });

  alertsRows = [];
  renderAlertsTable();

  document.getElementById("lastUpdate").textContent =
    "Updated: " + new Date().toLocaleTimeString();
}

loadAlerts();

alertsRefreshTimer = setInterval(() => {
  loadAlerts();
}, 3000);