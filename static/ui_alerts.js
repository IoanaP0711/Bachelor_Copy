let alertsRows = [];
let activeAlertFilter = "ALL";
let alertSearchText = "";
let alertsRefreshTimer = null;

function getFinalLabel(alertObj) {
  return String(
    getPreviewFinalDecision(alertObj) || "-"
  ).toUpperCase();
}

function getHumanStatus(alertObj) {
  const rawStatus =
    alertObj?.human_status?.status ??
    alertObj?.human_status ??
    "New";

  const normalised = String(rawStatus || "New")
    .trim()
    .toLowerCase();

  const statuses = {
    "new": "New",
    "seen": "Seen",
    "under review": "Under review",
    "under_review": "Under review",
    "under-review": "Under review",
    "resolved": "Resolved",
    "false positive": "False positive",
    "false_positive": "False positive",
    "false-positive": "False positive",
  };

  return statuses[normalised] || "New";
}

function humanStatusClass(status) {
  return getHumanStatus({
    human_status: status,
  })
    .toLowerCase()
    .replaceAll(" ", "-");
}

function renderHumanStatusBadge(status) {
  const value = getHumanStatus({
    human_status: status,
  });

  return `
    <span
      class="human-status-badge human-status-${humanStatusClass(value)}"
      title="Human review status"
    >
      ${escapeHtml(value)}
    </span>
  `;
}

function getAlertTimestampMs(alertObj) {
  const rawValue =
    alertObj?.ts_unix ??
    alertObj?.timestamp ??
    alertObj?.ts ??
    alertObj?.created_at ??
    0;

  const numericValue = Number(rawValue);

  if (
    Number.isFinite(numericValue) &&
    numericValue > 0
  ) {
    return numericValue > 1_000_000_000_000
      ? numericValue
      : numericValue * 1000;
  }

  const parsedValue = Date.parse(
    String(rawValue || "")
  );

  return Number.isNaN(parsedValue)
    ? 0
    : parsedValue;
}

function formatAlertTime(alertObj) {
  const timestampMs =
    getAlertTimestampMs(alertObj);

  if (!timestampMs) {
    return "-";
  }

  return (
    new Date(timestampMs)
      .toISOString()
      .replace("T", " ")
      .split(".")[0] + "Z"
  );
}

function getAlertIp(alertObj, side) {
  if (side === "source") {
    return (
      alertObj.src_ip ??
      alertObj.source_ip ??
      "-"
    );
  }

  return (
    alertObj.dest_ip ??
    alertObj.dst_ip ??
    alertObj.destination_ip ??
    "-"
  );
}

function getAlertPort(alertObj, side) {
  if (side === "source") {
    return (
      alertObj.src_port ??
      alertObj.source_port ??
      "-"
    );
  }

  return (
    alertObj.dest_port ??
    alertObj.dst_port ??
    alertObj.destination_port ??
    "-"
  );
}

function formatEndpoint(ipValue, portValue) {
  const ip = String(ipValue ?? "-");
  const port = String(portValue ?? "-");

  if (ip === "-") {
    return `-:${port}`;
  }

  if (ip.includes(":")) {
    return `[${ip}]:${port}`;
  }

  return `${ip}:${port}`;
}

function getAlertProcessName(alertObj) {
  return String(
    alertObj?.process_name ||
    "Unknown"
  );
}


function getAlertProcessConfidence(alertObj) {
  return String(
    alertObj
      ?.process_attribution_confidence ||
    "none"
  )
    .trim()
    .toLowerCase();
}


function renderProcessCell(alertObj) {
  const processName =
    getAlertProcessName(alertObj);

  const confidence =
    getAlertProcessConfidence(alertObj);

  const processPid =
    alertObj?.process_pid;

  const attribution =
    String(
      alertObj?.process_attribution ||
      "not_found"
    );

  const subtitle =
    processName === "Unknown"
      ? "No socket match"
      : (
          processPid !== null &&
          processPid !== undefined
            ? `PID ${processPid}`
            : attribution.replaceAll(
                "_",
                " "
              )
        );

  const confidenceLabel =
    confidence === "high"
      ? "High"
      : confidence === "medium"
        ? "Medium"
        : "None";

  return `
    <div class="process-cell-content">
      <strong
        class="process-name-text"
        title="${escapeHtml(processName)}"
      >
        ${escapeHtml(processName)}
      </strong>

      <span class="process-cell-subtitle">
        ${escapeHtml(subtitle)}
      </span>

      <span
        class="process-confidence-inline process-confidence-${escapeHtml(
          confidence
        )}"
      >
        ${escapeHtml(confidenceLabel)}
      </span>
    </div>
  `;
}

function getAlertReason(alertObj) {
  return String(
    alertObj.short_reason ||
    alertObj.display_label_reason ||
    alertObj.adjustment_reason ||
    alertObj.summary ||
    alertObj.interpretation ||
    shortTableReason(alertObj) ||
    "-"
  );
}

function getAlertSearchText(alertObj) {
  return [
    getAlertIp(alertObj, "source"),
    getAlertIp(alertObj, "destination"),
    alertObj.proto,
    alertObj.protocol,
    alertObj.traffic_class,
    alertObj.process_name,
    alertObj.process_pid,
    alertObj.process_exe,
    alertObj.process_attribution,
    getHumanStatus(alertObj),
    getAlertReason(alertObj),
  ]
    .filter((value) => {
      return (
        value !== null &&
        value !== undefined
      );
    })
    .join(" ")
    .toLowerCase();
}

function setAlertFilter(label) {
  activeAlertFilter =
    String(label || "ALL").toUpperCase();

  document
    .querySelectorAll(".filter-btn")
    .forEach((button) => {
      const buttonFilter = String(
        button.dataset.filter || "ALL"
      ).toUpperCase();

      button.classList.toggle(
        "active",
        buttonFilter === activeAlertFilter
      );
    });

  renderAlertsTable();
}

function setAlertSearch(value) {
  alertSearchText = String(value || "")
    .trim()
    .toLowerCase();

  renderAlertsTable();
}

function clearAlertSearch() {
  const searchInput =
    document.getElementById("alertSearch");

  if (searchInput) {
    searchInput.value = "";
  }

  alertSearchText = "";

  renderAlertsTable();
}

function getFilteredAlerts() {
  return alertsRows.filter((alertObj) => {
    const matchesFilter =
      activeAlertFilter === "ALL" ||
      getFinalLabel(alertObj) ===
        activeAlertFilter;

    const matchesSearch =
      !alertSearchText ||
      getAlertSearchText(alertObj).includes(
        alertSearchText
      );

    return matchesFilter && matchesSearch;
  });
}

function updateResultsSummary(visibleCount) {
  const summary =
    document.getElementById(
      "resultsSummary"
    );

  if (!summary) {
    return;
  }

  const filterText =
    activeAlertFilter === "ALL"
      ? "all decisions"
      : activeAlertFilter;

  const searchText = alertSearchText
    ? ` matching “${alertSearchText}”`
    : "";

  summary.textContent =
    `Showing ${visibleCount} of ` +
    `${alertsRows.length} alerts` +
    ` · ${filterText}${searchText}`;
}

function renderFlowCell(
  alertObj,
  originalIndex
) {
  const source = formatEndpoint(
    getAlertIp(alertObj, "source"),
    getAlertPort(alertObj, "source")
  );

  const destination = formatEndpoint(
    getAlertIp(alertObj, "destination"),
    getAlertPort(alertObj, "destination")
  );

  return `
    <div
      class="flow-route clickable-flow"
      role="button"
      tabindex="0"
      aria-label="Open quick preview for flow from ${escapeHtml(
        source
      )} to ${escapeHtml(destination)}"
      title="Click to open quick preview"
      onclick="openAlertPreviewModal(
        alertsRows[${originalIndex}]
      )"
      onkeydown="
        if (
          event.key === 'Enter' ||
          event.key === ' '
        ) {
          event.preventDefault();
          openAlertPreviewModal(
            alertsRows[${originalIndex}]
          );
        }
      "
    >
      <span
        class="endpoint-text mono"
        title="${escapeHtml(source)}"
      >
        ${escapeHtml(source)}
      </span>

      <span
        class="flow-arrow"
        aria-hidden="true"
      >
        →
      </span>

      <span
        class="endpoint-text mono"
        title="${escapeHtml(destination)}"
      >
        ${escapeHtml(destination)}
      </span>
    </div>
  `;
}

function renderProtocolClassCell(alertObj) {
  const protocol =
    alertObj.proto ||
    alertObj.protocol ||
    "-";

  const trafficClass =
    alertObj.traffic_class ||
    "-";

  return `
    <div class="protocol-class-cell">
      ${protoBadge(protocol)}

      <span
        class="traffic-class-text"
        title="${escapeHtml(trafficClass)}"
      >
        ${safeText(trafficClass)}
      </span>
    </div>
  `;
}

function renderActionCell(
  alertObj,
  originalIndex
) {
  const stableId =
    getAlertStableId(alertObj);

  const hasDetails =
    stableId &&
    stableId !== "-";

  const detailsControl = hasDetails
    ? `
      <a
        class="table-action-btn table-details-btn"
        href="/ui/alerts/${encodeURIComponent(
          String(stableId)
        )}"
        title="Open full alert details"
      >
        Details
      </a>
    `
    : `
      <span
        class="table-action-btn table-details-btn disabled-link"
        title="Full details are unavailable for this alert"
      >
        Details
      </span>
    `;

  return `
    <div class="table-actions">
      <button
        type="button"
        class="table-action-btn table-quick-btn"
        onclick="openAlertPreviewModal(
          alertsRows[${originalIndex}]
        )"
        title="Open quick preview"
      >
        Quick view
      </button>

      ${detailsControl}
    </div>
  `;
}

function renderAlertMobileCard(
  alertObj,
  originalIndex
) {
  const finalLabel =
    getFinalLabel(alertObj);

  const rawSeverity =
    getPreviewRawSeverity(alertObj);

  const reason =
    getAlertReason(alertObj);

  const formattedTime =
    formatAlertTime(alertObj);

  const source = formatEndpoint(
    getAlertIp(alertObj, "source"),
    getAlertPort(alertObj, "source")
  );

  const destination = formatEndpoint(
    getAlertIp(alertObj, "destination"),
    getAlertPort(
      alertObj,
      "destination"
    )
  );

  const protocol =
    alertObj.proto ||
    alertObj.protocol ||
    "-";

  const trafficClass =
    alertObj.traffic_class ||
    "-";

  const stableId =
    getAlertStableId(alertObj);

  const detailsControl =
    stableId && stableId !== "-"
      ? `
        <a
          class="mobile-card-action mobile-card-details"
          href="/ui/alerts/${encodeURIComponent(
            String(stableId)
          )}"
        >
          Details
        </a>
      `
      : `
        <span
          class="mobile-card-action mobile-card-details disabled-link"
          title="Full details are unavailable for this alert"
        >
          Details
        </span>
      `;

  return `
    <article
      class="mobile-data-card alert-mobile-card ${rowClassFromLabel(
        finalLabel
      )}"
    >
      <div class="mobile-card-header">
        ${renderSeverityBadge(finalLabel)}

        <time class="mobile-card-time mono">
          ${escapeHtml(formattedTime)}
        </time>
      </div>

      <p class="mobile-card-summary">
        ${escapeHtml(reason)}
      </p>

      <div
        class="mobile-card-flow clickable-flow"
        role="button"
        tabindex="0"
        aria-label="Open quick preview for flow from ${escapeHtml(
          source
        )} to ${escapeHtml(destination)}"
        onclick="openAlertPreviewModal(
          alertsRows[${originalIndex}]
        )"
        onkeydown="
          if (
            event.key === 'Enter' ||
            event.key === ' '
          ) {
            event.preventDefault();
            openAlertPreviewModal(
              alertsRows[${originalIndex}]
            );
          }
        "
      >
        <span class="mono">
          ${escapeHtml(source)}
        </span>

        <span
          class="flow-arrow"
          aria-hidden="true"
        >
          →
        </span>

        <span class="mono">
          ${escapeHtml(destination)}
        </span>
      </div>

      <dl class="mobile-card-details-grid">
        <div>
          <dt>Protocol / class</dt>

          <dd>
            <span class="mobile-protocol-row">
              ${protoBadge(protocol)}

              <span>
                ${safeText(trafficClass)}
              </span>
            </span>
          </dd>
        </div>

        <div>
          <dt>Local application</dt>

          <dd>
            <strong>
              ${escapeHtml(
                getAlertProcessName(alertObj)
              )}
            </strong>

            ${
              getAlertProcessConfidence(
                alertObj
              ) === "high"
                ? `
                  <span
                    class="process-confidence-inline process-confidence-high"
                  >
                    Exact socket match
                  </span>
                `
                : `
                  <span
                    class="process-confidence-inline process-confidence-none"
                  >
                    No match
                  </span>
                `
            }
          </dd>
        </div>

        <div>
          <dt>Raw → Final</dt>

          <dd>
            ${renderRawFinalBadge(
              rawSeverity,
              finalLabel
            )}
          </dd>
        </div>

        <div>
          <dt>Human status</dt>

          <dd>
            ${renderHumanStatusBadge(
              getHumanStatus(alertObj)
            )}
          </dd>
        </div>

        <div>
          <dt>Timestamp</dt>

          <dd class="mono">
            ${escapeHtml(formattedTime)}
          </dd>
        </div>
      </dl>

      <div class="mobile-card-actions">
        <button
          type="button"
          class="mobile-card-action mobile-card-quick"
          onclick="openAlertPreviewModal(
            alertsRows[${originalIndex}]
          )"
        >
          Quick view
        </button>

        ${detailsControl}
      </div>
    </article>
  `;
}

function renderAlertsTable() {
  const tbody =
    document.getElementById("rows");

  const mobileList =
    document.getElementById(
      "mobileAlertsList"
    );

  if (!tbody || !mobileList) {
    return;
  }

  const rows = getFilteredAlerts();

  updateResultsSummary(rows.length);

  if (!alertsRows.length) {
    tbody.innerHTML = `
      <tr>
        <td
          colspan="9"
          class="alerts-empty-state"
        >
          No alerts stored yet.
        </td>
      </tr>
    `;

    mobileList.innerHTML = `
      <div class="mobile-empty-state">
        No alerts stored yet.
      </div>
    `;

    return;
  }

  if (!rows.length) {
    tbody.innerHTML = `
      <tr>
        <td
          colspan="9"
          class="alerts-empty-state"
        >
          No alerts match the current
          filter and search.
        </td>
      </tr>
    `;

    mobileList.innerHTML = `
      <div class="mobile-empty-state">
        No alerts match the current
        filter and search.
      </div>
    `;

    return;
  }

  tbody.innerHTML = rows
    .map((alertObj) => {
      const originalIndex =
        alertsRows.indexOf(alertObj);

      const finalLabel =
        getFinalLabel(alertObj);

      const rawSeverity =
        getPreviewRawSeverity(alertObj);

      const reason =
        getAlertReason(alertObj);

      const formattedTime =
        formatAlertTime(alertObj);

      return `
        <tr
          class="alert-table-row ${rowClassFromLabel(
            finalLabel
          )}"
        >
          <td
            class="time-cell mono"
            title="${escapeHtml(
              formattedTime
            )}"
          >
            ${escapeHtml(formattedTime)}
          </td>

          <td class="severity-cell">
            ${renderSeverityBadge(
              finalLabel
            )}
          </td>

          <td class="human-status-cell">
            ${renderHumanStatusBadge(
              getHumanStatus(alertObj)
            )}
          </td>

          <td class="flow-cell">
            ${renderFlowCell(
              alertObj,
              originalIndex
            )}
          </td>

          <td class="protocol-cell">
          ${renderProtocolClassCell(
            alertObj
          )}
        </td>

        <td class="process-cell">
          ${renderProcessCell(
            alertObj
          )}
        </td>

        <td class="raw-final-cell">
          ${renderRawFinalBadge(
            rawSeverity,
            finalLabel
          )}
        </td>


        <td
          class="reason-cell"
          title="${escapeHtml(reason)}"
        >
          <div class="reason-cell-text">
            ${escapeHtml(reason)}
          </div>
        </td>

        <td class="action-cell">
          ${renderActionCell(
            alertObj,
            originalIndex
          )}
        </td>
                </tr>
              `;
            })
            .join("");

  mobileList.innerHTML = rows
    .map((alertObj) => {
      const originalIndex =
        alertsRows.indexOf(alertObj);

      return renderAlertMobileCard(
        alertObj,
        originalIndex
      );
    })
    .join("");
}

async function loadAlerts() {
  const limitInput =
    document.getElementById("limit");

  let limit = parseInt(
    limitInput?.value || "50",
    10
  );

  if (!Number.isFinite(limit)) {
    limit = 50;
  }

  limit = Math.max(
    1,
    Math.min(limit, 200)
  );

  if (limitInput) {
    limitInput.value =
      String(limit);
  }

  try {
    const response = await fetch(
      `/alerts?limit=${encodeURIComponent(
        limit
      )}`
    );

    if (!response.ok) {
      throw new Error(
        "Alerts request failed with " +
        `status ${response.status}`
      );
    }

    const data =
      await response.json();

    alertsRows = Array.isArray(
      data.alerts
    )
      ? data.alerts
          .slice()
          .sort((left, right) => {
            return (
              getAlertTimestampMs(right) -
              getAlertTimestampMs(left)
            );
          })
      : [];

    renderAlertsTable();

    const lastUpdate =
      document.getElementById(
        "lastUpdate"
      );

    if (lastUpdate) {
      lastUpdate.textContent =
        "Updated: " +
        new Date().toLocaleTimeString();
    }
  } catch (error) {
    console.error(
      "Could not load alerts:",
      error
    );

    const tbody =
      document.getElementById("rows");

    if (tbody) {
      tbody.innerHTML = `
        <tr>
          <td
            colspan="9"
            class="alerts-empty-state alerts-load-error"
          >
            Could not load alerts.
          </td>
        </tr>
      `;
    }

    const mobileList =
      document.getElementById(
        "mobileAlertsList"
      );

    if (mobileList) {
      mobileList.innerHTML = `
        <div
          class="mobile-empty-state alerts-load-error"
        >
          Could not load alerts.
        </div>
      `;
    }

    updateResultsSummary(0);

    const lastUpdate =
      document.getElementById(
        "lastUpdate"
      );

    if (lastUpdate) {
      lastUpdate.textContent =
        "Load failed: " +
        new Date().toLocaleTimeString();
    }
  }
}

async function clearAlertsUi() {
  try {
    const response = await fetch(
      "/alerts/clear",
      {
        method: "POST",
      }
    );

    if (!response.ok) {
      throw new Error(
        "Clear request failed with " +
        `status ${response.status}`
      );
    }

    alertsRows = [];

    renderAlertsTable();

    const lastUpdate =
      document.getElementById(
        "lastUpdate"
      );

    if (lastUpdate) {
      lastUpdate.textContent =
        "Updated: " +
        new Date().toLocaleTimeString();
    }
  } catch (error) {
    console.error(
      "Could not clear alerts:",
      error
    );

    const lastUpdate =
      document.getElementById(
        "lastUpdate"
      );

    if (lastUpdate) {
      lastUpdate.textContent =
        "Clear failed: " +
        new Date().toLocaleTimeString();
    }
  }
}

const alertSearchInput =
  document.getElementById(
    "alertSearch"
  );

if (alertSearchInput) {
  alertSearchInput.addEventListener(
    "input",
    (event) => {
      setAlertSearch(
        event.target.value
      );
    }
  );
}

loadAlerts();

alertsRefreshTimer = setInterval(
  () => {
    loadAlerts();
  },
  3000
);