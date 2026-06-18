function recentEndpoint(ipValue, portValue) {
  const ip = String(ipValue ?? "-");
  const port = String(portValue ?? "-");

  if (ip.includes(":") && ip !== "-") {
    return `[${ip}]:${port}`;
  }

  return `${ip}:${port}`;
}

function renderRecentMobileCard(row) {
  const finalLabel =
    row.final_label ||
    row.display_label ||
    "";

  const source = recentEndpoint(
    row.src_ip,
    row.src_port
  );

  const destination = recentEndpoint(
    row.dest_ip,
    row.dest_port
  );

  const protocol =
    row.proto ||
    row.protocol ||
    "-";

  const trafficClass =
    row.traffic_class ||
    "-";

  const labelHtml = finalLabel
    ? displayLabelBadge(finalLabel)
    : `
      <span class="badge badge-muted">
        No final label
      </span>
    `;

  return `
    <article
      class="mobile-data-card recent-mobile-card ${
        finalLabel
          ? rowClassFromLabel(finalLabel)
          : ""
      }"
    >
      <div class="mobile-card-header">
        ${labelHtml}

        <time class="mobile-card-time mono">
          ${safeText(formatTime(row.ts_unix))}
        </time>
      </div>

      <div class="mobile-card-flow">
        <span class="mono">
          ${safeText(source)}
        </span>

        <span class="flow-arrow" aria-hidden="true">
          →
        </span>

        <span class="mono">
          ${safeText(destination)}
        </span>
      </div>

      <dl class="mobile-card-details-grid">
        <div>
          <dt>Protocol / class</dt>

          <dd>
            <span class="mobile-protocol-row">
              ${protoBadge(protocol)}
              <span>${safeText(trafficClass)}</span>
            </span>
          </dd>
        </div>

        <div>
          <dt>Final label</dt>
          <dd>${labelHtml}</dd>
        </div>
      </dl>
    </article>
  `;
}

async function loadRecent() {
  const limitInput =
    document.getElementById("limit");

  const tbody =
    document.getElementById("rows");

  const mobileList =
    document.getElementById("mobileRecentList");

  let limit = parseInt(
    limitInput?.value || "50",
    10
  );

  if (!Number.isFinite(limit)) {
    limit = 50;
  }

  limit = Math.max(
    1,
    Math.min(limit, 300)
  );

  if (limitInput) {
    limitInput.value = String(limit);
  }

  try {
    const res = await fetch(
      `/recent?limit=${encodeURIComponent(limit)}`
    );

    if (!res.ok) {
      throw new Error(
        `Recent request failed with status ${res.status}`
      );
    }

    const data = await res.json();

    const rows = Array.isArray(data.recent)
      ? data.recent
      : [];

    if (!rows.length) {
      tbody.innerHTML = `
        <tr>
          <td colspan="9" class="muted">
            No recent traffic yet.
          </td>
        </tr>
      `;

      mobileList.innerHTML = `
        <div class="mobile-empty-state">
          No recent traffic yet.
        </div>
      `;

      setLastUpdate();
      return;
    }

    tbody.innerHTML = rows
      .map((row) => {
        const finalLabel =
          row.final_label ||
          row.display_label ||
          "-";

        const source = recentEndpoint(
          row.src_ip,
          row.src_port
        );

        const destination = recentEndpoint(
          row.dest_ip,
          row.dest_port
        );

        const summary =
          shortTableReason(row);

        return `
          <tr class="${rowClassFromLabel(finalLabel)}">
            <td class="mono compact">
              ${safeText(formatTime(row.ts_unix))}
            </td>

            <td class="mono compact">
              ${safeText(source)}
            </td>

            <td class="mono compact">
              ${safeText(destination)}
            </td>

            <td class="center">
              ${protoBadge(row.proto)}
            </td>

            <td class="mono compact">
              ${safeText(row.traffic_class)}
            </td>

            <td class="center mono">
              ${safeText(row.repeat_count ?? 0)}
            </td>

            <td class="center">
              ${displayLabelBadge(finalLabel)}
            </td>

            <td class="center mono">
              ${safeText(row.raw_severity)}
            </td>

            <td class="summary-cell">
              ${safeText(summary)}
            </td>
          </tr>
        `;
      })
      .join("");

    mobileList.innerHTML = rows
      .map(renderRecentMobileCard)
      .join("");

    setLastUpdate();
  } catch (err) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" class="muted">
          Failed to load recent traffic.
        </td>
      </tr>
    `;

    mobileList.innerHTML = `
      <div class="mobile-empty-state alerts-load-error">
        Failed to load recent traffic.
      </div>
    `;

    console.error(err);
  }
}

loadRecent();

setInterval(
  loadRecent,
  3000
);