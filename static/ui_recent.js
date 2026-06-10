async function loadRecent() {
  const limit = parseInt(document.getElementById("limit").value || "50", 10);
  const tbody = document.getElementById("rows");

  try {
    const res = await fetch(`/recent?limit=${limit}`);
    const data = await res.json();
    const rows = data.recent || [];

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="muted">No recent traffic yet.</td></tr>`;
      setLastUpdate();
      return;
    }

    tbody.innerHTML = rows.map(a => {
      const finalLabel = a.final_label || a.display_label || "-";
      const source = `${a.src_ip ?? "-"}:${a.src_port ?? "-"}`;
      const destination = `${a.dest_ip ?? "-"}:${a.dest_port ?? "-"}`;
      const summary = shortTableReason(a);

      return `
        <tr class="${rowClassFromLabel(finalLabel)}">
          <td class="mono compact">${safeText(formatTime(a.ts_unix))}</td>
          <td class="mono compact">${safeText(source)}</td>
          <td class="mono compact">${safeText(destination)}</td>
          <td class="center">${protoBadge(a.proto)}</td>
          <td class="mono compact">${safeText(a.traffic_class)}</td>
          <td class="center mono">${safeText(a.repeat_count ?? 0)}</td>
          <td class="center">${displayLabelBadge(finalLabel)}</td>
          <td class="center mono">${safeText(a.raw_severity)}</td>
          <td class="center mono">${safeNum(a.ae_score, 4)}</td>
          <td class="summary-cell">${safeText(summary)}</td>
        </tr>
      `;
    }).join("");

    setLastUpdate();
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10" class="muted">Failed to load recent traffic.</td></tr>`;
    console.error(err);
  }
}

loadRecent();
setInterval(loadRecent, 3000);