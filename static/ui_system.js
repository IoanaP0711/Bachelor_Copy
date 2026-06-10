async function loadSystemStatus() {
  try {
    const [healthRes, statsRes] = await Promise.all([
      fetch("/health"),
      fetch("/stats")
    ]);

    const health = await healthRes.json();
    const stats = await statsRes.json();

    const status = health.status || "unknown";

    setText("backendStatus", status.toUpperCase());
    setText("modelPath", health.model || "-");
    setText("featureCount", health.n_features ?? "-");

    setText("alertsBuffered", stats.alerts_buffered ?? "0");
    setText("recentBuffered", stats.recent_buffered ?? "0");
    setText("throughput", safeNum(stats.throughput_fps, 2));
    setText("cpu", safeNum(stats.cpu_proc_pct, 1) + "%");
    setText("rss", safeNum(stats.rss_mb, 1) + " MB");
    setText("repeatKeys", stats.repeat_keys_buffered ?? "0");

    if (stats.bands) {
      setText(
        "bands",
        `ok=${safeNum(stats.bands.ok, 6)} warn=${safeNum(stats.bands.warn, 6)} crit=${safeNum(stats.bands.crit, 6)}`
      );
    } else {
      setText("bands", "not set");
    }

    if (Array.isArray(health.display_labels)) {
      setHtml("displayLabels", health.display_labels.map(label => displayLabelBadge(label)).join(" "));
    } else {
      setText("displayLabels", "-");
    }

    const pill = document.getElementById("statusPill");
    if (pill) {
      if (status.toLowerCase() === "ok") {
        pill.className = "pill ok";
        pill.textContent = "OK";
      } else {
        pill.className = "pill review";
        pill.textContent = "CHECK";
      }
    }

    setLastUpdate();
  } catch (err) {
    setText("backendStatus", "ERROR");
    const pill = document.getElementById("statusPill");
    if (pill) {
      pill.className = "pill critical";
      pill.textContent = "ERROR";
    }
    setText("lastUpdate", "Failed to load system status");
    console.error(err);
  }
}

loadSystemStatus();