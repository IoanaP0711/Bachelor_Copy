async function loadSystemStatus() {
  try {
    const [healthRes, statsRes, configRes] = await Promise.all([
      fetch("/health", {
        cache: "no-store"
      }),
      fetch("/stats", {
        cache: "no-store"
      }),
      fetch("/runtime-config", {
        cache: "no-store"
      })
    ]);

    if (!healthRes.ok || !statsRes.ok || !configRes.ok) {
      throw new Error(
        `System request failed: health=${healthRes.status}, ` +
        `stats=${statsRes.status}, config=${configRes.status}`
      );
    }

    const health = await healthRes.json();
    const stats = await statsRes.json();
    const configPayload = await configRes.json();

    const status = health.status || "unknown";

    setText(
      "backendStatus",
      status.toUpperCase()
    );

    setText(
      "modelPath",
      health.model || "-"
    );

    setText(
      "featureCount",
      health.n_features ?? "-"
    );

    setText(
      "alertsBuffered",
      stats.alerts_buffered ?? "0"
    );

    setText(
      "recentBuffered",
      stats.recent_buffered ?? "0"
    );

    setText(
      "throughput",
      safeNum(
        stats.throughput_fps,
        2
      )
    );

    setText(
      "cpu",
      safeNum(
        stats.cpu_proc_pct,
        1
      ) + "%"
    );

    setText(
      "rss",
      safeNum(
        stats.rss_mb,
        1
      ) + " MB"
    );

    setText(
      "repeatKeys",
      stats.repeat_keys_buffered ?? "0"
    );

    renderRuntimeConfig(
      configPayload
    );

    if (
      Array.isArray(
        health.display_labels
      )
    ) {
      setHtml(
        "displayLabels",
        health.display_labels
          .map(
            label =>
              displayLabelBadge(
                label
              )
          )
          .join(" ")
      );
    } else {
      setText(
        "displayLabels",
        "-"
      );
    }

    const pill =
      document.getElementById(
        "statusPill"
      );

    if (pill) {
      if (
        status.toLowerCase() === "ok"
      ) {
        pill.className = "pill ok";
        pill.textContent = "OK";
      } else {
        pill.className =
          "pill review";
        pill.textContent = "CHECK";
      }
    }

    setLastUpdate();
  } catch (error) {
    setText(
      "backendStatus",
      "ERROR"
    );

    const pill =
      document.getElementById(
        "statusPill"
      );

    if (pill) {
      pill.className =
        "pill critical";
      pill.textContent = "ERROR";
    }

    setText(
      "lastUpdate",
      "Failed to load system status"
    );

    console.error(
      "System status load error:",
      error
    );
  }
}


function renderRuntimeConfig(
  payload
) {
  const config =
    payload &&
    typeof payload.config === "object"
      ? payload.config
      : {};

  const thresholds =
    config.thresholds &&
    typeof config.thresholds === "object"
      ? config.thresholds
      : {};

  const repeatLogic =
    config.repeat_logic &&
    typeof config.repeat_logic === "object"
      ? config.repeat_logic
      : {};

  const notifications =
    config.notifications &&
    typeof config.notifications === "object"
      ? config.notifications
      : {};

  setText(
    "configOkThreshold",
    formatConfigNumber(
      thresholds.ok
    )
  );

  setText(
    "configWarnThreshold",
    formatConfigNumber(
      thresholds.warn
    )
  );

  setText(
    "configCritThreshold",
    formatConfigNumber(
      thresholds.crit
    )
  );

  setText(
    "configRepeatWindow",
    formatSeconds(
      repeatLogic.window_seconds
    )
  );

  const repeatCount = Number(
    repeatLogic.critical_repeat_count
  );

  setText(
    "configRepeatThreshold",
    Number.isFinite(
      repeatCount
    )
      ? `${repeatCount} matching events`
      : "—"
  );

  setText(
    "configNotificationCooldown",
    formatSeconds(
      notifications.cooldown_seconds
    )
  );

  setText(
    "configNotificationState",
    `Notification state: ${
      notifications.enabled === false
        ? "disabled"
        : "enabled"
    }`
  );

  const sourceElement =
    document.getElementById(
      "configSource"
    );

  if (sourceElement) {
    const source = String(
      payload?.source || "defaults"
    ).toLowerCase();

    if (source === "file") {
      sourceElement.textContent =
        "Loaded from file";

      sourceElement.className =
        "config-source-badge " +
        "config-source-file";
    } else {
      sourceElement.textContent =
        "Using defaults";

      sourceElement.className =
        "config-source-badge " +
        "config-source-defaults";
    }
  }

  const warningElement =
    document.getElementById(
      "configWarning"
    );

  if (warningElement) {
    const warnings =
      Array.isArray(
        payload?.warnings
      )
        ? payload.warnings.filter(
            warning =>
              Boolean(warning)
          )
        : [];

    if (warnings.length > 0) {
      warningElement.textContent =
        warnings.join(" ");

      warningElement.classList.remove(
        "hidden"
      );
    } else {
      warningElement.textContent = "";

      warningElement.classList.add(
        "hidden"
      );
    }
  }
}


function formatConfigNumber(
  value
) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toLocaleString(
    undefined,
    {
      maximumFractionDigits: 6
    }
  );
}


function formatSeconds(
  value
) {
  const seconds = Number(value);

  if (!Number.isFinite(seconds)) {
    return "—";
  }

  return (
    seconds.toLocaleString() +
    " s"
  );
}


loadSystemStatus();

/* =========================================================
   Replay/demo mode controls
   ========================================================= */

function setReplayText(
  id,
  value
) {
  const element =
    document.getElementById(id);

  if (!element) {
    return;
  }

  element.textContent =
    value === null ||
    value === undefined ||
    value === ""
      ? "—"
      : String(value);
}


function renderReplayModePanel(
  payload
) {
  if (!payload) {
    return;
  }

  const isReplay =
    String(
      payload.mode || ""
    ).toLowerCase() === "replay";

  const replay =
    payload.replay &&
    typeof payload.replay === "object"
      ? payload.replay
      : {};

  const controls =
    document.getElementById(
      "replayControls"
    );

  const details =
    document.getElementById(
      "replayStatusDetails"
    );

  const description =
    document.getElementById(
      "runtimeModeDescription"
    );

  const startButton =
    document.getElementById(
      "startReplayButton"
    );

  const stopButton =
    document.getElementById(
      "stopReplayButton"
    );

  if (!isReplay) {
    if (description) {
      description.textContent =
        "Live mode is active. New Suricata flows are " +
        "sent through the normal model, contextual " +
        "filtering, repeat analysis, and explanation pipeline.";
    }

    controls?.classList.add(
      "hidden"
    );

    details?.classList.add(
      "hidden"
    );

    return;
  }

  if (description) {
    description.textContent =
      "Replay mode is active for a controlled " +
      "demonstration and validation. Saved decisions are " +
      "shown through the normal dashboard and explanation " +
      "views without changing model inference or contextual logic.";
  }

  controls?.classList.remove(
    "hidden"
  );

  details?.classList.remove(
    "hidden"
  );

  const running =
    Boolean(replay.running);

  if (startButton) {
    startButton.disabled = running;
  }

  if (stopButton) {
    stopButton.disabled = !running;
  }

  setReplayText(
    "replayRunningState",
    running
      ? "Running"
      : "Stopped"
  );

  setReplayText(
    "replaySourceFile",
    replay.source_file
  );

  setReplayText(
    "replayLoadedEvents",
    replay.loaded_events ?? 0
  );

  setReplayText(
    "replayEmittedEvents",
    replay.emitted_events ?? 0
  );

  const interval =
    Number(
      payload.replay_interval_seconds
    );

  setReplayText(
    "replayInterval",
    Number.isFinite(interval)
      ? `${interval} s`
      : "—"
  );

  setReplayText(
    "replayLoopState",
    payload.replay_loop
      ? "Yes"
      : "No"
  );

  const replayError =
    document.getElementById(
      "replayError"
    );

  if (replayError) {
    const errorText =
      String(
        replay.last_error || ""
      ).trim();

    replayError.textContent =
      errorText;

    replayError.classList.toggle(
      "hidden",
      !errorText
    );
  }
}


async function sendReplayCommand(
  action
) {
  const response = await fetch(
    `/replay/${action}`,
    {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type":
          "application/json"
      }
    }
  );

  let payload = {};

  try {
    payload =
      await response.json();
  } catch {
    payload = {};
  }

  if (!response.ok) {
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : (
            payload.detail?.message ||
            payload.message ||
            `Replay ${action} failed.`
          );

    throw new Error(detail);
  }

  return payload;
}


async function refreshReplayPanel() {
  if (
    typeof loadRuntimeMode !==
    "function"
  ) {
    return;
  }

  const payload =
    await loadRuntimeMode();

  if (payload) {
    renderReplayModePanel(
      payload
    );
  }
}


document.addEventListener(
  "DOMContentLoaded",
  () => {
    const startButton =
      document.getElementById(
        "startReplayButton"
      );

    const stopButton =
      document.getElementById(
        "stopReplayButton"
      );

    startButton?.addEventListener(
      "click",
      async () => {
        startButton.disabled = true;

        try {
          await sendReplayCommand(
            "start"
          );
        } catch (error) {
          window.alert(
            error.message
          );
        } finally {
          await refreshReplayPanel();
        }
      }
    );

    stopButton?.addEventListener(
      "click",
      async () => {
        stopButton.disabled = true;

        try {
          await sendReplayCommand(
            "stop"
          );
        } catch (error) {
          window.alert(
            error.message
          );
        } finally {
          await refreshReplayPanel();
        }
      }
    );

    refreshReplayPanel();
  }
);


window.addEventListener(
  "ids-runtime-mode-loaded",
  event => {
    renderReplayModePanel(
      event.detail
    );
  }
);


window.setInterval(
  () => {
    const panel =
      document.getElementById(
        "runtimeModePanelBadge"
      );

    if (panel) {
      refreshReplayPanel();
    }
  },
  3000
);