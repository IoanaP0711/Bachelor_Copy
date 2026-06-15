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

function severityDisplayValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value).trim().toUpperCase() || "-";
}

function severityCompareValue(value) {
  const normalized = severityDisplayValue(value);

  if (normalized === "-") return "-";
  if (normalized === "CRIT" || normalized === "CRITICAL" || normalized === "HIGH") return "CRITICAL";
  if (normalized === "MED" || normalized === "MEDIUM") return "MED";
  if (normalized === "WARN" || normalized === "WARNING") return "WARN";
  if (normalized === "REVIEW") return "REVIEW";
  if (normalized === "BENIGN") return "BENIGN";
  if (normalized === "OK" || normalized === "NORMAL" || normalized === "LOW") return "OK";

  return normalized;
}

function severityBadgeClass(value) {
  const normalized = severityDisplayValue(value);

  if (
    normalized.includes("CRITICAL") ||
    normalized.includes("CRIT") ||
    normalized.includes("HIGH")
  ) {
    return "badge-critical";
  }

  if (
    normalized.includes("REVIEW") ||
    normalized.includes("MEDIUM") ||
    normalized.includes("MED") ||
    normalized.includes("WARN")
  ) {
    return "badge-review";
  }

  if (normalized.includes("BENIGN")) {
    return "badge-benign";
  }

  if (
    normalized.includes("OK") ||
    normalized.includes("NORMAL") ||
    normalized.includes("LOW") ||
    normalized === "FALSE"
  ) {
    return "badge-ok";
  }

  return "badge-gray";
}

function renderSeverityBadge(value) {
  const displayValue = severityDisplayValue(value);
  return badge(displayValue, severityBadgeClass(displayValue));
}

function isContextAdjusted(raw, final) {
  const rawValue = severityCompareValue(raw);
  const finalValue = severityCompareValue(final);

  if (rawValue === "-" || finalValue === "-") return false;
  return rawValue !== finalValue;
}

function renderRawFinalBadge(raw, final) {
  const changed = isContextAdjusted(raw, final);

  return `
    <span class="raw-final-field">
      <span class="raw-final-values">
        ${renderSeverityBadge(raw)}
        <span class="raw-final-arrow" aria-hidden="true">→</span>
        ${renderSeverityBadge(final)}
      </span>
      ${
        changed
          ? '<span class="context-adjusted-badge">Changed by context</span>'
          : ""
      }
    </span>
  `;
}

function normaliseHelpAnchor(anchor) {
  const value = String(anchor ?? "")
    .trim()
    .replace(/^#/, "")
    .toLowerCase();

  return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value)
    ? value
    : "system-overview";
}

function renderHelpLink(anchor, accessibleLabel = "Open related help") {
  const safeAnchor = normaliseHelpAnchor(anchor);
  const label = String(accessibleLabel || "Open related help").trim();
  const escapedLabel = escapeHtml(label || "Open related help");

  return `
    <a
      class="context-help-link"
      href="/ui/help#${encodeURIComponent(safeAnchor)}"
      aria-label="${escapedLabel}"
      title="${escapedLabel}"
    >?</a>
  `;
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

/* -----------------------------
   Compact alert preview modal
------------------------------ */

function previewRawValue(value, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function previewNumber(value, digits = 6, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;

  const n = Number(value);
  if (Number.isNaN(n)) return previewRawValue(value, fallback);

  return n.toFixed(digits);
}

function previewTruncate(value, maxLength = 260) {
  const text = previewRawValue(value);
  if (text === "-" || text.length <= maxLength) return text;
  return text.slice(0, maxLength).trimEnd() + "…";
}

function getAlertStableId(alertObj) {
  if (!alertObj || typeof alertObj !== "object") return "-";

  const candidates = [
    alertObj.detail_id,
    alertObj.flow_id,
    alertObj.alert_id,
    alertObj.event_id,
    alertObj.uid,
    alertObj.id,
    alertObj.replay_original_id,
    alertObj.ts_unix,
    alertObj.timestamp
  ];

  for (const candidate of candidates) {
    if (candidate === null || candidate === undefined) {
      continue;
    }

    const value = String(candidate).trim();

    if (value) {
      return value;
    }
  }

  return "-";
}

function getPreviewFinalDecision(alertObj) {
  return (
    alertObj.final_label ||
    alertObj.final_decision ||
    alertObj.display_label ||
    alertObj.label ||
    alertObj.severity ||
    "-"
  );
}

function getPreviewRawSeverity(alertObj) {
  return (
    alertObj.raw_severity ||
    alertObj.raw_model_severity ||
    alertObj.raw_label ||
    alertObj.raw_output ||
    alertObj.model_severity ||
    alertObj.raw_model_flag ||
    "-"
  );
}

function getPreviewAnomalyScore(alertObj) {
  return (
    alertObj.ae_score ??
    alertObj.anomaly_score ??
    alertObj.reconstruction_error ??
    alertObj.score ??
    null
  );
}

function getPreviewShortReason(alertObj) {
  return (
    alertObj.short_reason ||
    alertObj.display_label_reason ||
    alertObj.adjustment_reason ||
    alertObj.summary ||
    alertObj.interpretation ||
    alertObj.explanation ||
    shortTableReason(alertObj) ||
    "-"
  );
}

function getPreviewSimpleExplanation(alertObj) {
  const fullExplanation = String(
    alertObj.simple_explanation ||
      alertObj.possible_explanation ||
      alertObj.interpretation ||
      alertObj.summary ||
      alertObj.explanation ||
      "-"
  ).trim();

  // The popup already shows "Changed by context",
  // so remove the repeated context sentence here.
  return fullExplanation
    .split(" The first automatic warning")[0]
    .trim();
}

function getPreviewRecommendedAction(alertObj) {
  const backendRecommendation = String(
    alertObj?.recommended_action || ""
  ).trim();

  // Prefer the recommendation generated by the backend.
  if (backendRecommendation) {
    return backendRecommendation;
  }

  // Safe fallback for alerts created before this field was added.
  const finalDecision = severityCompareValue(
    getPreviewFinalDecision(alertObj)
  );

  const fallbackRecommendations = {
    OK: "No immediate action required.",

    BENIGN:
      "Check only if this traffic appears outside normal local network " +
      "behavior, for example if the device or activity was unexpected.",

    REVIEW:
      "Verify the source IP (the device that sent the traffic), " +
      "the destination port (the service it contacted), and whether " +
      "similar events repeat.",

    CRITICAL:
      "Inspect the source host (the device that sent the traffic) and " +
      "review repeated connections with the same pattern.",
  };

  let recommendedCheck =
    fallbackRecommendations[finalDecision] ||
    fallbackRecommendations.REVIEW;

  const rawSeverity = getPreviewRawSeverity(alertObj);

  if (isContextAdjusted(rawSeverity, finalDecision)) {
    recommendedCheck += (
      " Context changed the final result: the model first assigned " +
      `raw severity ${severityDisplayValue(rawSeverity)}, while the ` +
      `final decision shown to the operator is ${severityDisplayValue(finalDecision)}.`
    );
  }

  return recommendedCheck;
}

function getOrCreatePreviewRecommendedCheck() {
  let valueElement =
    document.getElementById("previewRecommendedCheck");

  if (valueElement) {
    return valueElement;
  }

  const explanationElement =
    document.getElementById("previewSimpleExplanation");

  if (!explanationElement) {
    return null;
  }

  const recommendationRow =
    document.createElement("p");

  recommendationRow.id =
    "previewRecommendedCheckRow";

  recommendationRow.className =
    "preview-recommended-check";

  const recommendationLabel =
    document.createElement("strong");

  recommendationLabel.textContent =
    "Recommended check: ";

  valueElement =
    document.createElement("span");

  valueElement.id =
    "previewRecommendedCheck";

  recommendationRow.appendChild(
    recommendationLabel
  );

  recommendationRow.appendChild(
    valueElement
  );

  explanationElement.insertAdjacentElement(
    "afterend",
    recommendationRow
  );

  return valueElement;
}

function getPreviewContributors(alertObj) {
  const contributors =
    alertObj.top_features ||
    alertObj.top_contributors ||
    alertObj.anomaly_contributors ||
    alertObj.contributors ||
    alertObj.feature_contributions ||
    [];

  if (!Array.isArray(contributors)) return [];
  return contributors.slice(0, 3);
}

function previewBadgeClass(value) {
  const normalized = String(value || "").toUpperCase();

  if (
    normalized.includes("CRITICAL") ||
    normalized.includes("HIGH") ||
    normalized.includes("CRIT")
  ) {
    return "badge-critical";
  }

  if (
    normalized.includes("REVIEW") ||
    normalized.includes("MEDIUM") ||
    normalized.includes("MED") ||
    normalized.includes("WARN")
  ) {
    return "badge-review";
  }

  if (normalized.includes("BENIGN")) {
    return "badge-benign";
  }

  if (
    normalized.includes("OK") ||
    normalized.includes("LOW") ||
    normalized === "FALSE"
  ) {
    return "badge-ok";
  }

  return "badge-gray";
}

function setPreviewBadge(id, prefix, value) {
  const el = document.getElementById(id);
  if (!el) return;

  const displayValue = previewRawValue(value);
  el.textContent = `${prefix}: ${displayValue}`;
  el.className = `badge ${previewBadgeClass(displayValue)}`;
}

function formatPreviewContributor(contributor) {
  if (typeof contributor === "string") return contributor;
  if (!contributor || typeof contributor !== "object") return "-";

  const name =
    contributor.name ||
    contributor.feature ||
    contributor.field ||
    contributor.key ||
    "unknown_feature";

  const value =
    contributor.err ??
    contributor.error ??
    contributor.score ??
    contributor.contribution ??
    contributor.value ??
    null;

  if (value === null || value === undefined || value === "") {
    return String(name);
  }

  return `${name}: ${previewNumber(value, 6)}`;
}

function openAlertPreviewModal(alertObj) {
  console.log("Opening alert preview:", alertObj);

  if (!alertObj || typeof alertObj !== "object") {
    console.warn("No alert object received by openAlertPreviewModal.");
    return;
  }

  const modal = document.getElementById("alertPreviewModal");

  if (!modal) {
    console.error("Missing #alertPreviewModal in HTML.");
    return;
  }

  const finalDecision = getPreviewFinalDecision(alertObj);
  const rawSeverity = getPreviewRawSeverity(alertObj);
  const anomalyScore = getPreviewAnomalyScore(alertObj);
  const shortReason =
  getPreviewShortReason(alertObj);

  const simpleExplanation =
  getPreviewSimpleExplanation(alertObj);

  const recommendedAction =
  getPreviewRecommendedAction(alertObj);

  const stableId =
  getAlertStableId(alertObj);
  
  setHtml("previewRawFinalComparison", renderRawFinalBadge(rawSeverity, finalDecision));

  setPreviewBadge("previewFinalDecision", "Final decision", finalDecision);
  setPreviewBadge("previewRawSeverity", "Raw model", rawSeverity);

  const scoreEl = document.getElementById("previewAnomalyScore");
  if (scoreEl) scoreEl.textContent = previewNumber(anomalyScore, 6);

  const reasonEl = document.getElementById("previewShortReason");
  if (reasonEl) reasonEl.textContent = previewRawValue(shortReason);

  const explanationEl =
  document.getElementById(
    "previewSimpleExplanation"
  );

  if (explanationEl) {
    explanationEl.textContent =
      previewRawValue(simpleExplanation);
  }

  const recommendationEl =
    getOrCreatePreviewRecommendedCheck();

  if (recommendationEl) {
    const shortRecommendedAction = String(
      recommendedAction || ""
    )
      .split(" Context changed the final result:")[0]
      .trim();

    recommendationEl.textContent =
      shortRecommendedAction ||
      "No recommended check is available.";
  }

  const contributorsEl =
  document.getElementById(
    "previewContributors"
  );
  if (contributorsEl) {
    const contributors = getPreviewContributors(alertObj);
    contributorsEl.innerHTML = "";

    if (!contributors.length) {
      const li = document.createElement("li");
      li.textContent = "-";
      contributorsEl.appendChild(li);
    } else {
      contributors.forEach((contributor) => {
        const li = document.createElement("li");
        li.textContent = formatPreviewContributor(contributor);
        contributorsEl.appendChild(li);
      });
    }
  }

  const detailsBtn = document.getElementById("openFullDetailsBtn");

  if (detailsBtn) {
    if (stableId && stableId !== "-") {
      detailsBtn.href = `/ui/alerts/${encodeURIComponent(String(stableId))}`;
      detailsBtn.classList.remove("disabled-link");
      detailsBtn.style.display = "inline-flex";
    } else {
      detailsBtn.href = "#";
      detailsBtn.classList.add("disabled-link");
      detailsBtn.style.display = "none";
    }
  }

  modal.classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function closeAlertPreviewModal() {
  const modal = document.getElementById("alertPreviewModal");
  if (!modal) return;

  modal.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

window.openAlertPreviewModal = openAlertPreviewModal;
window.closeAlertPreviewModal = closeAlertPreviewModal;

document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") {
    closeAlertPreviewModal();
  }
});

document.addEventListener("click", function (event) {
  const modal = document.getElementById("alertPreviewModal");
  if (!modal || modal.classList.contains("hidden")) return;

  if (event.target === modal) {
    closeAlertPreviewModal();
  }
});

function idsText(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function idsEscapeHtml(value) {
  return idsText(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getAlertSimpleExplanation(alert) {
  if (!alert) {
    return "No explanation available.";
  }

  return (
    alert.simple_explanation ||
    alert.explanation ||
    alert.interpretation ||
    alert.summary ||
    "No explanation available."
  );
}

function getAlertLayeredExplanations(alert) {
  if (!alert) {
    return {
      simple: "No explanation available.",
      analyst: "No explanation available.",
      technical: "No explanation available."
    };
  }

  const simple =
    alert.simple_explanation ||
    alert.explanation ||
    alert.interpretation ||
    alert.summary ||
    "No simple explanation available.";

  const analyst =
    alert.analyst_explanation ||
    alert.possible_explanation ||
    alert.what_to_check ||
    simple;

  const technical =
    alert.technical_explanation ||
    alert.full_explanation ||
    alert.legacy_explanation ||
    alert.explanation ||
    simple;

  return {
    simple,
    analyst,
    technical
  };
}

function idsRenderExplanationParagraphs(text) {
  const cleanText = idsText(text).trim();

  if (!cleanText) {
    return "<p>No explanation available.</p>";
  }

  return cleanText
    .split(/\n\s*\n/)
    .map((paragraph) => `<p>${idsEscapeHtml(paragraph)}</p>`)
    .join("");
}

function renderAlertExplanationTabs(alert, containerOrId) {
  const container =
    typeof containerOrId === "string"
      ? document.getElementById(containerOrId)
      : containerOrId;

  if (!container) {
    return;
  }

  const explanations = getAlertLayeredExplanations(alert);

  container.innerHTML = `
    <div class="explanation-tabs" role="tablist" aria-label="Explanation detail level">
      <button class="explanation-tab active" type="button" data-layer="simple">Simple</button>
      <button class="explanation-tab" type="button" data-layer="analyst">Analyst</button>
      <button class="explanation-tab" type="button" data-layer="technical">Technical</button>
    </div>
    <div class="explanation-layer-body">
      ${idsRenderExplanationParagraphs(explanations.simple)}
    </div>
  `;

  const buttons = container.querySelectorAll(".explanation-tab");
  const body = container.querySelector(".explanation-layer-body");

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const layer = button.dataset.layer || "simple";

      buttons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");

      body.innerHTML = idsRenderExplanationParagraphs(
        explanations[layer] || explanations.simple
      );
    });
  });
}

window.renderSeverityBadge = renderSeverityBadge;
window.renderRawFinalBadge = renderRawFinalBadge;
window.renderHelpLink = renderHelpLink;
window.isContextAdjusted = isContextAdjusted;
window.getAlertSimpleExplanation = getAlertSimpleExplanation;
window.getAlertLayeredExplanations = getAlertLayeredExplanations;
window.renderAlertExplanationTabs = renderAlertExplanationTabs;

/* ------------------------------------------------------------------
   Shared in-app REVIEW / CRITICAL notification bell
------------------------------------------------------------------ */

const IDS_NOTIFICATION_KEYS = {
  read: "ids.notification.readIds.v1",
  sound: "ids.notification.soundEnabled.v1",
  notified: "ids.notification.lastNotifiedKeys.v1"
};

const IDS_NOTIFICATION_POLL_MS = 3000;
const IDS_NOTIFICATION_LIMIT = 200;
const IDS_NOTIFICATION_VISIBLE_LIMIT = 40;

const idsNotificationState = {
  items: [],
  loading: false,
  started: false,
  audioContext: null
};

function idsNotificationGetStoredArray(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function idsNotificationSetStoredArray(key, values) {
  const uniqueValues = Array.from(new Set(values.map(String))).slice(-500);

  try {
    localStorage.setItem(key, JSON.stringify(uniqueValues));
  } catch {
    // The dashboard still works if browser storage is unavailable.
  }
}

function idsNotificationGetSoundPreference() {
  try {
    const stored = localStorage.getItem(IDS_NOTIFICATION_KEYS.sound);

    if (stored === null) {
      localStorage.setItem(IDS_NOTIFICATION_KEYS.sound, "true");
      return true;
    }

    return stored === "true";
  } catch {
    return true;
  }
}

function idsNotificationSetSoundPreference(enabled) {
  try {
    localStorage.setItem(IDS_NOTIFICATION_KEYS.sound, String(enabled));
  } catch {
    // Ignore storage failures.
  }
}

function idsNotificationFinalDecision(alertObj) {
  const explicitFinalValue = String(
    alertObj?.final_label ??
    alertObj?.final_decision ??
    alertObj?.display_label ??
    alertObj?.final_severity ??
    alertObj?.label ??
    ""
  )
    .trim()
    .toUpperCase();

  if (explicitFinalValue === "CRITICAL") {
    return "CRITICAL";
  }

  if (explicitFinalValue === "REVIEW") {
    return "REVIEW";
  }

  /*
   * Some existing API responses use "severity" for the
   * final user-facing label. Only accept exact final labels.
   *
   * Do not convert WARN, MED, or CRIT here because those
   * may represent raw model severities.
   */
  const severityFallback = String(
    alertObj?.severity ?? ""
  )
    .trim()
    .toUpperCase();

  if (severityFallback === "CRITICAL") {
    return "CRITICAL";
  }

  if (severityFallback === "REVIEW") {
    return "REVIEW";
  }

  return "";
}

function idsNotificationTimestampMs(alertObj) {
  const raw =
    alertObj?.ts_unix ??
    alertObj?.timestamp ??
    alertObj?.ts ??
    alertObj?.created_at ??
    0;
  const numeric = Number(raw);

  if (Number.isFinite(numeric) && numeric > 0) {
    return numeric > 1_000_000_000_000 ? numeric : numeric * 1000;
  }

  const parsed = Date.parse(String(raw || ""));
  return Number.isNaN(parsed) ? 0 : parsed;
}

function idsNotificationFormatTime(timestampMs) {
  if (!timestampMs) return "Unknown time";

  return new Date(timestampMs)
    .toISOString()
    .replace("T", " ")
    .split(".")[0] + "Z";
}

function idsNotificationSourceIp(alertObj) {
  return String(
    alertObj?.src_ip ??
    alertObj?.source_ip ??
    ""
  ).trim();
}

function idsNotificationReason(alertObj) {
  const decision = idsNotificationFinalDecision(alertObj);

  const rawValue = String(
    alertObj?.raw_severity ??
    alertObj?.model_severity ??
    alertObj?.raw_label ??
    ""
  )
    .trim()
    .toUpperCase();

  const normalizedRaw =
    rawValue === "CRIT"
      ? "CRITICAL"
      : rawValue;

  const repeatCount = Number(
    alertObj?.repeat_count ??
    alertObj?.repetition_count ??
    alertObj?.repeat ??
    0
  );

  const explicitAdjustedValue =
    alertObj?.context_adjusted ??
    alertObj?.changed_by_context ??
    alertObj?.was_adjusted;

  const explicitlyAdjusted =
    explicitAdjustedValue === true ||
    String(explicitAdjustedValue).toLowerCase() === "true";

  const hasAdjustmentReason = Boolean(
    String(
      alertObj?.adjustment_reason ??
      alertObj?.contextual_reason ??
      ""
    ).trim()
  );

  const contextAdjusted =
    explicitlyAdjusted ||
    hasAdjustmentReason ||
    Boolean(
      normalizedRaw &&
      decision &&
      normalizedRaw !== decision
    );

  if (decision === "CRITICAL") {
    if (repeatCount > 1) {
      return "Repeated high-risk activity detected.";
    }

    if (contextAdjusted) {
      return "Context-adjusted high-risk anomaly detected.";
    }

    return "High-risk anomaly requires immediate attention.";
  }

  if (decision === "REVIEW") {
    if (repeatCount > 1) {
      return "Repeated anomalous activity requires review.";
    }

    if (contextAdjusted) {
      return "Context-adjusted anomaly requires analyst review.";
    }

    return "Anomalous flow requires analyst review.";
  }

  return "Important alert detected.";
}

function idsNotificationExplicitId(alertObj) {
  if (!alertObj || typeof alertObj !== "object") {
    return "";
  }

  const candidates = [
    alertObj.detail_id,
    alertObj.flow_id,
    alertObj.alert_id,
    alertObj.event_id,
    alertObj.uid,
    alertObj.id,
    alertObj.replay_original_id
  ];

  for (const candidate of candidates) {
    if (candidate === null || candidate === undefined) {
      continue;
    }

    const value = String(candidate).trim();

    if (value) {
      return value;
    }
  }

  return "";
}

function idsNotificationHash(value) {
  let hash = 2166136261;

  for (const character of String(value)) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }

  return (hash >>> 0).toString(36);
}

function idsNotificationKey(alertObj) {
  const explicitId = idsNotificationExplicitId(alertObj);
  if (explicitId) return `alert:${explicitId}`;

  const identity = [
    alertObj?.ts_unix ?? alertObj?.timestamp ?? alertObj?.ts ?? "",
    idsNotificationSourceIp(alertObj),
    alertObj?.src_port ?? alertObj?.source_port ?? "",
    alertObj?.dest_ip ?? alertObj?.dst_ip ?? alertObj?.destination_ip ?? "",
    alertObj?.dest_port ?? alertObj?.dst_port ?? alertObj?.destination_port ?? "",
    alertObj?.proto ?? alertObj?.protocol ?? "",
    idsNotificationFinalDecision(alertObj),
    alertObj?.ae_score ?? alertObj?.anomaly_score ?? alertObj?.score ?? ""
  ].join("|");

  return `derived:${idsNotificationHash(identity)}`;
}

function idsNotificationDetailHref(alertObj) {
  let stableId = idsNotificationExplicitId(alertObj);

  if (!stableId && typeof getAlertStableId === "function") {
    const helperId = getAlertStableId(alertObj);
    stableId = helperId && helperId !== "-" ? String(helperId) : "";
  }

  return stableId
    ? `/ui/alerts/${encodeURIComponent(stableId)}`
    : "/ui/alerts";
}

function idsNotificationCreateItem(alertObj) {
  const timestampMs = idsNotificationTimestampMs(alertObj);

  return {
    key: idsNotificationKey(alertObj),
    severity: idsNotificationFinalDecision(alertObj),
    timestampMs,
    timestamp: idsNotificationFormatTime(timestampMs),
    sourceIp: idsNotificationSourceIp(alertObj),
    reason: idsNotificationReason(alertObj),
    href: idsNotificationDetailHref(alertObj)
  };
}

function idsNotificationCreateUi() {
  const existing = document.getElementById("idsNotificationCenter");
  if (existing) return existing;

  const topbar = document.querySelector(".topbar");
  if (!topbar) return null;

  let actions = Array.from(topbar.children).find((child) =>
    child.classList?.contains("topbar-actions")
  );

  if (!actions) {
    actions = document.createElement("div");
    actions.className = "topbar-actions";

    const nav = Array.from(topbar.children).find((child) =>
      child.classList?.contains("main-nav")
    );

    if (nav) {
      topbar.insertBefore(actions, nav);
      actions.appendChild(nav);
    } else {
      topbar.appendChild(actions);
    }
  }

  actions.insertAdjacentHTML("beforeend", `
    <div class="notification-center" id="idsNotificationCenter">
      <button
        id="idsNotificationBell"
        class="notification-bell"
        type="button"
        aria-label="Open notifications"
        aria-haspopup="true"
        aria-expanded="false"
        aria-controls="idsNotificationDropdown"
        title="Notifications"
      >
        <svg class="notification-bell-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0"></path>
        </svg>
        <span id="idsNotificationUnreadCount" class="notification-unread-count" hidden>0</span>
      </button>

      <section
        id="idsNotificationDropdown"
        class="notification-dropdown"
        aria-label="Important alert notifications"
        hidden
      >
        <div class="notification-dropdown-header">
          <div>
            <strong>Notifications</strong>
            <span>REVIEW and CRITICAL only</span>
          </div>
          <button
            id="idsNotificationSoundToggle"
            class="notification-sound-toggle"
            type="button"
            role="switch"
            aria-checked="true"
          >Sound ON</button>
        </div>

        <div id="idsNotificationList" class="notification-list" aria-live="polite">
          <div class="notification-empty-state">Loading notifications…</div>
        </div>

        <div class="notification-dropdown-footer">
          <button id="idsNotificationMarkAllRead" class="notification-mark-read" type="button">
            Mark all read
          </button>
        </div>
      </section>
    </div>
  `);

  return document.getElementById("idsNotificationCenter");
}

function idsNotificationReadSet() {
  return new Set(idsNotificationGetStoredArray(IDS_NOTIFICATION_KEYS.read));
}

function idsNotificationMarkRead(keys) {
  const readIds = idsNotificationReadSet();
  keys.forEach((key) => readIds.add(String(key)));
  idsNotificationSetStoredArray(IDS_NOTIFICATION_KEYS.read, [...readIds]);
}

function idsNotificationRender() {
  const list = document.getElementById("idsNotificationList");
  const count = document.getElementById("idsNotificationUnreadCount");
  const bell = document.getElementById("idsNotificationBell");
  const markAll = document.getElementById("idsNotificationMarkAllRead");
  const soundToggle = document.getElementById("idsNotificationSoundToggle");

  if (!list || !count) return;

  const readIds = idsNotificationReadSet();
  const unreadCount = idsNotificationState.items.filter(
    (item) => !readIds.has(item.key)
  ).length;
  const soundEnabled = idsNotificationGetSoundPreference();
  count.textContent = unreadCount > 99 ? "99+" : String(unreadCount);
  count.hidden = unreadCount === 0;

  if (bell) {
    bell.classList.toggle("has-unread", unreadCount > 0);
    bell.setAttribute(
      "aria-label",
      unreadCount
        ? `Open notifications, ${unreadCount} unread`
        : "Open notifications, no unread items"
    );
  }

  if (markAll) markAll.disabled = unreadCount === 0;

  if (soundToggle) {
    soundToggle.textContent = soundEnabled ? "Sound ON" : "Sound OFF";
    soundToggle.setAttribute("aria-checked", String(soundEnabled));
    soundToggle.classList.toggle("is-off", !soundEnabled);
  }

  if (!idsNotificationState.items.length) {
    list.innerHTML = `
      <div class="notification-empty-state">
        No REVIEW or CRITICAL notifications.
      </div>
    `;
    return;
  }

  list.innerHTML = idsNotificationState.items.map((item) => {
    const unread = !readIds.has(item.key);
    const source = item.sourceIp
      ? `Source ${escapeHtml(item.sourceIp)}`
      : "Source unavailable";

    return `
      <a
        class="notification-item${unread ? " is-unread" : ""}"
        href="${escapeHtml(item.href)}"
        data-notification-key="${escapeHtml(item.key)}"
      >
        <span class="notification-unread-dot" aria-hidden="true"></span>
        <span class="notification-item-content">
          <span class="notification-item-topline">
            ${renderSeverityBadge(item.severity)}
            <time>${escapeHtml(item.timestamp)}</time>
          </span>
          <span class="notification-source">${source}</span>
          <span class="notification-reason">${escapeHtml(item.reason)}</span>
          <span class="notification-detail-link">View Details →</span>
        </span>
      </a>
    `;
  }).join("");
}

function idsNotificationSetDropdown(open) {
  const dropdown = document.getElementById("idsNotificationDropdown");
  const bell = document.getElementById("idsNotificationBell");
  if (!dropdown || !bell) return;

  dropdown.hidden = !open;
  bell.setAttribute("aria-expanded", String(open));
  if (open) idsNotificationRender();
}

async function idsNotificationUnlockAudio() {
  if (!idsNotificationGetSoundPreference()) {
    return false;
  }

  const AudioContextClass =
    window.AudioContext ||
    window.webkitAudioContext;

  if (!AudioContextClass) {
    console.warn("Web Audio API is not supported by this browser.");
    return false;
  }

  try {
    if (
      !idsNotificationState.audioContext ||
      idsNotificationState.audioContext.state === "closed"
    ) {
      idsNotificationState.audioContext = new AudioContextClass();
    }

    const context = idsNotificationState.audioContext;

    if (
      context.state === "suspended" ||
      context.state === "interrupted"
    ) {
      await context.resume();
    }

    return context.state === "running";
  } catch (error) {
    console.warn("Notification audio could not be enabled:", error);
    return false;
  }
}

async function idsNotificationPlaySound(hasCritical) {
  if (!idsNotificationGetSoundPreference()) {
    return;
  }

  const audioReady = await idsNotificationUnlockAudio();

  if (!audioReady) {
    console.warn(
      "Notification sound was blocked. Click once inside the page to enable browser audio."
    );
    return;
  }

  const context = idsNotificationState.audioContext;
  const startTime = context.currentTime;

  function scheduleTone(
    frequency,
    delay,
    duration,
    volume
  ) {
    const oscillator = context.createOscillator();
    const gain = context.createGain();

    const toneStart = startTime + delay;
    const toneEnd = toneStart + duration;

    oscillator.type = "triangle";
    oscillator.frequency.setValueAtTime(
      frequency,
      toneStart
    );

    gain.gain.setValueAtTime(
      0.0001,
      toneStart
    );

    gain.gain.linearRampToValueAtTime(
      volume,
      toneStart + 0.02
    );

    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      toneEnd
    );

    oscillator.connect(gain);
    gain.connect(context.destination);

    oscillator.start(toneStart);
    oscillator.stop(toneEnd + 0.02);
  }

  if (hasCritical) {
    scheduleTone(880, 0, 0.16, 0.22);
    scheduleTone(1100, 0.18, 0.20, 0.22);
  } else {
    scheduleTone(660, 0, 0.15, 0.20);
    scheduleTone(820, 0.16, 0.17, 0.18);
  }
}

function idsNotificationProcessNewItems(items) {
  const storedRaw = (() => {
    try {
      return localStorage.getItem(IDS_NOTIFICATION_KEYS.notified);
    } catch {
      return null;
    }
  })();
  const currentKeys = items.map((item) => item.key);

  // First page load establishes a baseline, so buffered old alerts do not beep.
  if (storedRaw === null) {
    idsNotificationSetStoredArray(IDS_NOTIFICATION_KEYS.notified, currentKeys);
    return;
  }

  const notified = new Set(
    idsNotificationGetStoredArray(IDS_NOTIFICATION_KEYS.notified)
  );
  const newItems = items.filter((item) => !notified.has(item.key));

  if (!newItems.length) return;

  idsNotificationSetStoredArray(
    IDS_NOTIFICATION_KEYS.notified,
    [...notified, ...newItems.map((item) => item.key)]
  );

  void idsNotificationPlaySound(
  newItems.some((item) => item.severity === "CRITICAL")
  );
}

async function idsNotificationRefresh() {
  if (idsNotificationState.loading) return;
  idsNotificationState.loading = true;

  try {
    const response = await fetch(
      `/alerts?limit=${IDS_NOTIFICATION_LIMIT}`,
      { cache: "no-store", headers: { Accept: "application/json" } }
    );

    if (!response.ok) {
      throw new Error(`Notifications request failed: ${response.status}`);
    }

    const data = await response.json();
    const alerts = Array.isArray(data?.alerts) ? data.alerts : [];
    const uniqueItems = new Map();

    alerts
      .filter((alertObj) => {
        const decision = idsNotificationFinalDecision(alertObj);
        return decision === "REVIEW" || decision === "CRITICAL";
      })
      .map(idsNotificationCreateItem)
      .sort((left, right) => right.timestampMs - left.timestampMs)
      .forEach((item) => {
        if (!uniqueItems.has(item.key)) uniqueItems.set(item.key, item);
      });

    idsNotificationState.items = [...uniqueItems.values()].slice(
      0,
      IDS_NOTIFICATION_VISIBLE_LIMIT
    );

    idsNotificationProcessNewItems(idsNotificationState.items);
    idsNotificationRender();
  } catch (error) {
    console.warn("Could not refresh IDS notifications:", error);

    if (!idsNotificationState.items.length) {
      const list = document.getElementById("idsNotificationList");
      if (list) {
        list.innerHTML = `
          <div class="notification-empty-state notification-load-error">
            Notifications could not be loaded.
          </div>
        `;
      }
    }
  } finally {
    idsNotificationState.loading = false;
  }
}

function idsNotificationBindEvents(center) {
  const bell = document.getElementById("idsNotificationBell");
  const dropdown = document.getElementById("idsNotificationDropdown");
  const list = document.getElementById("idsNotificationList");
  const markAll = document.getElementById("idsNotificationMarkAllRead");
  const soundToggle = document.getElementById("idsNotificationSoundToggle");

  bell?.addEventListener("click", (event) => {
    event.stopPropagation();

    void idsNotificationUnlockAudio();

    idsNotificationSetDropdown(
      dropdown?.hidden ?? true
    );
  });

  dropdown?.addEventListener("click", (event) => event.stopPropagation());

  list?.addEventListener("click", (event) => {
    const item = event.target.closest("[data-notification-key]");
    if (!item) return;

    idsNotificationMarkRead([item.dataset.notificationKey]);
    idsNotificationRender();
  });

  markAll?.addEventListener("click", () => {
    idsNotificationMarkRead(
      idsNotificationState.items.map((item) => item.key)
    );
    idsNotificationRender();
  });

  soundToggle?.addEventListener("click", () => {
    const enabled = !idsNotificationGetSoundPreference();
    idsNotificationSetSoundPreference(enabled);
    if (enabled) idsNotificationUnlockAudio();
    idsNotificationRender();
  });

  document.addEventListener("click", (event) => {
    if (!center.contains(event.target)) idsNotificationSetDropdown(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") idsNotificationSetDropdown(false);
  });

  document.addEventListener(
    "pointerdown",
    () => {
      void idsNotificationUnlockAudio();
    },
    {
      once: true,
      capture: true
    }
  );

  window.addEventListener("storage", (event) => {
    if (
      event.key === IDS_NOTIFICATION_KEYS.read ||
      event.key === IDS_NOTIFICATION_KEYS.sound
    ) {
      idsNotificationRender();
    }
  });
}

function idsNotificationInit() {
  if (idsNotificationState.started) return;

  const center = idsNotificationCreateUi();
  if (!center) return;

  idsNotificationState.started = true;
  idsNotificationGetSoundPreference();
  idsNotificationBindEvents(center);
  idsNotificationRender();
  idsNotificationRefresh();
  window.setInterval(idsNotificationRefresh, IDS_NOTIFICATION_POLL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", idsNotificationInit, { once: true });
} else {
  idsNotificationInit();
}

/* =========================================================
   Shared live/replay mode badge
   ========================================================= */

function applyRuntimeModeBadge(
  element,
  payload
) {
  if (!element) {
    return;
  }

  const isReplay =
    String(
      payload?.mode || "live"
    ).toLowerCase() === "replay";

  element.textContent = isReplay
    ? "REPLAY MODE"
    : "LIVE MODE";

  element.classList.remove(
    "runtime-mode-loading",
    "runtime-mode-live",
    "runtime-mode-replay"
  );

  element.classList.add(
    isReplay
      ? "runtime-mode-replay"
      : "runtime-mode-live"
  );

  element.title = isReplay
    ? (
        "Controlled demonstration mode using " +
        "previously saved events."
      )
    : "Live traffic monitoring mode.";
}


async function loadRuntimeMode() {
  const headerBadge =
    document.getElementById(
      "runtimeModeBadge"
    );

  const panelBadge =
    document.getElementById(
      "runtimeModePanelBadge"
    );

  if (!headerBadge && !panelBadge) {
    return null;
  }

  try {
    const response = await fetch(
      "/runtime/mode",
      {
        cache: "no-store",
        credentials: "same-origin"
      }
    );

    if (!response.ok) {
      throw new Error(
        `Runtime mode request failed: ${response.status}`
      );
    }

    const payload = await response.json();

    applyRuntimeModeBadge(
      headerBadge,
      payload
    );

    applyRuntimeModeBadge(
      panelBadge,
      payload
    );

    window.dispatchEvent(
      new CustomEvent(
        "ids-runtime-mode-loaded",
        {
          detail: payload
        }
      )
    );

    return payload;
  } catch (error) {
    console.error(
      "Runtime mode load error:",
      error
    );

    for (const badge of [
      headerBadge,
      panelBadge
    ]) {
      if (!badge) {
        continue;
      }

      badge.textContent =
        "MODE UNKNOWN";

      badge.classList.remove(
        "runtime-mode-live",
        "runtime-mode-replay"
      );

      badge.classList.add(
        "runtime-mode-loading"
      );
    }

    return null;
  }
}


document.addEventListener(
  "DOMContentLoaded",
  () => {
    loadRuntimeMode();
  }
);

/*
 * Append this block to static/ui_common.js.
 * It adds the Blocklist link to existing navigation bars without requiring
 * every template to be edited separately.
 */

function ensureBlocklistNavigationLink() {
  const navigationBars = document.querySelectorAll(
    ".main-nav, .patterns-nav, .ids-nav"
  );

  navigationBars.forEach((navigation) => {
    let link = navigation.querySelector(
      'a[href="/ui/blocklist"]'
    );

    if (!link) {
      link = document.createElement("a");
      link.href = "/ui/blocklist";
      link.textContent = "Blocklist";
      link.dataset.nav = "blocklist";

      if (navigation.classList.contains("ids-nav")) {
        link.classList.add("ids-nav-link");
      }

      const insertionPoint =
        navigation.querySelector('a[href="/ui/help"]') ||
        navigation.querySelector('a[href="/logout"]');

      if (insertionPoint) {
        navigation.insertBefore(link, insertionPoint);
      } else {
        navigation.appendChild(link);
      }
    }

    const isBlocklistPage =
      window.location.pathname === "/ui/blocklist";

    link.classList.toggle("active", isBlocklistPage);

    if (isBlocklistPage) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}

document.addEventListener(
  "DOMContentLoaded",
  ensureBlocklistNavigationLink
);
