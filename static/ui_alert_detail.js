(function () {
    const alertId =
        window.ALERT_DETAIL_ID || "";

    let alertFeedbackState = {
        is_false_positive: false,
        user_feedback: null,
        timestamp: null,
    };

    let currentHumanStatus = "New";
    let currentAlertDetail = null;

    function safeValue(value) {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return "-";
        }

        return String(value);
    }

    function cleanText(
        value,
        fallback =
            "No explanation available."
    ) {
        if (
            value === null ||
            value === undefined
        ) {
            return fallback;
        }

        const text =
            String(value).trim();

        return text || fallback;
    }

    function escapeHtml(value) {
        return safeValue(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll(
                "'",
                "&#039;"
            );
    }

    function setText(id, value) {
        const element =
            document.getElementById(id);

        if (!element) {
            return;
        }

        element.textContent =
            safeValue(value);
    }

    function formatProcessAttribution(value) {
        const normalised =
            String(value || "")
                .trim()
                .toLowerCase();

        const labels = {
            exact_source_socket_match:
                "Exact local source socket match",

            exact_destination_socket_match:
                "Exact local destination socket match",

            not_found:
                "No matching local socket found",
        };

        return (
            labels[normalised] ||
            (
                normalised
                    ? normalised
                        .replaceAll("_", " ")
                    : "Not available"
            )
        );
    }


    function renderProcessAttribution(alert) {
        const processName =
            alert.process_name ||
            "Unknown";

        const processPid =
            alert.process_pid ??
            "-";

        const processExecutable =
            alert.process_exe ||
            "-";

        const processAttribution =
            formatProcessAttribution(
                alert.process_attribution
            );

        const confidence =
            String(
                alert
                    .process_attribution_confidence ||
                "none"
            )
                .trim()
                .toLowerCase();

        setText(
            "processName",
            processName
        );

        setText(
            "processPid",
            processPid
        );

        setText(
            "processExecutable",
            processExecutable
        );

        setText(
            "processAttribution",
            processAttribution
        );

        const confidenceBadge =
            document.getElementById(
                "processConfidenceBadge"
            );

        if (confidenceBadge) {
            confidenceBadge.className =
                "process-confidence-badge " +
                (
                    confidence === "high"
                        ? "process-confidence-high"
                        : confidence === "medium"
                            ? "process-confidence-medium"
                            : "process-confidence-none"
                );

            confidenceBadge.textContent =
                confidence === "high"
                    ? "High confidence"
                    : confidence === "medium"
                        ? "Medium confidence"
                        : "No match";
        }

        const note =
            document.getElementById(
                "processAttributionNote"
            );

        if (note) {
            if (
                alert.process_name &&
                confidence === "high"
            ) {
                note.textContent =
                    `The flow was matched to the local ` +
                    `application ${processName} using its ` +
                    `active or recently cached network socket. ` +
                    `This identifies the origin of the flow, ` +
                    `but it does not independently prove that ` +
                    `the communication is safe.`;
            } else {
                note.textContent =
                    "No exact local process association was " +
                    "available for this flow. This is common " +
                    "for broadcast, multicast, short UDP " +
                    "traffic, connections from other devices, " +
                    "or sockets that closed before observation.";
            }
        }
    }

    function setInnerHtml(id, html) {
        const element =
            document.getElementById(id);

        if (!element) {
            return;
        }

        element.innerHTML = html;
    }

    function formatTemporalTimestamp(
        value
    ) {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return "-";
        }

        const numericValue =
            Number(value);

        const date =
            Number.isFinite(
                numericValue
            )
                ? new Date(
                    numericValue * 1000
                )
                : new Date(value);

        if (
            Number.isNaN(
                date.getTime()
            )
        ) {
            return safeValue(value);
        }

        return date.toLocaleString();
    }

    function renderTemporalContext(
        alert
    ) {
        const temporal =
            alert &&
            typeof alert
                .temporal_context ===
                "object"
                ? alert.temporal_context
                : {};

        setText(
            "temporalSameSourceCount",
            temporal
                .same_source_ip_count
        );

        setText(
            "temporalSameFlowCount",
            temporal.same_flow_count
        );

        setText(
            "temporalFirstSeen",
            formatTemporalTimestamp(
                temporal.first_seen
            )
        );

        setText(
            "temporalLastSeen",
            formatTemporalTimestamp(
                temporal.last_seen
            )
        );

        setText(
            "temporalRepeatCount",
            temporal.repeat_count
        );

        setText(
            "temporalTimeWindow",
            temporal.time_window
        );

        setText(
            "temporalHistoryScope",
            temporal.history_scope
        );

        const basis = cleanText(
            temporal.seen_basis,
            "no matching records"
        );

        setText(
            "temporalSeenBasis",
            "First and last seen are " +
            `based on: ${basis}.`
        );
    }

    function severityClass(value) {
        const text =
            safeValue(value)
                .toUpperCase();

        if (
            text.includes("CRIT") ||
            text.includes("CRITICAL")
        ) {
            return (
                "detail-badge-critical"
            );
        }

        if (
            text.includes("REVIEW") ||
            text.includes("WARN") ||
            text.includes("MED")
        ) {
            return (
                "detail-badge-review"
            );
        }

        if (
            text.includes("BENIGN")
        ) {
            return (
                "detail-badge-benign"
            );
        }

        if (text.includes("OK")) {
            return "detail-badge-ok";
        }

        return (
            "detail-badge-unknown"
        );
    }

    function setBadge(id, value) {
        const element =
            document.getElementById(id);

        if (!element) {
            return;
        }

        element.textContent =
            safeValue(value)
                .toUpperCase();

        element.className =
            "detail-badge " +
            severityClass(value);
    }

    function showNotFound() {
        const content =
            document.getElementById(
                "alertDetailContent"
            );

        const notFound =
            document.getElementById(
                "alertNotFound"
            );

        const reportButton =
            document.getElementById(
                "downloadJsonReportBtn"
            );

        if (content) {
            content.classList.add(
                "hidden"
            );
        }

        if (notFound) {
            notFound.classList.remove(
                "hidden"
            );
        }

        if (reportButton) {
            reportButton.classList.add(
                "hidden"
            );
        }
    }

    function showContent() {
        const content =
            document.getElementById(
                "alertDetailContent"
            );

        const notFound =
            document.getElementById(
                "alertNotFound"
            );

        const reportButton =
            document.getElementById(
                "downloadJsonReportBtn"
            );

        if (notFound) {
            notFound.classList.add(
                "hidden"
            );
        }

        if (content) {
            content.classList.remove(
                "hidden"
            );
        }

        if (reportButton) {
            reportButton.classList.remove(
                "hidden"
            );
        }
    }

    function normaliseHumanStatus(
        value
    ) {
        const text =
            String(value || "New")
                .trim()
                .toLowerCase();

        const statuses = {
            "new": "New",
            "seen": "Seen",
            "under review":
                "Under review",
            "under_review":
                "Under review",
            "under-review":
                "Under review",
            "resolved": "Resolved",
            "false positive":
                "False positive",
            "false_positive":
                "False positive",
            "false-positive":
                "False positive",
        };

        return (
            statuses[text] ||
            "New"
        );
    }

    function humanStatusCssClass(
        value
    ) {
        return normaliseHumanStatus(
            value
        )
            .toLowerCase()
            .replaceAll(
                " ",
                "-"
            );
    }

    function formatStatusTimestamp(
        value
    ) {
        if (!value) {
            return "";
        }

        const date =
            new Date(value);

        if (
            Number.isNaN(
                date.getTime()
            )
        ) {
            return String(value);
        }

        return date.toLocaleString();
    }

    function setAlertStatusMessage(
        message,
        type = ""
    ) {
        const element =
            document.getElementById(
                "alertStatusMessage"
            );

        if (!element) {
            return;
        }

        element.textContent =
            message || "";

        element.className =
            "alert-status-message";

        if (type) {
            element.classList.add(
                "alert-status-message-" +
                type
            );
        }
    }

    function renderHumanStatus(
        status,
        updatedAt = null
    ) {
        currentHumanStatus =
            normaliseHumanStatus(
                status
            );

        const badge =
            document.getElementById(
                "alertStatusCurrentBadge"
            );

        const select =
            document.getElementById(
                "alertStatusSelect"
            );

        const updatedElement =
            document.getElementById(
                "alertStatusUpdatedAt"
            );

        if (badge) {
            badge.textContent =
                currentHumanStatus;

            badge.className =
                "human-status-badge " +
                "human-status-" +
                humanStatusCssClass(
                    currentHumanStatus
                );
        }

        if (select) {
            select.value =
                currentHumanStatus;
        }

        if (updatedElement) {
            const formatted =
                formatStatusTimestamp(
                    updatedAt
                );

            updatedElement.textContent =
                formatted
                    ? (
                        "Last status update: " +
                        formatted
                    )
                    : "Not updated yet.";
        }
    }

    async function saveAlertStatus() {
        const select =
            document.getElementById(
                "alertStatusSelect"
            );

        const button =
            document.getElementById(
                "saveAlertStatusBtn"
            );

        if (
            !select ||
            !button ||
            !alertId
        ) {
            return;
        }

        const requestedStatus =
            normaliseHumanStatus(
                select.value
            );

        button.disabled = true;
        button.textContent =
            "Saving...";

        setAlertStatusMessage(
            "Saving human review " +
            "status..."
        );

        try {
            const response =
                await fetch(
                    `/alerts/${encodeURIComponent(
                        alertId
                    )}/status`,
                    {
                        method: "POST",

                        headers: {
                            Accept:
                                "application/json",

                            "Content-Type":
                                "application/json",
                        },

                        body:
                            JSON.stringify({
                                status:
                                    requestedStatus,
                            }),
                    }
                );

            let payload = {};

            try {
                payload =
                    await response.json();
            } catch (parseError) {
                payload = {};
            }

            if (!response.ok) {
                const detail =
                    payload.detail;

                throw new Error(
                    typeof detail ===
                    "string"
                        ? detail
                        : (
                            detail
                                ?.message ||
                            "The human review " +
                            "status could not " +
                            "be saved."
                        )
                );
            }

            renderHumanStatus(
                payload.status,
                payload.updated_at
            );

            setAlertStatusMessage(
                payload.message ||
                "Human review status " +
                "updated.",
                "success"
            );
        } catch (error) {
            console.error(
                "Failed to save human " +
                "review status:",
                error
            );

            select.value =
                currentHumanStatus;

            setAlertStatusMessage(
                error.message ||
                "The human review status " +
                "could not be saved.",
                "error"
            );
        } finally {
            button.disabled = false;
            button.textContent =
                "Save status";
        }
    }

    function configureAlertStatusControls() {
        const button =
            document.getElementById(
                "saveAlertStatusBtn"
            );

        if (!button) {
            return;
        }

        button.addEventListener(
            "click",
            saveAlertStatus
        );
    }

    function formatFeedbackTimestamp(
        value
    ) {
        if (!value) {
            return "";
        }

        const date =
            new Date(value);

        if (
            Number.isNaN(
                date.getTime()
            )
        ) {
            return String(value);
        }

        return date.toLocaleString();
    }

    function setFeedbackMessage(
        message,
        type = ""
    ) {
        const element =
            document.getElementById(
                "feedbackMessage"
            );

        if (!element) {
            return;
        }

        element.textContent =
            message || "";

        element.className =
            "feedback-message";

        if (type) {
            element.classList.add(
                "feedback-message-" +
                type
            );
        }
    }

    function renderFeedbackState(
        feedback
    ) {
        const button =
            document.getElementById(
                "markFalsePositiveBtn"
            );

        const badge =
            document.getElementById(
                "feedbackStateBadge"
            );

        const normalised =
            feedback &&
            typeof feedback === "object"
                ? feedback
                : {};

        alertFeedbackState = {
            is_false_positive:
                Boolean(
                    normalised
                        .is_false_positive
                ),

            user_feedback:
                normalised
                    .user_feedback ||
                null,

            timestamp:
                normalised.timestamp ||
                null,
        };

        if (!button || !badge) {
            return;
        }

        if (
            alertFeedbackState
                .is_false_positive
        ) {
            button.disabled = true;

            button.textContent =
                "Marked as false positive";

            button.classList.add(
                "is-marked"
            );

            badge.classList.remove(
                "hidden"
            );

            const markedAt =
                formatFeedbackTimestamp(
                    alertFeedbackState
                        .timestamp
                );

            setFeedbackMessage(
                markedAt
                    ? (
                        "Recorded on " +
                        `${markedAt}.`
                    )
                    : (
                        "This alert is " +
                        "already marked as " +
                        "a false positive."
                    ),
                "success"
            );

            return;
        }

        button.disabled = false;

        button.textContent =
            "Mark as false positive";

        button.classList.remove(
            "is-marked"
        );

        badge.classList.add(
            "hidden"
        );

        setFeedbackMessage("");
    }

    async function submitFalsePositiveFeedback() {
        const button =
            document.getElementById(
                "markFalsePositiveBtn"
            );

        if (
            !button ||
            !alertId ||
            alertFeedbackState
                .is_false_positive
        ) {
            return;
        }

        button.disabled = true;

        button.textContent =
            "Saving feedback...";

        setFeedbackMessage(
            "Saving analyst feedback..."
        );

        try {
            const response =
                await fetch(
                    `/alerts/${encodeURIComponent(
                        alertId
                    )}/feedback`,
                    {
                        method: "POST",

                        headers: {
                            Accept:
                                "application/json",
                        },
                    }
                );

            let payload = {};

            try {
                payload =
                    await response.json();
            } catch (parseError) {
                payload = {};
            }

            if (!response.ok) {
                throw new Error(
                    payload.detail ||
                    (
                        "The analyst " +
                        "feedback could not " +
                        "be saved."
                    )
                );
            }

            renderFeedbackState(
                payload.feedback ||
                {
                    is_false_positive:
                        true,

                    user_feedback:
                        "false_positive",

                    timestamp: null,
                }
            );

            if (
                payload.human_status &&
                typeof payload
                    .human_status ===
                    "object"
            ) {
                renderHumanStatus(
                    payload
                        .human_status
                        .status,

                    payload
                        .human_status
                        .updated_at
                );
            }

            setFeedbackMessage(
                payload.message ||
                (
                    "Alert marked as a " +
                    "false positive."
                ),
                "success"
            );
        } catch (error) {
            console.error(
                "Failed to save alert " +
                "feedback:",
                error
            );

            button.disabled = false;

            button.textContent =
                "Mark as false positive";

            setFeedbackMessage(
                error.message ||
                (
                    "The analyst feedback " +
                    "could not be saved."
                ),
                "error"
            );
        }
    }

    function configureFeedbackButton() {
        const button =
            document.getElementById(
                "markFalsePositiveBtn"
            );

        if (!button) {
            return;
        }

        button.addEventListener(
            "click",
            submitFalsePositiveFeedback
        );
    }

    function normaliseContributors(
        contributors
    ) {
        if (
            !Array.isArray(
                contributors
            )
        ) {
            return [];
        }

        return contributors
            .map((item) => {
                if (
                    !item ||
                    typeof item !==
                        "object"
                ) {
                    return null;
                }

                const feature =
                    cleanText(
                        item.feature ||
                        item.name,
                        "Unknown feature"
                    );

                const rawError =
                    item.error ??
                    item.err ??
                    item.value ??
                    item.contribution;

                const numericError =
                    Number(rawError);

                return {
                    feature:
                        feature,

                    error:
                        rawError,

                    numericError:
                        Number.isFinite(
                            numericError
                        )
                            ? numericError
                            : null,

                    scaledValue:
                        item
                            .scaled_value ??
                        item.x ??
                        "-",

                    reconstructedValue:
                        item
                            .reconstructed_value ??
                        item.x_hat ??
                        "-",
                };
            })
            .filter(Boolean);
    }

    function renderContributorChart(
        contributors
    ) {
        const chart =
            document.getElementById(
                "contributorsChart"
            );

        const status =
            document.getElementById(
                "contributorsChartStatus"
            );

        const fallback =
            document.getElementById(
                "contributorsFallback"
            );

        if (!chart || !status) {
            return;
        }

        const usableContributors =
            normaliseContributors(
                contributors
            )
                .filter(
                    (item) =>
                        item.numericError !==
                        null
                )
                .sort(
                    (a, b) =>
                        b.numericError -
                        a.numericError
                );

        if (
            typeof window.Plotly !==
            "undefined"
        ) {
            window.Plotly.purge(
                chart
            );
        }

        if (
            usableContributors
                .length === 0
        ) {
            chart.hidden = true;

            status.textContent =
                "No contributor data " +
                "available.";

            status.classList.remove(
                "hidden"
            );

            if (fallback) {
                fallback.open = true;
            }

            return;
        }

        if (
            typeof window.Plotly ===
            "undefined"
        ) {
            chart.hidden = true;

            status.textContent =
                "The contributor chart " +
                "could not be loaded. " +
                "Contributor values are " +
                "available below.";

            status.classList.remove(
                "hidden"
            );

            if (fallback) {
                fallback.open = true;
            }

            return;
        }

        status.classList.add(
            "hidden"
        );

        chart.hidden = false;

        const labels =
            usableContributors.map(
                (item) =>
                    item.feature
            );

        const values =
            usableContributors.map(
                (item) =>
                    item.numericError
            );

        const longestLabel =
            labels.reduce(
                (
                    maximum,
                    label
                ) =>
                    Math.max(
                        maximum,
                        label.length
                    ),
                0
            );

        const chartHeight =
            Math.max(
                340,
                (
                    usableContributors
                        .length *
                    34
                ) + 110
            );

        chart.style.height =
            `${chartHeight}px`;

        window.Plotly.newPlot(
            chart,
            [
                {
                    type: "bar",
                    orientation: "h",
                    x: values,
                    y: labels,

                    hovertemplate:
                        "<b>%{y}</b>" +
                        "<br>Contribution: " +
                        "%{x:.6g}" +
                        "<extra></extra>",
                },
            ],
            {
                autosize: true,
                height: chartHeight,

                margin: {
                    t: 20,
                    r: 30,
                    b: 65,

                    l: Math.min(
                        320,
                        Math.max(
                            140,
                            longestLabel *
                            7
                        )
                    ),
                },

                paper_bgcolor:
                    "rgba(0, 0, 0, 0)",

                plot_bgcolor:
                    "rgba(0, 0, 0, 0)",

                bargap: 0.22,

                xaxis: {
                    title:
                        "Contributor / " +
                        "reconstruction error",

                    rangemode:
                        "tozero",

                    gridcolor:
                        "#e5e7eb",

                    zeroline: false,
                    automargin: true,
                },

                yaxis: {
                    autorange:
                        "reversed",

                    automargin: true,
                },

                showlegend: false,
            },
            {
                responsive: true,
                displaylogo: false,

                modeBarButtonsToRemove: [
                    "select2d",
                    "lasso2d",
                ],
            }
        );
    }

    function renderContributors(
        contributors
    ) {
        const container =
            document.getElementById(
                "contributorsList"
            );

        if (!container) {
            return;
        }

        const normalised =
            normaliseContributors(
                contributors
            );

        container.innerHTML = "";

        if (
            normalised.length === 0
        ) {
            container.innerHTML = `
                <p class="muted-text">
                    No contributor data
                    available.
                </p>
            `;

            return;
        }

        normalised.forEach(
            (item, index) => {
                const row =
                    document
                        .createElement(
                            "div"
                        );

                row.className =
                    "contributor-row";

                row.innerHTML = `
                    <span
                      class="contributor-rank"
                    >
                        ${index + 1}
                    </span>

                    <div>
                        <strong>
                            ${escapeHtml(
                                item.feature
                            )}
                        </strong>

                        <p>
                            Error:
                            ${escapeHtml(
                                item.error
                            )}
                        </p>

                        <p>
                            Scaled value:
                            ${escapeHtml(
                                item.scaledValue
                            )}
                        </p>

                        <p>
                            Reconstructed value:
                            ${escapeHtml(
                                item
                                    .reconstructedValue
                            )}
                        </p>
                    </div>
                `;

                container
                    .appendChild(row);
            }
        );
    }

    function explanationParagraphs(
        text
    ) {
        const clean =
            cleanText(text);

        return clean
            .split(/\n\s*\n/)
            .map(
                (paragraph) =>
                    `<p>${escapeHtml(
                        paragraph
                    )}</p>`
            )
            .join("");
    }

    function renderLayeredExplanation(
        alert
    ) {
        const container =
            document.getElementById(
                "layeredExplanation"
            );

        if (!container) {
            return;
        }

        const simple =
            cleanText(
                alert
                    .simple_explanation ||
                alert.explanation ||
                alert.short_reason ||
                alert.full_explanation
            );

        const analyst =
            cleanText(
                alert
                    .analyst_explanation ||
                alert
                    .possible_explanation ||
                alert.what_to_check ||
                simple
            );

        const technical =
            cleanText(
                alert
                    .technical_explanation ||
                alert.full_explanation ||
                simple
            );

        const explanations = {
            simple:
                simple,

            analyst:
                analyst,

            technical:
                technical,
        };

        container.innerHTML = `
            <div
              class="explanation-tabs"
              role="tablist"
              aria-label="Explanation level"
            >
                <button
                  class="explanation-tab active"
                  type="button"
                  data-layer="simple"
                >
                    Simple
                </button>

                <button
                  class="explanation-tab"
                  type="button"
                  data-layer="analyst"
                >
                    Analyst
                </button>

                <button
                  class="explanation-tab"
                  type="button"
                  data-layer="technical"
                >
                    Technical
                </button>
            </div>

            <div
              class="explanation-layer-body"
            >
                ${explanationParagraphs(
                    explanations.simple
                )}
            </div>
        `;

        const buttons =
            container.querySelectorAll(
                ".explanation-tab"
            );

        const body =
            container.querySelector(
                ".explanation-layer-body"
            );

        buttons.forEach(
            (button) => {
                button.addEventListener(
                    "click",
                    () => {
                        const layer =
                            button
                                .dataset
                                .layer ||
                            "simple";

                        buttons.forEach(
                            (tabButton) =>
                                tabButton
                                    .classList
                                    .remove(
                                        "active"
                                    )
                        );

                        button.classList.add(
                            "active"
                        );

                        body.innerHTML =
                            explanationParagraphs(
                                explanations[
                                    layer
                                ] ||
                                explanations
                                    .simple
                            );
                    }
                );
            }
        );
    }

    let blocklistDangerousConfirmation = false;

    function setSourceBlocklistMessage(
        message,
        type = ""
    ) {
        const element = document.getElementById(
            "sourceBlocklistMessage"
        );

        if (!element) {
            return;
        }

        element.textContent = message || "";
        element.className = "blocklist-message";

        if (type) {
            element.classList.add(
                `blocklist-message-${type}`
            );
        }
    }

    function renderSourceBlocklistState(alert) {
        const blocklistState =
            alert?.blocklist_state &&
            typeof alert.blocklist_state === "object"
                ? alert.blocklist_state
                : {};

        const isBlocked = Boolean(
            blocklistState.is_blocked
        );

        const badge = document.getElementById(
            "sourceBlocklistBadge"
        );

        const addButton = document.getElementById(
            "addSourceToBlocklistBtn"
        );

        const pageLink = document.getElementById(
            "openBlocklistPageLink"
        );

        const demoWarning = document.getElementById(
            "blocklistDemoWarning"
        );

        setText(
            "blocklistSourceIp",
            alert.source_ip
        );

        setText(
            "blocklistSourceEnforcement",
            blocklistState.enforcement_state
                ? String(
                    blocklistState.enforcement_state
                ).replaceAll("_", " ")
                : "Not enforced"
        );

        setText(
            "blocklistSourceNotice",
            blocklistState.enforcement_message ||
            "Stored in the operator blocklist. Firewall enforcement is not enabled."
        );

        demoWarning?.classList.toggle(
            "hidden",
            !Boolean(alert.is_demo_alert)
        );

        if (badge) {
            badge.textContent = isBlocked
                ? "Source IP is in the blocklist"
                : "Not in blocklist";

            badge.className =
                "blocklist-source-badge " +
                (
                    isBlocked
                        ? "blocklist-source-listed"
                        : "blocklist-source-not-listed"
                );
        }

        if (addButton) {
            addButton.disabled = isBlocked;
            addButton.textContent = isBlocked
                ? "Source IP is in the blocklist"
                : "Add source IP to blocklist";
        }

        pageLink?.classList.toggle(
            "hidden",
            !isBlocked
        );

        if (isBlocked) {
            const record = blocklistState.record || {};
            const createdAt = record.created_at
                ? formatStatusTimestamp(record.created_at)
                : "";

            setSourceBlocklistMessage(
                createdAt
                    ? `Stored on ${createdAt}. ${blocklistState.enforcement_message || ""}`
                    : blocklistState.enforcement_message,
                "success"
            );
        } else {
            setSourceBlocklistMessage("");
        }
    }

    function closeAlertBlocklistDialog() {
        const dialog = document.getElementById(
            "alertBlocklistDialog"
        );

        if (dialog?.open) {
            dialog.close();
        }

        blocklistDangerousConfirmation = false;

        const warningContainer = document.getElementById(
            "alertBlocklistSafetyWarnings"
        );

        if (warningContainer) {
            warningContainer.classList.add("hidden");
            warningContainer.innerHTML = "";
        }
    }

    function openAlertBlocklistDialog() {
        if (!currentAlertDetail) {
            return;
        }

        blocklistDangerousConfirmation = false;

        setText(
            "confirmBlocklistIp",
            currentAlertDetail.source_ip
        );

        setText(
            "confirmBlocklistDecision",
            currentAlertDetail.final_decision
        );

        const dialog = document.getElementById(
            "alertBlocklistDialog"
        );

        if (typeof dialog?.showModal === "function") {
            dialog.showModal();
        } else {
            dialog?.setAttribute("open", "");
        }
    }

    function showAlertBlocklistSafetyWarnings(payload) {
        const warningContainer = document.getElementById(
            "alertBlocklistSafetyWarnings"
        );

        if (!warningContainer) {
            return;
        }

        const warnings = [
            ...(payload.confirmation_reasons || []),
            ...(payload.warnings || []),
        ];

        warningContainer.innerHTML = `
            <strong>Explicit confirmation is required</strong>
            <ul>
                ${warnings
                    .map(
                        (warning) =>
                            `<li>${escapeHtml(warning)}</li>`
                    )
                    .join("")}
            </ul>
        `;

        warningContainer.classList.remove("hidden");

        const confirmButton = document.getElementById(
            "confirmAlertBlocklistBtn"
        );

        if (confirmButton) {
            confirmButton.textContent =
                "Confirm dangerous address";
        }
    }

    async function submitSourceBlocklistAction() {
        if (!currentAlertDetail || !alertId) {
            return;
        }

        const confirmButton = document.getElementById(
            "confirmAlertBlocklistBtn"
        );

        const reasonInput = document.getElementById(
            "confirmBlocklistReason"
        );

        const reason = String(
            reasonInput?.value || ""
        ).trim();

        if (!reason) {
            setSourceBlocklistMessage(
                "Enter a reason before adding the address.",
                "error"
            );
            return;
        }

        if (confirmButton) {
            confirmButton.disabled = true;
            confirmButton.textContent = "Saving…";
        }

        try {
            const response = await fetch(
                "/blocklist",
                {
                    method: "POST",
                    headers: {
                        Accept: "application/json",
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        ip_address:
                            currentAlertDetail.source_ip,
                        reason,
                        source_alert_id: alertId,
                        confirm_dangerous:
                            blocklistDangerousConfirmation,
                        update_existing_reason: false,
                    }),
                }
            );

            let payload = {};

            try {
                payload = await response.json();
            } catch (parseError) {
                payload = {};
            }

            if (
                response.status === 409 &&
                payload.confirmation_required
            ) {
                blocklistDangerousConfirmation = true;
                showAlertBlocklistSafetyWarnings(payload);
                return;
            }

            if (!response.ok) {
                const detail = payload.detail;

                throw new Error(
                    typeof detail === "string"
                        ? detail
                        : detail?.message ||
                          "The source IP could not be added."
                );
            }

            closeAlertBlocklistDialog();

            currentAlertDetail.blocklist_state = {
                is_blocked: true,
                record: payload.record,
                enforcement_state:
                    payload.enforcement_state,
                firewall_enforcement_enabled:
                    payload.firewall_enforcement_enabled,
                enforcement_message:
                    payload.enforcement_message,
            };

            renderSourceBlocklistState(
                currentAlertDetail
            );

            setSourceBlocklistMessage(
                `${payload.message} ${payload.enforcement_message}`,
                "success"
            );
        } catch (error) {
            console.error(
                "Failed to add source IP to blocklist:",
                error
            );

            setSourceBlocklistMessage(
                error.message ||
                "The source IP could not be added.",
                "error"
            );
        } finally {
            if (confirmButton) {
                confirmButton.disabled = false;

                if (!blocklistDangerousConfirmation) {
                    confirmButton.textContent = "Add address";
                }
            }
        }
    }

    function configureSourceBlocklistControls() {
        document.getElementById(
            "addSourceToBlocklistBtn"
        )?.addEventListener(
            "click",
            openAlertBlocklistDialog
        );

        document.getElementById(
            "closeAlertBlocklistDialogBtn"
        )?.addEventListener(
            "click",
            closeAlertBlocklistDialog
        );

        document.getElementById(
            "cancelAlertBlocklistBtn"
        )?.addEventListener(
            "click",
            closeAlertBlocklistDialog
        );

        document.getElementById(
            "confirmAlertBlocklistBtn"
        )?.addEventListener(
            "click",
            submitSourceBlocklistAction
        );

        document.getElementById(
            "alertBlocklistDialog"
        )?.addEventListener(
            "cancel",
            (event) => {
                event.preventDefault();
                closeAlertBlocklistDialog();
            }
        );
    }

    function renderAlert(alert) {
        showContent();
        currentAlertDetail = alert;

        const finalDecision =
            typeof getPreviewFinalDecision ===
            "function"
                ? getPreviewFinalDecision(
                    alert
                )
                : alert.final_decision;

        const rawSeverity =
            typeof getPreviewRawSeverity ===
            "function"
                ? getPreviewRawSeverity(
                    alert
                )
                : alert
                    .raw_model_severity;

        setBadge(
            "finalDecisionBadge",
            finalDecision
        );

        setBadge(
            "rawSeverityBadge",
            rawSeverity
        );

        setInnerHtml(
            "detailRawFinalComparison",

            typeof renderRawFinalBadge ===
            "function"
                ? renderRawFinalBadge(
                    rawSeverity,
                    finalDecision
                )
                : (
                    `${escapeHtml(
                        rawSeverity
                    )} → ` +
                    `${escapeHtml(
                        finalDecision
                    )}`
                )
        );

        setText(
            "anomalyScore",
            alert.anomaly_score ??
            alert.ae_score
        );

        setText(
            "shortReason",
            alert.short_reason
        );

        setText(
            "contextualReason",
            alert.contextual_reason
        );

        setText(
            "recommendedAction",
            alert.recommended_action ||
            (
                "No recommended check " +
                "is available for this " +
                "alert."
            )
        );

        setText(
            "sourceIp",
            alert.source_ip
        );

        setText(
            "sourcePort",
            alert.source_port
        );

        setText(
            "destinationIp",
            alert.destination_ip
        );

        setText(
            "destinationPort",
            alert.destination_port
        );

        setText(
            "protocol",
            alert.protocol
        );

        setText(
            "trafficClass",
            alert.traffic_class
        );

        setText(
            "repeatCount",
            alert.repeat_count
        );

        renderProcessAttribution(alert);
        renderTemporalContext(alert);

        renderHumanStatus(
            alert.human_status,
            alert
                .human_status_updated_at
        );

        renderFeedbackState(
            alert.feedback
        );

        renderSourceBlocklistState(alert);

        renderContributorChart(
            alert
                .top_anomaly_contributors
        );

        renderContributors(
            alert
                .top_anomaly_contributors
        );

        renderLayeredExplanation(
            alert
        );
    }

    function configureJsonReportDownload() {
        const button =
            document.getElementById(
                "downloadJsonReportBtn"
            );

        if (!button) {
            return;
        }

        if (!alertId) {
            button.classList.add(
                "hidden"
            );

            button.setAttribute(
                "aria-disabled",
                "true"
            );

            return;
        }

        button.href =
            `/alerts/${encodeURIComponent(
                alertId
            )}/report.json`;
    }

    async function loadAlertDetail() {
        if (!alertId) {
            showNotFound();
            return;
        }

        try {
            const response =
                await fetch(
                    `/alerts/${encodeURIComponent(
                        alertId
                    )}/detail`
                );

            if (!response.ok) {
                showNotFound();
                return;
            }

            const alert =
                await response.json();

            renderAlert(alert);
        } catch (error) {
            console.error(
                "Failed to load alert " +
                "detail:",
                error
            );

            showNotFound();
        }
    }

    document.addEventListener(
        "DOMContentLoaded",
        () => {
            configureJsonReportDownload();
            configureAlertStatusControls();
            configureFeedbackButton();
            configureSourceBlocklistControls();
            loadAlertDetail();
        }
    );
})();