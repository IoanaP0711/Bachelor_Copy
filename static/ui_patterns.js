"use strict";

const PATTERNS_REFRESH_INTERVAL_MS = 5000;

const patternUi = {
    loading: document.getElementById(
        "patternsLoading"
    ),

    error: document.getElementById(
        "patternsError"
    ),

    empty: document.getElementById(
        "patternsEmpty"
    ),

    tableWrapper: document.getElementById(
        "patternsTableWrapper"
    ),

    tableBody: document.getElementById(
        "patternsTableBody"
    ),

    cards: document.getElementById(
        "patternsCards"
    ),

    count: document.getElementById(
        "patternsCount"
    ),

    recordsCount: document.getElementById(
        "patternsRecordsCount"
    ),

    source: document.getElementById(
        "patternsSource"
    ),

    updated: document.getElementById(
        "patternsUpdated"
    ),

    refreshButton: document.getElementById(
        "refreshPatternsButton"
    ),
};

let patternRequestRunning = false;
let patternsLoaded = false;


function escapePatternHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function patternValue(
    value,
    fallback = "Unknown"
) {
    const text = String(
        value ?? ""
    ).trim();

    return text || fallback;
}


function patternClassToken(value) {
    return patternValue(
        value,
        "unknown"
    )
        .toLowerCase()
        .replaceAll(
            /[^a-z0-9_-]+/g,
            "-"
        );
}


function formatPatternTimestamp(value) {
    if (
        value === null
        || value === undefined
        || value === ""
    ) {
        return "Unavailable";
    }

    let date;

    const numericText = String(value);

    if (
        typeof value === "number"
        || /^\d+(\.\d+)?$/.test(
            numericText
        )
    ) {
        const numeric = Number(value);

        const milliseconds = (
            numeric < 1_000_000_000_000
                ? numeric * 1000
                : numeric
        );

        date = new Date(
            milliseconds
        );
    } else {
        date = new Date(
            value
        );
    }

    if (
        Number.isNaN(
            date.getTime()
        )
    ) {
        return "Unavailable";
    }

    return date.toLocaleString(
        [],
        {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        }
    );
}


function formatTrafficClass(value) {
    return patternValue(
        value
    ).replaceAll(
        "_",
        " "
    );
}


function renderPatternBadge(
    value,
    kind
) {
    const text = patternValue(
        value
    );

    const token = patternClassToken(
        text
    );

    return `
        <span
            class="pattern-badge
                   pattern-badge--${kind}-${token}"
        >
            ${escapePatternHtml(text)}
        </span>
    `;
}


function renderPatternRow(pattern) {
    const id = patternValue(
        pattern.pattern_id
    );

    const key = patternValue(
        pattern.pattern_key
    );

    const reason = patternValue(
        pattern.short_reason,
        "Repeated matching anomalous traffic."
    );

    const eventCount = patternValue(
        pattern.event_count,
        "0"
    );

    return `
        <tr>
            <td class="patterns-key-cell">
                <strong>
                    ${escapePatternHtml(id)}
                </strong>

                <span
                    title="${escapePatternHtml(key)}"
                >
                    ${escapePatternHtml(key)}
                </span>
            </td>

            <td>
                ${escapePatternHtml(
                    patternValue(
                        pattern.source_ip
                    )
                )}
            </td>

            <td>
                ${escapePatternHtml(
                    patternValue(
                        pattern.destination_ip
                    )
                )}
            </td>

            <td>
                ${escapePatternHtml(
                    patternValue(
                        pattern.destination_port
                    )
                )}
            </td>

            <td>
                ${escapePatternHtml(
                    patternValue(
                        pattern.protocol
                    )
                )}
            </td>

            <td>
                ${escapePatternHtml(
                    formatTrafficClass(
                        pattern.traffic_class
                    )
                )}
            </td>

            <td>
                <span class="patterns-count-badge">
                    ${escapePatternHtml(eventCount)}
                </span>
            </td>

            <td>
                ${renderPatternBadge(
                    pattern.highest_raw_severity,
                    "raw"
                )}
            </td>

            <td>
                ${renderPatternBadge(
                    pattern.highest_final_decision,
                    "final"
                )}
            </td>

            <td>
                ${escapePatternHtml(
                    formatPatternTimestamp(
                        pattern.first_seen
                    )
                )}
            </td>

            <td>
                ${escapePatternHtml(
                    formatPatternTimestamp(
                        pattern.last_seen
                    )
                )}
            </td>

            <td
                class="patterns-reason-cell"
                title="${escapePatternHtml(reason)}"
            >
                ${escapePatternHtml(reason)}
            </td>
        </tr>
    `;
}

function renderPatternCard(pattern) {
    const id = patternValue(
        pattern.pattern_id
    );

    const key = patternValue(
        pattern.pattern_key
    );

    const reason = patternValue(
        pattern.short_reason,
        "Repeated matching anomalous traffic."
    );

    return `
        <article class="pattern-card">
            <div class="pattern-card-header">
                <div>
                    <span class="pattern-card-label">
                        Pattern ID
                    </span>

                    <strong class="pattern-card-id">
                        ${escapePatternHtml(id)}
                    </strong>
                </div>

                <span class="patterns-count-badge">
                    ${escapePatternHtml(
                        patternValue(
                            pattern.event_count,
                            "0"
                        )
                    )}
                    events
                </span>
            </div>

            <div class="pattern-card-key">
                ${escapePatternHtml(key)}
            </div>

            <dl class="pattern-card-grid">
                <div>
                    <dt>Source</dt>
                    <dd>
                        ${escapePatternHtml(
                            patternValue(
                                pattern.source_ip
                            )
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Destination</dt>
                    <dd>
                        ${escapePatternHtml(
                            patternValue(
                                pattern.destination_ip
                            )
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Destination port</dt>
                    <dd>
                        ${escapePatternHtml(
                            patternValue(
                                pattern.destination_port
                            )
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Protocol</dt>
                    <dd>
                        ${escapePatternHtml(
                            patternValue(
                                pattern.protocol
                            )
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Traffic class</dt>
                    <dd>
                        ${escapePatternHtml(
                            formatTrafficClass(
                                pattern.traffic_class
                            )
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Highest raw severity</dt>
                    <dd>
                        ${renderPatternBadge(
                            pattern.highest_raw_severity,
                            "raw"
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Highest final decision</dt>
                    <dd>
                        ${renderPatternBadge(
                            pattern.highest_final_decision,
                            "final"
                        )}
                    </dd>
                </div>

                <div>
                    <dt>First seen</dt>
                    <dd>
                        ${escapePatternHtml(
                            formatPatternTimestamp(
                                pattern.first_seen
                            )
                        )}
                    </dd>
                </div>

                <div>
                    <dt>Last seen</dt>
                    <dd>
                        ${escapePatternHtml(
                            formatPatternTimestamp(
                                pattern.last_seen
                            )
                        )}
                    </dd>
                </div>

                <div class="pattern-card-reason">
                    <dt>Short reason</dt>
                    <dd>
                        ${escapePatternHtml(reason)}
                    </dd>
                </div>
            </dl>
        </article>
    `;
}

function showPatternState(
    state,
    message = ""
) {
    patternUi.loading.hidden = (
        state !== "loading"
    );

    patternUi.error.hidden = (
        state !== "error"
    );

    patternUi.empty.hidden = (
        state !== "empty"
    );

    patternUi.tableWrapper.hidden = (
        state !== "data"
    );

    patternUi.cards.hidden = (
        state !== "data"
    );

    if (state === "error") {
        patternUi.error.textContent = (
            message
        );
    }
}


function renderPatterns(payload) {
    const patterns = Array.isArray(
        payload.patterns
    )
        ? payload.patterns
        : [];

    patternUi.count.textContent = String(
        payload.pattern_count ?? 0
    );

    patternUi.recordsCount.textContent = String(
        payload.records_considered ?? 0
    );

    patternUi.source.textContent = patternValue(
        payload.source,
        "none"
    );

    patternUi.updated.textContent = (
        formatPatternTimestamp(
            payload.generated_at
        )
    );

    if (patterns.length === 0) {
        patternUi.tableBody.innerHTML = "";
        patternUi.cards.innerHTML = "";

        showPatternState(
            "empty"
        );

        return;
    }

    patternUi.tableBody.innerHTML = (
        patterns
            .map(
                renderPatternRow
            )
            .join("")
    );

    patternUi.cards.innerHTML = (
        patterns
            .map(
                renderPatternCard
            )
            .join("")
    );

    showPatternState(
        "data"
    );
}

async function loadPatterns() {
    if (patternRequestRunning) {
        return;
    }

    patternRequestRunning = true;

    patternUi.refreshButton.disabled = true;

    if (!patternsLoaded) {
        showPatternState(
            "loading"
        );
    }

    try {
        const response = await fetch(
            "/patterns?limit=200",
            {
                method: "GET",
                credentials: "same-origin",
                cache: "no-store",
                headers: {
                    Accept: "application/json",
                },
            }
        );

        if (response.status === 401) {
            window.location.href = "/login";
            return;
        }

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const payload = await response.json();

        renderPatterns(
            payload
        );

        patternsLoaded = true;

    } catch (error) {
        console.error(
            "Could not load repeated patterns:",
            error
        );

        if (!patternsLoaded) {
            showPatternState(
                "error",
                (
                    "Repeated patterns could not "
                    + "be loaded. Check the server "
                    + "log and try again."
                )
            );
        }

    } finally {
        patternRequestRunning = false;

        patternUi.refreshButton.disabled = false;
    }
}


patternUi.refreshButton.addEventListener(
    "click",
    loadPatterns
);

loadPatterns();


window.setInterval(
    () => {
        if (!document.hidden) {
            loadPatterns();
        }
    },
    PATTERNS_REFRESH_INTERVAL_MS
);