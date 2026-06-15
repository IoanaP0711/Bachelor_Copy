(function () {
    "use strict";

    const DEFAULT_ENFORCEMENT_MESSAGE =
        "Stored in the operator blocklist. Firewall enforcement is not enabled.";

    const state = {
        records: [],
        pendingAction: null,
        dangerousConfirmation: false,
        safetyWarnings: [],
        enforcementMessage: DEFAULT_ENFORCEMENT_MESSAGE,
    };

    function byId(id) {
        return document.getElementById(id);
    }

    function text(value, fallback = "-") {
        if (
            value === null ||
            value === undefined ||
            String(value).trim() === ""
        ) {
            return fallback;
        }

        return String(value);
    }

    function escapeHtml(value) {
        return text(value, "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function formatTimestamp(value) {
        if (!value) {
            return "-";
        }

        const date = new Date(value);

        if (Number.isNaN(date.getTime())) {
            return text(value);
        }

        return date.toLocaleString();
    }

    function setMessage(message, type = "") {
        const element = byId("blocklistFormMessage");

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

    function getResponseError(payload, fallback) {
        const detail = payload?.detail;

        if (typeof detail === "string") {
            return detail;
        }

        if (
            detail &&
            typeof detail === "object"
        ) {
            return (
                detail.message ||
                fallback
            );
        }

        return (
            payload?.message ||
            fallback
        );
    }

    async function readJson(response) {
        try {
            return await response.json();
        } catch (error) {
            return {};
        }
    }

    function enforcementBadge(record) {
        const value = text(
            record?.enforcement_state,
            "not_enforced"
        );

        return `
            <span
                class="
                    blocklist-enforcement-badge
                    blocklist-enforcement-not-enforced
                "
            >
                ${escapeHtml(
                    value.replaceAll("_", " ")
                )}
            </span>
        `;
    }

    function addressTypeBadge(record) {
        return `
            <span class="blocklist-type-badge">
                ${escapeHtml(
                    text(record?.address_type)
                )}
            </span>
        `;
    }

    function decisionBadge(record) {
        const decision = text(
            record?.source_final_decision,
            ""
        ).toUpperCase();

        if (!decision) {
            return `
                <span class="muted-text">
                    -
                </span>
            `;
        }

        const allowedClasses = {
            OK: "badge-ok",
            BENIGN: "badge-benign",
            REVIEW: "badge-review",
            CRITICAL: "badge-critical",
        };

        const className =
            allowedClasses[decision] ||
            "badge-muted";

        return `
            <span class="badge ${className}">
                ${escapeHtml(decision)}
            </span>
        `;
    }

    function associatedAlert(record) {
        const alertId = text(
            record?.source_alert_id,
            ""
        );

        if (!alertId) {
            return `
                <span class="muted-text">
                    Manual entry
                </span>
            `;
        }

        const demoBadge = record?.demo_derived
            ? `
                <span class="blocklist-demo-badge">
                    Demo-derived
                </span>
            `
            : "";

        return `
            <div class="blocklist-alert-reference">
                <a
                    href="/ui/alerts/${encodeURIComponent(
                        alertId
                    )}"
                >
                    ${escapeHtml(alertId)}
                </a>

                ${demoBadge}
            </div>
        `;
    }

    function getFilteredRecords() {
        const search = text(
            byId("blocklistSearch")?.value,
            ""
        )
            .trim()
            .toLowerCase();

        const addressType = text(
            byId(
                "blocklistAddressTypeFilter"
            )?.value,
            "ALL"
        );

        const enforcementState = text(
            byId(
                "blocklistEnforcementFilter"
            )?.value,
            "ALL"
        );

        return state.records.filter(
            (record) => {
                const typeMatches =
                    addressType === "ALL" ||
                    record.address_type ===
                        addressType;

                const enforcementMatches =
                    enforcementState ===
                        "ALL" ||
                    record.enforcement_state ===
                        enforcementState;

                const searchableText = [
                    record.ip_address,
                    record.address_type,
                    record.reason,
                    record.source_alert_id,
                    record.source_final_decision,
                    record.source_origin,
                    record.enforcement_state,
                ]
                    .filter(Boolean)
                    .join(" ")
                    .toLowerCase();

                const searchMatches =
                    !search ||
                    searchableText.includes(
                        search
                    );

                return (
                    typeMatches &&
                    enforcementMatches &&
                    searchMatches
                );
            }
        );
    }

    function resetDisplayStates() {
        byId(
            "blocklistErrorState"
        )?.classList.add("hidden");

        byId(
            "blocklistEmptyState"
        )?.classList.add("hidden");

        byId(
            "blocklistNoMatchState"
        )?.classList.add("hidden");

        byId(
            "blocklistTableWrap"
        )?.classList.remove("hidden");

        byId(
            "blocklistMobileCards"
        )?.classList.remove("hidden");
    }

    function showLoadError(message) {
        resetDisplayStates();

        const errorElement = byId(
            "blocklistErrorState"
        );

        if (errorElement) {
            errorElement.textContent =
                message;

            errorElement.classList.remove(
                "hidden"
            );
        }

        byId(
            "blocklistTableWrap"
        )?.classList.add("hidden");

        byId(
            "blocklistMobileCards"
        )?.classList.add("hidden");
    }

    function renderTableRow(record) {
        const ipAddress = text(
            record.ip_address
        );

        return `
            <tr>
                <td
                    class="mono blocklist-ip-cell"
                    title="${escapeHtml(ipAddress)}"
                >
                    ${escapeHtml(ipAddress)}
                </td>

                <td>
                    ${addressTypeBadge(record)}
                </td>

                <td class="blocklist-reason-cell">
                    ${escapeHtml(
                        text(record.reason)
                    )}
                </td>

                <td class="mono">
                    ${escapeHtml(
                        formatTimestamp(
                            record.created_at
                        )
                    )}
                </td>

                <td>
                    ${associatedAlert(record)}
                </td>

                <td>
                    ${decisionBadge(record)}
                </td>

                <td>
                    ${enforcementBadge(record)}
                </td>

                <td class="action-sticky">
                    <button
                        type="button"
                        class="blocklist-unblock-btn"
                        data-unblock-ip="${escapeHtml(
                            ipAddress
                        )}"
                    >
                        Unblock
                    </button>
                </td>
            </tr>
        `;
    }

    function renderMobileCard(record) {
        const ipAddress = text(
            record.ip_address
        );

        return `
            <article
                class="
                    mobile-data-card
                    blocklist-mobile-card
                "
            >
                <div class="mobile-card-header">
                    <strong
                        class="
                            mono
                            blocklist-mobile-ip
                        "
                    >
                        ${escapeHtml(ipAddress)}
                    </strong>

                    ${addressTypeBadge(record)}
                </div>

                <p class="mobile-card-summary">
                    ${escapeHtml(
                        text(record.reason)
                    )}
                </p>

                <dl class="mobile-card-details-grid">
                    <div>
                        <dt>Added at</dt>

                        <dd>
                            ${escapeHtml(
                                formatTimestamp(
                                    record.created_at
                                )
                            )}
                        </dd>
                    </div>

                    <div>
                        <dt>Associated alert</dt>

                        <dd>
                            ${associatedAlert(record)}
                        </dd>
                    </div>

                    <div>
                        <dt>Source decision</dt>

                        <dd>
                            ${decisionBadge(record)}
                        </dd>
                    </div>

                    <div>
                        <dt>Enforcement</dt>

                        <dd>
                            ${enforcementBadge(record)}
                        </dd>
                    </div>
                </dl>

                <div
                    class="
                        mobile-card-actions
                        blocklist-mobile-actions
                    "
                >
                    <button
                        type="button"
                        class="
                            mobile-card-action
                            blocklist-unblock-btn
                        "
                        data-unblock-ip="${escapeHtml(
                            ipAddress
                        )}"
                    >
                        Unblock
                    </button>
                </div>
            </article>
        `;
    }

    function renderRecords() {
        resetDisplayStates();

        const visibleRecords =
            getFilteredRecords();

        const totalElement = byId(
            "blocklistTotal"
        );

        const visibleElement = byId(
            "blocklistVisibleTotal"
        );

        if (totalElement) {
            totalElement.textContent =
                String(
                    state.records.length
                );
        }

        if (visibleElement) {
            visibleElement.textContent =
                String(
                    visibleRecords.length
                );
        }

        if (!state.records.length) {
            byId(
                "blocklistEmptyState"
            )?.classList.remove("hidden");

            byId(
                "blocklistTableWrap"
            )?.classList.add("hidden");

            byId(
                "blocklistMobileCards"
            )?.classList.add("hidden");

            return;
        }

        if (!visibleRecords.length) {
            byId(
                "blocklistNoMatchState"
            )?.classList.remove("hidden");

            byId(
                "blocklistTableWrap"
            )?.classList.add("hidden");

            byId(
                "blocklistMobileCards"
            )?.classList.add("hidden");

            return;
        }

        const tableBody = byId(
            "blocklistRows"
        );

        if (tableBody) {
            tableBody.innerHTML =
                visibleRecords
                    .map(renderTableRow)
                    .join("");
        }

        const mobileCards = byId(
            "blocklistMobileCards"
        );

        if (mobileCards) {
            mobileCards.innerHTML =
                visibleRecords
                    .map(renderMobileCard)
                    .join("");
        }
    }

    function renderServerState(payload) {
        state.enforcementMessage =
            payload.enforcement_message ||
            DEFAULT_ENFORCEMENT_MESSAGE;

        const storageState = byId(
            "blocklistStorageState"
        );

        const enforcementState = byId(
            "blocklistEnforcementState"
        );

        const enforcementMessage = byId(
            "blocklistEnforcementMessage"
        );

        const adapterState = byId(
            "blocklistAdapterState"
        );

        if (storageState) {
            storageState.textContent =
                payload.storage_warning
                    ? "Loaded with a storage warning"
                    : "Stored records loaded";
        }

        if (enforcementState) {
            enforcementState.textContent =
                "Not enforced";
        }

        if (enforcementMessage) {
            enforcementMessage.textContent =
                state.enforcementMessage;
        }

        if (adapterState) {
            adapterState.textContent =
                payload
                    .firewall_enforcement_enabled
                    ? "Enabled"
                    : "Disabled";
        }

        if (payload.storage_warning) {
            setMessage(
                payload.storage_warning,
                "warning"
            );
        }
    }

    async function loadBlocklist() {
        const refreshButton = byId(
            "refreshBlocklistBtn"
        );

        if (refreshButton) {
            refreshButton.disabled = true;
            refreshButton.textContent =
                "Loading…";
        }

        try {
            const response = await fetch(
                "/blocklist",
                {
                    headers: {
                        Accept:
                            "application/json",
                    },
                }
            );

            const payload =
                await readJson(response);

            if (!response.ok) {
                throw new Error(
                    getResponseError(
                        payload,
                        "The blocklist could not be loaded."
                    )
                );
            }

            state.records =
                Array.isArray(
                    payload.records
                )
                    ? payload.records
                    : [];

            renderServerState(payload);
            renderRecords();
        } catch (error) {
            console.error(
                "Failed to load blocklist:",
                error
            );

            showLoadError(
                error.message ||
                "The blocklist could not be loaded."
            );
        } finally {
            if (refreshButton) {
                refreshButton.disabled =
                    false;

                refreshButton.textContent =
                    "Refresh";
            }
        }
    }

    function getDialog() {
        return byId(
            "blocklistConfirmDialog"
        );
    }

    function openDialog() {
        const dialog = getDialog();

        if (!dialog) {
            return;
        }

        if (
            typeof dialog.showModal ===
            "function"
        ) {
            if (!dialog.open) {
                dialog.showModal();
            }
        } else {
            dialog.setAttribute(
                "open",
                ""
            );
        }
    }

    function closeDialog() {
        const dialog = getDialog();

        if (dialog?.open) {
            dialog.close();
        } else {
            dialog?.removeAttribute(
                "open"
            );
        }

        state.pendingAction = null;
        state.dangerousConfirmation =
            false;
        state.safetyWarnings = [];
    }

    function renderAddConfirmation() {
        const pending =
            state.pendingAction;

        if (
            !pending ||
            pending.type !== "add"
        ) {
            return;
        }

        const payload =
            pending.payload;

        const title = byId(
            "blocklistConfirmTitle"
        );

        const body = byId(
            "blocklistConfirmBody"
        );

        const confirmButton = byId(
            "blocklistConfirmActionBtn"
        );

        if (title) {
            title.textContent =
                state.dangerousConfirmation
                    ? "Confirm dangerous address"
                    : "Add IP to blocklist";
        }

        const associatedAlert =
            payload.source_alert_id
                ? `
                    <div>
                        <span>
                            Associated alert
                        </span>

                        <strong class="mono">
                            ${escapeHtml(
                                payload.source_alert_id
                            )}
                        </strong>
                    </div>
                `
                : "";

        const warnings =
            state.safetyWarnings.length
                ? `
                    <div
                        class="
                            blocklist-dialog-warning
                        "
                    >
                        <strong>
                            Additional safety warning
                        </strong>

                        <ul>
                            ${state.safetyWarnings
                                .map(
                                    (warning) => `
                                        <li>
                                            ${escapeHtml(
                                                warning
                                            )}
                                        </li>
                                    `
                                )
                                .join("")}
                        </ul>
                    </div>
                `
                : "";

        if (body) {
            body.innerHTML = `
                <div class="blocklist-confirm-grid">
                    <div>
                        <span>IP address</span>

                        <strong class="mono">
                            ${escapeHtml(
                                payload.ip_address
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>Reason</span>

                        <strong>
                            ${escapeHtml(
                                payload.reason
                            )}
                        </strong>
                    </div>

                    ${associatedAlert}

                    <div>
                        <span>
                            Firewall enforcement state
                        </span>

                        <strong>
                            Not enforced
                        </strong>
                    </div>
                </div>

                <p class="blocklist-dialog-caution">
                    A detection result is not proof that
                    an attack occurred. Confirm the
                    address and its network role before
                    continuing.
                </p>

                <p class="blocklist-dialog-caution">
                    ${escapeHtml(
                        state.enforcementMessage
                    )}
                </p>

                ${warnings}
            `;
        }

        if (confirmButton) {
            confirmButton.textContent =
                state.dangerousConfirmation
                    ? "Confirm dangerous address"
                    : "Add address";
        }

        openDialog();
    }

    function beginAddAction(payload) {
        state.pendingAction = {
            type: "add",
            payload,
        };

        state.dangerousConfirmation =
            false;
        state.safetyWarnings = [];

        renderAddConfirmation();
    }

    function beginRemoveAction(
        ipAddress
    ) {
        const record =
            state.records.find(
                (item) =>
                    item.ip_address ===
                    ipAddress
            );

        state.pendingAction = {
            type: "remove",
            payload: {
                ip_address: ipAddress,
            },
        };

        const title = byId(
            "blocklistConfirmTitle"
        );

        const body = byId(
            "blocklistConfirmBody"
        );

        const confirmButton = byId(
            "blocklistConfirmActionBtn"
        );

        if (title) {
            title.textContent =
                "Remove IP from blocklist";
        }

        if (body) {
            body.innerHTML = `
                <div class="blocklist-confirm-grid">
                    <div>
                        <span>IP address</span>

                        <strong class="mono">
                            ${escapeHtml(
                                ipAddress
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>Stored reason</span>

                        <strong>
                            ${escapeHtml(
                                record?.reason ||
                                "-"
                            )}
                        </strong>
                    </div>
                </div>

                <p class="blocklist-dialog-caution">
                    This removes the operator blocklist
                    record. No firewall rule will be
                    removed because firewall enforcement
                    is disabled.
                </p>
            `;
        }

        if (confirmButton) {
            confirmButton.textContent =
                "Unblock address";
        }

        openDialog();
    }

    function buildResultMessage(
        payload
    ) {
        return [
            payload.message,
            payload.enforcement_message,
            payload.storage_warning,
            payload.audit_warning,
        ]
            .filter(Boolean)
            .join(" ");
    }

    async function submitAddAction(
        pending
    ) {
        const confirmButton = byId(
            "blocklistConfirmActionBtn"
        );

        const addButton = byId(
            "blocklistAddBtn"
        );

        if (confirmButton) {
            confirmButton.disabled = true;
            confirmButton.textContent =
                "Saving…";
        }

        if (addButton) {
            addButton.disabled = true;
        }

        try {
            const response = await fetch(
                "/blocklist",
                {
                    method: "POST",

                    headers: {
                        Accept:
                            "application/json",

                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify({
                        ...pending.payload,

                        confirm_dangerous:
                            state
                                .dangerousConfirmation,

                        update_existing_reason:
                            false,
                    }),
                }
            );

            const payload =
                await readJson(response);

            if (
                response.status === 409 &&
                payload.confirmation_required
            ) {
                state.dangerousConfirmation =
                    true;

                state.safetyWarnings = [
                    ...(
                        payload
                            .confirmation_reasons ||
                        []
                    ),

                    ...(
                        payload.warnings ||
                        []
                    ),
                ];

                renderAddConfirmation();
                return;
            }

            if (!response.ok) {
                throw new Error(
                    getResponseError(
                        payload,
                        "The IP address could not be added."
                    )
                );
            }

            closeDialog();

            setMessage(
                buildResultMessage(
                    payload
                ),
                "success"
            );

            if (
                !payload.already_blocked
            ) {
                byId(
                    "blocklistAddForm"
                )?.reset();
            }

            await loadBlocklist();
        } catch (error) {
            console.error(
                "Failed to add blocklist record:",
                error
            );

            setMessage(
                error.message ||
                "The IP address could not be added.",
                "error"
            );
        } finally {
            if (addButton) {
                addButton.disabled = false;
            }

            if (confirmButton) {
                confirmButton.disabled =
                    false;

                if (
                    getDialog()?.open &&
                    state.pendingAction
                        ?.type === "add"
                ) {
                    confirmButton.textContent =
                        state
                            .dangerousConfirmation
                            ? "Confirm dangerous address"
                            : "Add address";
                }
            }
        }
    }

    async function submitRemoveAction(
        pending
    ) {
        const ipAddress =
            pending.payload.ip_address;

        const confirmButton = byId(
            "blocklistConfirmActionBtn"
        );

        if (confirmButton) {
            confirmButton.disabled = true;
            confirmButton.textContent =
                "Removing…";
        }

        try {
            const response = await fetch(
                `/blocklist/${encodeURIComponent(
                    ipAddress
                )}`,
                {
                    method: "DELETE",

                    headers: {
                        Accept:
                            "application/json",
                    },
                }
            );

            const payload =
                await readJson(response);

            if (!response.ok) {
                throw new Error(
                    getResponseError(
                        payload,
                        "The IP address could not be removed."
                    )
                );
            }

            closeDialog();

            setMessage(
                buildResultMessage(
                    payload
                ),
                "success"
            );

            await loadBlocklist();
        } catch (error) {
            console.error(
                "Failed to remove blocklist record:",
                error
            );

            setMessage(
                error.message ||
                "The IP address could not be removed.",
                "error"
            );
        } finally {
            if (confirmButton) {
                confirmButton.disabled =
                    false;
            }
        }
    }

    async function confirmPendingAction() {
        const pending =
            state.pendingAction;

        if (!pending) {
            return;
        }

        if (pending.type === "add") {
            await submitAddAction(
                pending
            );
            return;
        }

        if (pending.type === "remove") {
            await submitRemoveAction(
                pending
            );
        }
    }

    function configureAddForm() {
        byId(
            "blocklistAddForm"
        )?.addEventListener(
            "submit",
            (event) => {
                event.preventDefault();

                setMessage("");

                const ipAddress = text(
                    byId(
                        "blocklistIpAddress"
                    )?.value,
                    ""
                ).trim();

                const reason = text(
                    byId(
                        "blocklistReason"
                    )?.value,
                    ""
                ).trim();

                const sourceAlertId = text(
                    byId(
                        "blocklistSourceAlertId"
                    )?.value,
                    ""
                ).trim();

                if (
                    !ipAddress ||
                    !reason
                ) {
                    setMessage(
                        "Enter both an IP address and a reason.",
                        "error"
                    );

                    return;
                }

                beginAddAction({
                    ip_address: ipAddress,
                    reason,
                    source_alert_id:
                        sourceAlertId ||
                        null,
                });
            }
        );
    }

    function configureDialog() {
        document
            .querySelectorAll(
                "[data-dialog-cancel]"
            )
            .forEach((button) => {
                button.addEventListener(
                    "click",
                    closeDialog
                );
            });

        byId(
            "blocklistConfirmActionBtn"
        )?.addEventListener(
            "click",
            confirmPendingAction
        );

        getDialog()?.addEventListener(
            "cancel",
            (event) => {
                event.preventDefault();
                closeDialog();
            }
        );
    }

    function configureRecordActions() {
        document.addEventListener(
            "click",
            (event) => {
                const button =
                    event.target.closest(
                        "[data-unblock-ip]"
                    );

                if (!button) {
                    return;
                }

                const ipAddress =
                    button.dataset
                        .unblockIp ||
                    "";

                if (!ipAddress) {
                    return;
                }

                beginRemoveAction(
                    ipAddress
                );
            }
        );
    }

    function configureFilters() {
        const filterIds = [
            "blocklistSearch",
            "blocklistAddressTypeFilter",
            "blocklistEnforcementFilter",
        ];

        filterIds.forEach((id) => {
            const element = byId(id);

            element?.addEventListener(
                "input",
                renderRecords
            );

            element?.addEventListener(
                "change",
                renderRecords
            );
        });
    }

    document.addEventListener(
        "DOMContentLoaded",
        () => {
            configureAddForm();
            configureDialog();
            configureRecordActions();
            configureFilters();

            byId(
                "refreshBlocklistBtn"
            )?.addEventListener(
                "click",
                loadBlocklist
            );

            loadBlocklist();
        }
    );
})();