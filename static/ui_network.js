"use strict";

const NETWORK_SEARCH_DELAY_MS = 350;

const networkState = {
  cy: null,
  payload: null,
  requestId: 0,
  searchTimer: null,
  resizeObserver: null,
};

const networkUi = {};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function text(value, fallback = "Unavailable") {
  const cleaned = String(value ?? "").trim();
  return cleaned || fallback;
}

function boundedInteger(
  value,
  fallback,
  minimum,
  maximum
) {
  const parsed = Number.parseInt(
    String(value ?? ""),
    10
  );

  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return Math.max(
    minimum,
    Math.min(parsed, maximum)
  );
}

function formatNumber(
  value,
  maximumDigits = 6
) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return "Unavailable";
  }

  return parsed.toLocaleString(
    [],
    {
      minimumFractionDigits: 0,
      maximumFractionDigits:
        maximumDigits,
    }
  );
}

function formatTimestamp(value) {
  if (
    value === null
    || value === undefined
    || value === ""
  ) {
    return "Unavailable";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
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

function normalizeDecision(value) {
  const decision = text(
    value,
    "UNKNOWN"
  ).toUpperCase();

  if (
    [
      "OK",
      "BENIGN",
      "REVIEW",
      "CRITICAL",
    ].includes(decision)
  ) {
    return decision;
  }

  return "UNKNOWN";
}

function decisionBadge(
  value,
  prefix = ""
) {
  const decision =
    normalizeDecision(value);

  const className =
    decision === "UNKNOWN"
      ? "badge-muted"
      : `badge-${decision.toLowerCase()}`;

  const label = prefix
    ? `${prefix}: ${decision}`
    : decision;

  return `
    <span class="badge ${className}">
      ${escapeHtml(label)}
    </span>
  `;
}

function addressTypeLabel(value) {
  const normalized = text(
    value,
    "unknown"
  ).toLowerCase();

  return {
    loopback: "Loopback",
    local: "Private / local",
    external: "Public / external",
    unknown: "Unknown",
  }[normalized] || "Unknown";
}

function listText(values) {
  if (
    !Array.isArray(values)
    || values.length === 0
  ) {
    return "Unavailable";
  }

  return values
    .map((value) => String(value))
    .join(", ");
}

function demoBadge(data) {
  if (data?.is_demo) {
    return `
      <span
        class="
          network-demo-badge
          network-demo-only-badge
        "
      >
        Demo-only
      </span>
    `;
  }

  if (data?.contains_demo) {
    return `
      <span
        class="
          network-demo-badge
          network-mixed-demo-badge
        "
      >
        Contains demo records
      </span>
    `;
  }

  return `
    <span class="network-live-badge">
      Live/current-buffer records
    </span>
  `;
}

function decisionVisual(value) {
  const decision =
    normalizeDecision(value);

  return {
    OK: {
      nodeColor: "#dcfce7",
      borderColor: "#166534",
      lineColor: "#15803d",
      shape: "ellipse",
    },

    BENIGN: {
      nodeColor: "#cffafe",
      borderColor: "#0e7490",
      lineColor: "#0891b2",
      shape: "round-rectangle",
    },

    REVIEW: {
      nodeColor: "#fef3c7",
      borderColor: "#92400e",
      lineColor: "#d97706",
      shape: "diamond",
    },

    CRITICAL: {
      nodeColor: "#fee2e2",
      borderColor: "#991b1b",
      lineColor: "#dc2626",
      shape: "hexagon",
    },

    UNKNOWN: {
      nodeColor: "#e2e8f0",
      borderColor: "#475569",
      lineColor: "#64748b",
      shape: "rectangle",
    },
  }[decision];
}

function scaledValue(
  value,
  maximum,
  minimumSize,
  maximumSize
) {
  const safeValue = Math.max(
    1,
    Number(value) || 1
  );

  const safeMaximum = Math.max(
    1,
    Number(maximum) || 1
  );

  const ratio =
    Math.log1p(safeValue)
    / Math.log1p(safeMaximum);

  return minimumSize
    + ratio
    * (maximumSize - minimumSize);
}

function readFilters() {
  const minimumCount =
    boundedInteger(
      networkUi.minimumCount.value,
      1,
      1,
      100000
    );

  const maximumNodes =
    boundedInteger(
      networkUi.maximumNodes.value,
      120,
      2,
      500
    );

  const maximumEdges =
    boundedInteger(
      networkUi.maximumEdges.value,
      200,
      1,
      1000
    );

  networkUi.minimumCount.value =
    String(minimumCount);

  networkUi.maximumNodes.value =
    String(maximumNodes);

  networkUi.maximumEdges.value =
    String(maximumEdges);

  return {
    search:
      networkUi.search.value.trim(),

    decision:
      networkUi.decision.value,

    addressType:
      networkUi.addressType.value,

    minimumCount,
    maximumNodes,
    maximumEdges,

    includeDemo:
      networkUi.includeDemo.checked,
  };
}

function buildRequestUrl() {
  const filters = readFilters();

  const query = new URLSearchParams(
    {
      search:
        filters.search,

      decision:
        filters.decision,

      address_type:
        filters.addressType,

      min_edge_count:
        String(
          filters.minimumCount
        ),

      include_demo:
        filters.includeDemo
          ? "true"
          : "false",

      max_nodes:
        String(
          filters.maximumNodes
        ),

      max_edges:
        String(
          filters.maximumEdges
        ),
    }
  );

  return `/network/map?${query.toString()}`;
}

function destroyGraph() {
  if (
    networkState.resizeObserver
  ) {
    networkState.resizeObserver
      .disconnect();

    networkState.resizeObserver =
      null;
  }

  if (networkState.cy) {
    networkState.cy.destroy();
    networkState.cy = null;
  }
}

function setLoading(loading) {
  networkUi.refresh.disabled =
    loading;

  networkUi.resetFilters.disabled =
    loading;

  networkUi.resetView.disabled =
    loading;

  if (!loading) {
    return;
  }

  destroyGraph();

  networkUi.requestError.hidden =
    true;

  networkUi.libraryError.hidden =
    true;

  networkUi.empty.hidden =
    true;

  networkUi.graph.hidden =
    false;

  networkUi.graph.innerHTML = `
    <div class="network-state">
      Loading communication relationships...
    </div>
  `;

  networkUi.edgeList.innerHTML = `
    <div class="network-state">
      Loading communication relationships...
    </div>
  `;
}

function updateSummary(payload) {
  networkUi.nodeCount.textContent =
    String(
      payload.returned_node_count
      ?? payload.nodes?.length
      ?? 0
    );

  networkUi.edgeCount.textContent =
    String(
      payload.returned_edge_count
      ?? payload.edges?.length
      ?? 0
    );

  networkUi.recordCount.textContent =
    String(
      payload.records_used ?? 0
    );

  networkUi.source.textContent =
    text(
      payload.source,
      "None"
    );

  networkUi.source.title =
    text(
      payload.source_explanation,
      "The map uses the current IDS traffic buffers."
    );

  networkUi.scope.textContent =
    text(
      payload.scope_label,
      "Current buffer"
    );

  networkUi.generatedAt.textContent =
    formatTimestamp(
      payload.generated_at
    );

  networkUi.truncation.hidden =
    !payload.truncated;

  networkUi.truncation.textContent =
    payload.truncated
      ? text(
          payload.truncation_notice,
          "The map was limited to protect browser performance."
        )
      : "";
}

function defaultDetail() {
  networkUi.detail.innerHTML = `
    <div class="network-detail-placeholder">
      Select an IP address or communication
      relationship to view beginner-friendly
      details.
    </div>
  `;
}

function showNodeDetail(node) {
  networkUi.detail.innerHTML = `
    <div class="network-detail-title-row">
      <div>
        <span class="network-detail-type">
          Observed IP address
        </span>

        <h3 class="mono">
          ${escapeHtml(
            text(
              node.label || node.id
            )
          )}
        </h3>
      </div>

      ${decisionBadge(
        node.highest_decision
      )}
    </div>

    <div class="network-detail-badge-row">
      <span class="network-address-badge">
        ${escapeHtml(
          addressTypeLabel(
            node.address_type
          )
        )}
      </span>

      ${demoBadge(node)}
    </div>

    <dl class="network-detail-grid">
      <div>
        <dt>Address</dt>
        <dd class="mono">
          ${escapeHtml(
            text(node.id)
          )}
        </dd>
      </div>

      <div>
        <dt>Address type</dt>
        <dd>
          ${escapeHtml(
            addressTypeLabel(
              node.address_type
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Associated flows</dt>
        <dd>
          ${escapeHtml(
            text(
              node.flow_count,
              "0"
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Incoming records</dt>
        <dd>
          ${escapeHtml(
            text(
              node.incoming_count,
              "0"
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Outgoing records</dt>
        <dd>
          ${escapeHtml(
            text(
              node.outgoing_count,
              "0"
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Highest decision</dt>
        <dd>
          ${escapeHtml(
            normalizeDecision(
              node.highest_decision
            )
          )}
        </dd>
      </div>
    </dl>

    <p class="network-detail-note">
      Incoming and outgoing values count observed
      traffic records, not physical cables or devices.

      <a href="/ui/help#basic-networking-terms">
        Read basic networking terms.
      </a>
    </p>
  `;
}

function showEdgeDetail(edge) {
  const demoText =
    edge.is_demo
      ? (
          "This relationship contains only "
          + "replay/demo records."
        )
      : edge.contains_demo
        ? (
            "This relationship contains both "
            + "live/current-buffer and "
            + "replay/demo records."
          )
        : (
            "This relationship contains "
            + "live/current-buffer records only."
          );

  networkUi.detail.innerHTML = `
    <div class="network-detail-title-row">
      <div>
        <span class="network-detail-type">
          Directed communication relationship
        </span>

        <h3 class="mono">
          ${escapeHtml(
            text(edge.source)
          )}
          →
          ${escapeHtml(
            text(edge.target)
          )}
        </h3>
      </div>

      ${decisionBadge(
        edge.highest_decision
      )}
    </div>

    <div class="network-detail-badge-row">
      ${demoBadge(edge)}
    </div>

    <dl class="network-detail-grid">
      <div>
        <dt>Source address</dt>
        <dd class="mono">
          ${escapeHtml(
            text(edge.source)
          )}
        </dd>
      </div>

      <div>
        <dt>Destination address</dt>
        <dd class="mono">
          ${escapeHtml(
            text(edge.target)
          )}
        </dd>
      </div>

      <div>
        <dt>Observed records</dt>
        <dd>
          ${escapeHtml(
            text(
              edge.count,
              "0"
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Highest decision</dt>
        <dd>
          ${escapeHtml(
            normalizeDecision(
              edge.highest_decision
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Latest decision</dt>
        <dd>
          ${escapeHtml(
            normalizeDecision(
              edge.latest_decision
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Protocols</dt>
        <dd>
          ${escapeHtml(
            listText(
              edge.protocols
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Destination ports</dt>
        <dd>
          ${escapeHtml(
            listText(
              edge.ports
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Traffic classes</dt>
        <dd>
          ${escapeHtml(
            listText(
              edge.traffic_classes
            )
          )}
        </dd>
      </div>

      <div>
        <dt>First seen</dt>
        <dd>
          ${escapeHtml(
            formatTimestamp(
              edge.first_seen
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Last seen</dt>
        <dd>
          ${escapeHtml(
            formatTimestamp(
              edge.last_seen
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Average anomaly score</dt>
        <dd>
          ${escapeHtml(
            formatNumber(
              edge.average_anomaly_score
            )
          )}
        </dd>
      </div>

      <div>
        <dt>Maximum anomaly score</dt>
        <dd>
          ${escapeHtml(
            formatNumber(
              edge.maximum_anomaly_score
            )
          )}
        </dd>
      </div>
    </dl>

    <p class="network-detail-note">
      ${escapeHtml(demoText)}

      <a href="/ui/help#final-decisions">
        Read about final decisions.
      </a>
    </p>
  `;
}

function renderEdgeList(edges) {
  if (
    !Array.isArray(edges)
    || edges.length === 0
  ) {
    networkUi.edgeList.innerHTML = `
      <div class="network-state">
        No communication relationships are
        available for the selected filters.
      </div>
    `;

    return;
  }

  networkUi.edgeList.innerHTML =
    edges
      .map((edge) => {
        let demoLabel = "";

        if (edge.is_demo) {
          demoLabel = `
            <span
              class="
                network-demo-badge
                network-demo-only-badge
              "
            >
              Demo-only
            </span>
          `;
        } else if (
          edge.contains_demo
        ) {
          demoLabel = `
            <span
              class="
                network-demo-badge
                network-mixed-demo-badge
              "
            >
              Mixed live/demo
            </span>
          `;
        }

        return `
          <button
            type="button"
            class="network-edge-list-item"
            data-network-edge-id="${
              escapeHtml(edge.id)
            }"
          >
            <span
              class="
                network-edge-list-route
                mono
              "
            >
              ${escapeHtml(
                text(edge.source)
              )}

              <span aria-hidden="true">
                →
              </span>

              ${escapeHtml(
                text(edge.target)
              )}
            </span>

            <span
              class="network-edge-list-meta"
            >
              ${decisionBadge(
                edge.highest_decision
              )}

              <span>
                Observed records:
                ${escapeHtml(
                  text(
                    edge.count,
                    "0"
                  )
                )}
              </span>

              <span>
                Protocols:
                ${escapeHtml(
                  listText(
                    edge.protocols
                  )
                )}
              </span>

              ${demoLabel}
            </span>
          </button>
        `;
      })
      .join("");
}

function buildGraphElements(payload) {
  const nodes =
    Array.isArray(payload.nodes)
      ? payload.nodes
      : [];

  const edges =
    Array.isArray(payload.edges)
      ? payload.edges
      : [];

  const maximumNodeCount =
    Math.max(
      1,
      ...nodes.map(
        (node) =>
          Number(node.flow_count) || 1
      )
    );

  const maximumEdgeCount =
    Math.max(
      1,
      ...edges.map(
        (edge) =>
          Number(edge.count) || 1
      )
    );

  const nodeElements =
    nodes.map((node) => {
      const decision =
        normalizeDecision(
          node.highest_decision
        );

      const visual =
        decisionVisual(decision);

      const demoPrefix =
        node.is_demo
          ? "DEMO · "
          : "";

      return {
        group: "nodes",

        data: {
          ...node,

          displayLabel:
            `${demoPrefix}`
            + `${text(node.label)}\n`
            + `${decision}`,

          visualSize:
            scaledValue(
              node.flow_count,
              maximumNodeCount,
              40,
              82
            ),

          visualColor:
            visual.nodeColor,

          visualBorder:
            visual.borderColor,

          visualShape:
            visual.shape,

          visualBorderStyle:
            node.is_demo
              ? "dashed"
              : "solid",
        },
      };
    });

  const edgeElements =
    edges.map((edge) => {
      const decision =
        normalizeDecision(
          edge.highest_decision
        );

      const visual =
        decisionVisual(decision);

      const demoPrefix =
        edge.is_demo
          ? "DEMO · "
          : "";

      return {
        group: "edges",

        data: {
          ...edge,

          displayLabel:
            `${demoPrefix}`
            + `${text(
              edge.count,
              "0"
            )} × ${decision}`,

          visualWidth:
            scaledValue(
              edge.count,
              maximumEdgeCount,
              1.8,
              10
            ),

          visualLine:
            visual.lineColor,

          visualLineStyle:
            edge.is_demo
              ? "dashed"
              : "solid",
        },
      };
    });

  return [
    ...nodeElements,
    ...edgeElements,
  ];
}

function renderGraph(payload) {
  destroyGraph();

  const nodes =
    Array.isArray(payload.nodes)
      ? payload.nodes
      : [];

  const edges =
    Array.isArray(payload.edges)
      ? payload.edges
      : [];

  if (
    nodes.length === 0
    || edges.length === 0
  ) {
    networkUi.graph.hidden =
      true;

    networkUi.empty.hidden =
      false;

    networkUi.libraryError.hidden =
      true;

    return;
  }

  networkUi.graph.hidden =
    false;

  networkUi.empty.hidden =
    true;

  networkUi.graph.innerHTML =
    "";

  if (
    typeof window.cytoscape
    !== "function"
  ) {
    networkUi.libraryError.hidden =
      false;

    networkUi.graph.innerHTML = `
      <div
        class="
          network-state
          network-library-error
        "
      >
        The interactive map library could not
        be loaded. Use the textual relationship
        list below.
      </div>
    `;

    return;
  }

  networkUi.libraryError.hidden =
    true;

  networkState.cy =
    window.cytoscape(
      {
        container:
          networkUi.graph,

        elements:
          buildGraphElements(
            payload
          ),

        wheelSensitivity:
          0.18,

        minZoom:
          0.2,

        maxZoom:
          3.5,

        boxSelectionEnabled:
          false,

        userPanningEnabled:
          true,

        userZoomingEnabled:
          true,

        style: [
          {
            selector: "node",

            style: {
              width:
                "data(visualSize)",

              height:
                "data(visualSize)",

              shape:
                "data(visualShape)",

              "background-color":
                "data(visualColor)",

              "border-color":
                "data(visualBorder)",

              "border-width":
                3,

              "border-style":
                "data(visualBorderStyle)",

              label:
                "data(displayLabel)",

              color:
                "#0f172a",

              "font-size":
                10,

              "font-weight":
                700,

              "text-wrap":
                "wrap",

              "text-max-width":
                130,

              "text-valign":
                "center",

              "text-halign":
                "center",

              "overlay-opacity":
                0,
            },
          },

          {
            selector: "edge",

            style: {
              width:
                "data(visualWidth)",

              "line-color":
                "data(visualLine)",

              "target-arrow-color":
                "data(visualLine)",

              "target-arrow-shape":
                "triangle",

              "arrow-scale":
                1.15,

              "curve-style":
                "bezier",

              "line-style":
                "data(visualLineStyle)",

              label:
                "data(displayLabel)",

              color:
                "#334155",

              "font-size":
                9,

              "font-weight":
                700,

              "text-rotation":
                "autorotate",

              "text-background-color":
                "#ffffff",

              "text-background-opacity":
                0.88,

              "text-background-padding":
                2,

              "overlay-opacity":
                0,
            },
          },

          {
            selector: ":selected",

            style: {
              "border-color":
                "#111827",

              "border-width":
                5,

              "line-color":
                "#111827",

              "target-arrow-color":
                "#111827",

              "z-index":
                999,
            },
          },
        ],

        layout: {
          name:
            "cose",

          animate:
            false,

          randomize:
            true,

          fit:
            true,

          padding:
            45,

          nodeRepulsion:
            450000,

          idealEdgeLength:
            115,

          edgeElasticity:
            90,

          gravity:
            0.25,

          numIter:
            800,
        },
      }
    );

  networkState.cy.on(
    "tap",
    "node",
    (event) => {
      showNodeDetail(
        event.target.data()
      );
    }
  );

  networkState.cy.on(
    "tap",
    "edge",
    (event) => {
      showEdgeDetail(
        event.target.data()
      );
    }
  );

  networkState.cy.on(
    "tap",
    (event) => {
      if (
        event.target
        === networkState.cy
      ) {
        defaultDetail();
      }
    }
  );

  networkState.cy.ready(
    () => {
      networkState.cy.fit(
        networkState.cy.elements(),
        45
      );
    }
  );

  if (
    typeof ResizeObserver
    === "function"
  ) {
    networkState.resizeObserver =
      new ResizeObserver(
        () => {
          networkState.cy?.resize();
        }
      );

    networkState.resizeObserver
      .observe(
        networkUi.graph
      );
  }
}

function renderPayload(payload) {
  networkState.payload =
    payload;

  updateSummary(payload);

  defaultDetail();

  renderEdgeList(
    payload.edges || []
  );

  renderGraph(payload);
}

function renderRequestError(
  message
) {
  destroyGraph();

  networkState.payload =
    null;

  networkUi.requestError.hidden =
    false;

  networkUi.requestError.textContent =
    message;

  networkUi.truncation.hidden =
    true;

  networkUi.libraryError.hidden =
    true;

  networkUi.empty.hidden =
    true;

  networkUi.graph.hidden =
    false;

  networkUi.graph.innerHTML = `
    <div
      class="
        network-state
        network-request-error
      "
    >
      ${escapeHtml(message)}
    </div>
  `;

  networkUi.edgeList.innerHTML = `
    <div
      class="
        network-state
        network-request-error
      "
    >
      Communication relationships
      could not be loaded.
    </div>
  `;

  defaultDetail();
}

async function loadNetworkMap() {
  const requestId =
    ++networkState.requestId;

  setLoading(true);

  try {
    const response =
      await fetch(
        buildRequestUrl(),
        {
          cache:
            "no-store",

          credentials:
            "same-origin",

          headers: {
            Accept:
              "application/json",
          },
        }
      );

    if (
      response.status === 401
    ) {
      window.location.assign(
        "/login"
      );

      return;
    }

    if (!response.ok) {
      throw new Error(
        `Request failed with HTTP ${response.status}.`
      );
    }

    const payload =
      await response.json();

    if (
      requestId
      !== networkState.requestId
    ) {
      return;
    }

    networkUi.requestError.hidden =
      true;

    renderPayload(payload);
  } catch (error) {
    if (
      requestId
      !== networkState.requestId
    ) {
      return;
    }

    console.error(
      "Network map load error:",
      error
    );

    renderRequestError(
      "The network map request failed. "
      + "Check that the FastAPI server is "
      + "running and that /network/map "
      + "is available."
    );
  } finally {
    if (
      requestId
      === networkState.requestId
    ) {
      setLoading(false);
    }
  }
}

function resetFilters() {
  networkUi.search.value =
    "";

  networkUi.decision.value =
    "ALL";

  networkUi.addressType.value =
    "ALL";

  networkUi.minimumCount.value =
    "1";

  networkUi.maximumNodes.value =
    "120";

  networkUi.maximumEdges.value =
    "200";

  networkUi.includeDemo.checked =
    true;

  loadNetworkMap();
}

function resetView() {
  if (!networkState.cy) {
    return;
  }

  networkState.cy.resize();

  networkState.cy.fit(
    networkState.cy.elements(),
    45
  );
}

function selectEdgeFromList(
  edgeId
) {
  const edges =
    Array.isArray(
      networkState.payload?.edges
    )
      ? networkState.payload.edges
      : [];

  const edge =
    edges.find(
      (item) =>
        String(item.id)
        === String(edgeId)
    );

  if (!edge) {
    return;
  }

  showEdgeDetail(edge);

  if (networkState.cy) {
    networkState.cy
      .elements()
      .unselect();

    const graphEdge =
      networkState.cy
        .getElementById(
          String(edgeId)
        );

    if (graphEdge.length) {
      graphEdge.select();

      networkState.cy.center(
        graphEdge
      );
    }
  }
}

function initializeReferences() {
  Object.assign(
    networkUi,
    {
      refresh:
        byId(
          "refreshNetworkButton"
        ),

      resetFilters:
        byId(
          "resetNetworkFiltersButton"
        ),

      resetView:
        byId(
          "resetNetworkViewButton"
        ),

      search:
        byId(
          "networkSearchInput"
        ),

      decision:
        byId(
          "networkDecisionFilter"
        ),

      addressType:
        byId(
          "networkAddressFilter"
        ),

      minimumCount:
        byId(
          "networkMinimumCountInput"
        ),

      maximumNodes:
        byId(
          "networkMaximumNodesInput"
        ),

      maximumEdges:
        byId(
          "networkMaximumEdgesInput"
        ),

      includeDemo:
        byId(
          "networkIncludeDemoCheckbox"
        ),

      nodeCount:
        byId(
          "networkNodeCount"
        ),

      edgeCount:
        byId(
          "networkEdgeCount"
        ),

      recordCount:
        byId(
          "networkRecordCount"
        ),

      source:
        byId(
          "networkSourceLabel"
        ),

      scope:
        byId(
          "networkScopeLabel"
        ),

      generatedAt:
        byId(
          "networkGeneratedAt"
        ),

      truncation:
        byId(
          "networkTruncationNotice"
        ),

      requestError:
        byId(
          "networkRequestError"
        ),

      graph:
        byId(
          "networkGraph"
        ),

      libraryError:
        byId(
          "networkLibraryError"
        ),

      empty:
        byId(
          "networkEmptyState"
        ),

      detail:
        byId(
          "networkDetailContent"
        ),

      edgeList:
        byId(
          "networkEdgeList"
        ),
    }
  );
}

function bindEvents() {
  networkUi.refresh
    .addEventListener(
      "click",
      loadNetworkMap
    );

  networkUi.resetFilters
    .addEventListener(
      "click",
      resetFilters
    );

  networkUi.resetView
    .addEventListener(
      "click",
      resetView
    );

  networkUi.search
    .addEventListener(
      "input",
      () => {
        window.clearTimeout(
          networkState.searchTimer
        );

        networkState.searchTimer =
          window.setTimeout(
            loadNetworkMap,
            NETWORK_SEARCH_DELAY_MS
          );
      }
    );

  networkUi.search
    .addEventListener(
      "keydown",
      (event) => {
        if (
          event.key === "Enter"
        ) {
          event.preventDefault();

          window.clearTimeout(
            networkState.searchTimer
          );

          loadNetworkMap();
        }
      }
    );

  for (
    const control
    of [
      networkUi.decision,
      networkUi.addressType,
      networkUi.minimumCount,
      networkUi.maximumNodes,
      networkUi.maximumEdges,
      networkUi.includeDemo,
    ]
  ) {
    control.addEventListener(
      "change",
      loadNetworkMap
    );
  }

  networkUi.edgeList
    .addEventListener(
      "click",
      (event) => {
        const item =
          event.target.closest(
            "[data-network-edge-id]"
          );

        if (item) {
          selectEdgeFromList(
            item.dataset
              .networkEdgeId
          );
        }
      }
    );
}

document.addEventListener(
  "DOMContentLoaded",
  () => {
    initializeReferences();
    bindEvents();
    loadNetworkMap();
  }
);