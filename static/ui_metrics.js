"use strict";

const METRICS_REFRESH_INTERVAL_MS = 5000;

let metricsRefreshTimer = null;
let metricsRefreshInProgress = false;


document.addEventListener("DOMContentLoaded", () => {
    const refreshButton = document.getElementById("metricsRefreshButton");

    if (refreshButton) {
        refreshButton.addEventListener("click", () => {
            refreshMetricsPage();
        });
    }

    refreshMetricsPage();

    metricsRefreshTimer = window.setInterval(
        refreshMetricsPage,
        METRICS_REFRESH_INTERVAL_MS
    );

    window.addEventListener("beforeunload", () => {
        if (metricsRefreshTimer !== null) {
            window.clearInterval(metricsRefreshTimer);
        }
    });
});


function getMetricsTextColor() {
    const bodyStyle = window.getComputedStyle(document.body);
    return bodyStyle.color || "#1f2937";
}


function getBasePlotLayout() {
    return {
        autosize: true,
        margin: {
            l: 55,
            r: 20,
            t: 15,
            b: 55
        },
        paper_bgcolor: "rgba(0, 0, 0, 0)",
        plot_bgcolor: "rgba(0, 0, 0, 0)",
        font: {
            color: getMetricsTextColor(),
            family: "inherit"
        },
        hoverlabel: {
            namelength: -1
        }
    };
}


function getPlotConfiguration() {
    return {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    };
}


function setRefreshStatus(message, isError = false) {
    const element = document.getElementById("metricsRefreshStatus");

    if (!element) {
        return;
    }

    element.textContent = message;
    element.classList.toggle("metrics-refresh-error", isError);
}


function setSummaryValue(elementId, value) {
    const element = document.getElementById(elementId);

    if (element) {
        element.textContent = value;
    }
}


function formatSourceLabel(source) {
    const value = String(source || "none");

    if (value === "memory:recent") {
        return "Current traffic buffer";
    }

    if (value === "memory:alerts") {
        return "Current alert buffer";
    }

    if (value.startsWith("jsonl:")) {
        return value.slice("jsonl:".length);
    }

    return "No data";
}


function formatRefreshTime(timestamp) {
    const numericTimestamp = Number(timestamp);

    if (!Number.isFinite(numericTimestamp)) {
        return new Date().toLocaleTimeString();
    }

    return new Date(numericTimestamp * 1000).toLocaleTimeString();
}


function hasRows(rows) {
    return (
        Array.isArray(rows) &&
        rows.some((row) => Number(row?.count) > 0)
    );
}


function showChartEmptyState(chartId, emptyId, message = null) {
    const chartElement = document.getElementById(chartId);
    const emptyElement = document.getElementById(emptyId);

    if (window.Plotly && chartElement) {
        try {
            window.Plotly.purge(chartElement);
        } catch (error) {
            console.debug("Plotly purge skipped:", error);
        }
    }

    if (chartElement) {
        chartElement.hidden = true;
    }

    if (emptyElement) {
        if (message) {
            emptyElement.textContent = message;
        }

        emptyElement.hidden = false;
    }
}


function showChart(chartId, emptyId) {
    const chartElement = document.getElementById(chartId);
    const emptyElement = document.getElementById(emptyId);

    if (chartElement) {
        chartElement.hidden = false;
    }

    if (emptyElement) {
        emptyElement.hidden = true;
    }
}


function renderVerticalBarChart(chartId, emptyId, rows, xAxisTitle) {
    if (!hasRows(rows)) {
        showChartEmptyState(chartId, emptyId);
        return;
    }

    showChart(chartId, emptyId);

    const labels = rows.map((row) => String(row.label));
    const counts = rows.map((row) => Number(row.count));

    const layout = {
        ...getBasePlotLayout(),
        xaxis: {
            title: xAxisTitle,
            automargin: true,
            fixedrange: true
        },
        yaxis: {
            title: "Flows",
            rangemode: "tozero",
            automargin: true,
            fixedrange: true
        }
    };

    const trace = {
        type: "bar",
        x: labels,
        y: counts,
        text: counts.map(String),
        textposition: "auto",
        hovertemplate: "%{x}: %{y}<extra></extra>"
    };

    window.Plotly.react(
        chartId,
        [trace],
        layout,
        getPlotConfiguration()
    );
}


function renderHorizontalBarChart(chartId, emptyId, rows, xAxisTitle) {
    if (!hasRows(rows)) {
        showChartEmptyState(chartId, emptyId);
        return;
    }

    showChart(chartId, emptyId);

    const orderedRows = [...rows].reverse();

    const labels = orderedRows.map((row) => String(row.label));
    const counts = orderedRows.map((row) => Number(row.count));

    const layout = {
        ...getBasePlotLayout(),
        margin: {
            l: 110,
            r: 25,
            t: 15,
            b: 50
        },
        xaxis: {
            title: xAxisTitle,
            rangemode: "tozero",
            automargin: true,
            fixedrange: true
        },
        yaxis: {
            type: "category",
            castegoryorder: "array",
            categoryarray: labels,
            automargin: true,
            fixedrange: true
        }
    };

    const trace = {
        type: "bar",
        orientation: "h",
        x: counts,
        y: labels,
        text: counts.map(String),
        textposition: "auto",
        hovertemplate: "%{y}: %{x}<extra></extra>"
    };

    window.Plotly.react(
        chartId,
        [trace],
        layout,
        getPlotConfiguration()
    );
}


function renderScoreChart(points) {
    const chartId = "metricsScoreChart";
    const emptyId = "metricsScoreEmpty";

    if (!Array.isArray(points) || points.length === 0) {
        showChartEmptyState(chartId, emptyId);
        return;
    }

    const validPoints = points.filter((point) => {
        return (
            Number.isFinite(Number(point?.ts_unix)) &&
            Number.isFinite(Number(point?.average))
        );
    });

    if (validPoints.length === 0) {
        showChartEmptyState(chartId, emptyId);
        return;
    }

    showChart(chartId, emptyId);

    const timestamps = validPoints.map((point) => {
        return new Date(Number(point.ts_unix) * 1000);
    });

    const averages = validPoints.map((point) => {
        return Number(point.average);
    });

    const sampleCounts = validPoints.map((point) => {
        return Number(point.count || 0);
    });

    const trace = {
        type: "scatter",
        mode: "lines+markers",
        x: timestamps,
        y: averages,
        customdata: sampleCounts,
        hovertemplate:
            "%{x}<br>" +
            "Average score: %{y:.6f}<br>" +
            "Samples: %{customdata}" +
            "<extra></extra>"
    };

    const layout = {
        ...getBasePlotLayout(),
        margin: {
            l: 70,
            r: 25,
            t: 15,
            b: 60
        },
        xaxis: {
            title: "Time",
            automargin: true,
            fixedrange: true
        },
        yaxis: {
            title: "Average anomaly score",
            rangemode: "tozero",
            automargin: true,
            fixedrange: true
        }
    };

    window.Plotly.react(
        chartId,
        [trace],
        layout,
        getPlotConfiguration()
    );
}


function renderMetrics(data) {
    setSummaryValue(
        "metricsTotalFlows",
        Number(data?.total_flows || 0).toLocaleString()
    );

    setSummaryValue(
        "metricsDataSource",
        formatSourceLabel(data?.source)
    );

    setSummaryValue(
        "metricsLastRefresh",
        formatRefreshTime(data?.generated_at)
    );

    renderVerticalBarChart(
        "metricsSeverityChart",
        "metricsSeverityEmpty",
        data?.severity_distribution,
        "Final decision"
    );

    renderHorizontalBarChart(
        "metricsTrafficClassChart",
        "metricsTrafficClassEmpty",
        data?.traffic_class_distribution,
        "Flows"
    );

    renderScoreChart(
        data?.average_anomaly_score_over_time
    );

    renderHorizontalBarChart(
        "metricsProtocolsChart",
        "metricsProtocolsEmpty",
        data?.top_protocols,
        "Flows"
    );

    renderHorizontalBarChart(
        "metricsPortsChart",
        "metricsPortsEmpty",
        data?.top_destination_ports,
        "Flows"
    );
}


function showMetricsLoadError(message) {
    const emptyMessage = message || "Unable to load metrics data.";

    showChartEmptyState(
        "metricsSeverityChart",
        "metricsSeverityEmpty",
        emptyMessage
    );

    showChartEmptyState(
        "metricsTrafficClassChart",
        "metricsTrafficClassEmpty",
        emptyMessage
    );

    showChartEmptyState(
        "metricsScoreChart",
        "metricsScoreEmpty",
        emptyMessage
    );

    showChartEmptyState(
        "metricsProtocolsChart",
        "metricsProtocolsEmpty",
        emptyMessage
    );

    showChartEmptyState(
        "metricsPortsChart",
        "metricsPortsEmpty",
        emptyMessage
    );
}


async function refreshMetricsPage() {
    if (metricsRefreshInProgress) {
        return;
    }

    metricsRefreshInProgress = true;
    setRefreshStatus("Refreshing metrics...");

    try {
        if (!window.Plotly) {
            throw new Error("Plotly could not be loaded.");
        }

        const response = await fetch(
            "/metrics/ui-summary",
            {
                method: "GET",
                cache: "no-store",
                headers: {
                    "Accept": "application/json"
                }
            }
        );

        if (response.status === 401) {
            window.location.href = "/login";
            return;
        }

        if (!response.ok) {
            throw new Error(
                `Metrics request failed with status ${response.status}.`
            );
        }

        const data = await response.json();

        renderMetrics(data);

        if (Number(data?.total_flows || 0) === 0) {
            setRefreshStatus(
                "No flow metrics are currently available. Refreshing every 5 seconds."
            );
        } else {
            setRefreshStatus(
                "Metrics updated. Automatic refresh runs every 5 seconds."
            );
        }
    } catch (error) {
        console.error("Metrics refresh failed:", error);

        setRefreshStatus(
            "Unable to refresh metrics.",
            true
        );

        showMetricsLoadError(
            "Metrics could not be loaded. The page will retry automatically."
        );
    } finally {
        metricsRefreshInProgress = false;
    }
}