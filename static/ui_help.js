"use strict";

(function initialiseHelpPage() {
  const searchInput = document.getElementById("helpSearchInput");
  const clearButton = document.getElementById("clearHelpSearch");
  const searchStatus = document.getElementById("helpSearchStatus");
  const emptyState = document.getElementById("helpEmptyState");
  const sections = Array.from(
    document.querySelectorAll(".help-section")
  );

  function normaliseSearchText(value) {
    return String(value ?? "")
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  }

  function searchableText(element) {
    if (!element) {
      return "";
    }

    return normaliseSearchText(
      [
        element.dataset.helpKeywords || "",
        element.textContent || "",
      ].join(" ")
    );
  }

  function sectionHeadingText(section) {
    const heading = section.querySelector(
      ":scope > .help-section-heading"
    );

    return normaliseSearchText(
      [
        section.dataset.helpKeywords || "",
        heading?.textContent || "",
      ].join(" ")
    );
  }

  function setItemVisible(item, visible) {
    item.hidden = !visible;
    item.setAttribute(
      "aria-hidden",
      visible ? "false" : "true"
    );
  }

  function restoreAllTopics() {
    sections.forEach((section) => {
      section.hidden = false;
      section.setAttribute(
        "aria-hidden",
        "false"
      );

      section
        .querySelectorAll("[data-help-item]")
        .forEach((item) => {
          setItemVisible(item, true);
        });
    });
  }

  function updateSearchStatus(
    query,
    visibleTopics,
    totalTopics,
    visibleSections
  ) {
    if (!searchStatus) {
      return;
    }

    if (!query) {
      searchStatus.textContent =
        `${totalTopics} help topics available.`;
      return;
    }

    if (visibleSections === 0) {
      searchStatus.textContent =
        "No matching topics.";
      return;
    }

    const topicWord =
      visibleTopics === 1
        ? "topic"
        : "topics";

    const sectionWord =
      visibleSections === 1
        ? "section"
        : "sections";

    searchStatus.textContent =
      `${visibleTopics} matching ${topicWord} ` +
      `in ${visibleSections} ${sectionWord}.`;
  }

  function applyHelpSearch() {
    const query = normaliseSearchText(
      searchInput?.value || ""
    );

    const allItems = Array.from(
      document.querySelectorAll(
        "[data-help-item]"
      )
    );

    clearButton?.toggleAttribute(
      "disabled",
      query.length === 0
    );

    if (!query) {
      restoreAllTopics();

      if (emptyState) {
        emptyState.hidden = true;
      }

      updateSearchStatus(
        "",
        allItems.length,
        allItems.length,
        sections.length
      );

      return;
    }

    let visibleSections = 0;
    let visibleTopics = 0;

    sections.forEach((section) => {
      const sectionMatches =
        sectionHeadingText(section).includes(
          query
        );

      const items = Array.from(
        section.querySelectorAll(
          "[data-help-item]"
        )
      );

      let sectionHasVisibleItem = false;

      items.forEach((item) => {
        const itemMatches =
          sectionMatches ||
          searchableText(item).includes(
            query
          );

        setItemVisible(
          item,
          itemMatches
        );

        if (itemMatches) {
          sectionHasVisibleItem = true;
          visibleTopics += 1;
        }
      });

      const shouldShowSection =
        sectionMatches ||
        sectionHasVisibleItem;

      section.hidden =
        !shouldShowSection;

      section.setAttribute(
        "aria-hidden",
        shouldShowSection
          ? "false"
          : "true"
      );

      if (shouldShowSection) {
        visibleSections += 1;
      }
    });

    if (emptyState) {
      emptyState.hidden =
        visibleSections !== 0;
    }

    updateSearchStatus(
      query,
      visibleTopics,
      allItems.length,
      visibleSections
    );
  }

  function clearHelpSearch(
    options = {}
  ) {
    if (!searchInput) {
      return;
    }

    searchInput.value = "";
    applyHelpSearch();

    if (options.focus !== false) {
      searchInput.focus();
    }
  }

  function formatConfigNumber(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return null;
    }

    return number.toLocaleString(
      undefined,
      {
        maximumFractionDigits: 8,
      }
    );
  }

  function setElementText(
    id,
    value
  ) {
    const element =
      document.getElementById(id);

    if (!element) {
      return;
    }

    element.textContent =
      String(value);
  }

  function showConfigurationFallback() {
    setElementText(
      "helpConfigStatus",
      "Current threshold values are unavailable. " +
        "The general explanation remains valid."
    );

    setElementText(
      "helpThresholdOk",
      "Current value unavailable"
    );

    setElementText(
      "helpThresholdWarn",
      "Current value unavailable"
    );

    setElementText(
      "helpThresholdMed",
      "Current value unavailable"
    );

    setElementText(
      "helpThresholdCrit",
      "Current value unavailable"
    );

    setElementText(
      "helpConfigSource",
      "Unavailable"
    );

    setElementText(
      "helpRepeatWindow",
      "Current value unavailable"
    );

    setElementText(
      "helpRepeatThreshold",
      "Current value unavailable"
    );
  }

  async function loadSafeRuntimeConfiguration() {
    try {
      const response = await fetch(
        "/runtime-config",
        {
          method: "GET",
          cache: "no-store",
          credentials: "same-origin",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(
          "Runtime configuration request " +
            `failed with ${response.status}`
        );
      }

      const payload =
        await response.json();

      const config =
        payload &&
        typeof payload.config ===
          "object"
          ? payload.config
          : null;

      const thresholds =
        config &&
        typeof config.thresholds ===
          "object"
          ? config.thresholds
          : null;

      const repeatLogic =
        config &&
        typeof config.repeat_logic ===
          "object"
          ? config.repeat_logic
          : null;

      const ok = formatConfigNumber(
        thresholds?.ok
      );

      const warn = formatConfigNumber(
        thresholds?.warn
      );

      const crit = formatConfigNumber(
        thresholds?.crit
      );

      const repeatWindow = Number(
        repeatLogic?.window_seconds
      );

      const repeatThreshold = Number(
        repeatLogic?.critical_repeat_count
      );

      if (
        ok === null ||
        warn === null ||
        crit === null
      ) {
        throw new Error(
          "Runtime configuration did not " +
            "contain valid threshold values"
        );
      }

      setElementText(
        "helpConfigStatus",
        "Validated values currently " +
          "loaded by the application:"
      );

      setElementText(
        "helpThresholdOk",
        `score < ${ok}`
      );

      setElementText(
        "helpThresholdWarn",
        `${ok} ≤ score < ${warn}`
      );

      setElementText(
        "helpThresholdMed",
        `${warn} ≤ score < ${crit}`
      );

      setElementText(
        "helpThresholdCrit",
        `score ≥ ${crit}`
      );

      setElementText(
        "helpConfigSource",
        String(
          payload.source ||
            "validated runtime configuration"
        )
      );

      setElementText(
        "helpRepeatWindow",
        Number.isFinite(repeatWindow)
          ? `${repeatWindow} second${
              repeatWindow === 1
                ? ""
                : "s"
            }`
          : "Current value unavailable"
      );

      setElementText(
        "helpRepeatThreshold",
        Number.isFinite(
          repeatThreshold
        )
          ? `${repeatThreshold} matching event${
              repeatThreshold === 1
                ? ""
                : "s"
            }`
          : "Current value unavailable"
      );
    } catch (error) {
      console.warn(
        "Help page could not load safe " +
          "runtime configuration:",
        error
      );

      showConfigurationFallback();
    }
  }

  function renderRuntimeMode(payload) {
    const modeElement =
      document.getElementById(
        "helpRuntimeMode"
      );

    if (!modeElement) {
      return;
    }

    const isReplay =
      String(
        payload?.mode || ""
      ).toLowerCase() === "replay";

    if (isReplay) {
      modeElement.textContent =
        "REPLAY MODE is active. " +
        "Previously saved processed events " +
        "may be inserted into the interface " +
        "for demonstration and validation.";
    } else {
      modeElement.textContent =
        "LIVE MODE is active. " +
        "The application is using its live " +
        "monitoring pipeline rather than " +
        "replay insertion.";
    }
  }

  async function loadHealthStatus() {
    const statusPill =
      document.getElementById(
        "statusPill"
      );

    if (!statusPill) {
      return;
    }

    try {
      const response = await fetch(
        "/health",
        {
          method: "GET",
          cache: "no-store",
          credentials: "same-origin",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(
          `Health request failed with ${response.status}`
        );
      }

      const payload =
        await response.json();

      statusPill.textContent =
        String(
          payload?.status || "ok"
        ).toUpperCase();

      statusPill.classList.remove(
        "critical"
      );

      statusPill.classList.add(
        "ok"
      );
    } catch (error) {
      console.warn(
        "Help page could not load " +
          "system health:",
        error
      );

      statusPill.textContent =
        "UNKNOWN";

      statusPill.classList.remove(
        "ok"
      );

      statusPill.classList.add(
        "critical"
      );
    }
  }

  async function loadRuntimeModeForHelp() {
    try {
      const response = await fetch(
        "/runtime/mode",
        {
          method: "GET",
          cache: "no-store",
          credentials: "same-origin",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(
          "Runtime mode request failed " +
            `with ${response.status}`
        );
      }

      const payload =
        await response.json();

      renderRuntimeMode(payload);
    } catch (error) {
      console.warn(
        "Help page could not load " +
          "runtime mode:",
        error
      );

      setElementText(
        "helpRuntimeMode",
        "Current operating mode is " +
          "unavailable. The replay " +
          "explanation above remains valid."
      );
    }
  }

  function focusHashTarget() {
    const rawHash =
      window.location.hash.replace(
        /^#/,
        ""
      );

    if (!rawHash) {
      return;
    }

    let decodedHash = rawHash;

    try {
      decodedHash =
        decodeURIComponent(rawHash);
    } catch {
      decodedHash = rawHash;
    }

    const target =
      document.getElementById(
        decodedHash
      );

    if (!target) {
      return;
    }

    window.requestAnimationFrame(
      () => {
        target.scrollIntoView({
          block: "start",
        });
      }
    );
  }

  searchInput?.addEventListener(
    "input",
    applyHelpSearch
  );

  searchInput?.addEventListener(
    "keydown",
    (event) => {
      if (
        event.key === "Escape" &&
        searchInput.value
      ) {
        event.preventDefault();
        clearHelpSearch();
      }
    }
  );

  clearButton?.addEventListener(
    "click",
    () => {
      clearHelpSearch();
    }
  );

  window.addEventListener(
    "hashchange",
    focusHashTarget
  );

  window.addEventListener(
    "ids-runtime-mode-loaded",
    (event) => {
      renderRuntimeMode(
        event.detail
      );
    }
  );

  applyHelpSearch();
  loadSafeRuntimeConfiguration();
  loadHealthStatus();
  loadRuntimeModeForHelp();
  focusHashTarget();
})();