/*
 * DrCode Desktop — terminal client (multi-tab).
 *
 * One window, one tab per project: each tab owns an xterm.js instance
 * and its own WebSocket to the local aiohttp server's /ws bridge (the
 * tab id rides along as a query param), i.e. its own drcode process in
 * its own PTY:
 *   - binary frames: raw PTY output (server->client) / stdin (client->server)
 *   - text frames:   JSON control events (start/resize/exited/...)
 *
 * Tab switching hides the inactive pane and re-fits the active terminal
 * (fit() + a resize control message to its PTY). Everything not tied to
 * a single project is window-global: the OSC 52 clipboard filter (per
 * stream, but the behavior is shared), the font settings (applied to
 * every live terminal), the settings dialog, fullscreen, recents.
 *
 * OSC 52 (the clipboard escape drcode uses) is intercepted in the output
 * byte stream BEFORE xterm.js sees it and routed to the system clipboard:
 * first navigator.clipboard, then the pywebview bridge as fallback
 * (WKWebView often rejects clipboard writes without a user gesture).
 */
(function () {
  "use strict";

  var params = new URLSearchParams(location.search);
  var TOKEN = params.get("token") || "";

  var elTabbar = document.getElementById("tabbar");
  var elTabs = document.getElementById("tabs");
  var elNewTab = document.getElementById("new-tab-btn");
  var elPanes = document.getElementById("panes");
  var elWelcome = document.getElementById("welcome");
  var elPick = document.getElementById("pick-project");
  var elRecentsBlock = document.getElementById("recents-block");
  var elRecentsList = document.getElementById("recents-list");
  var elWelcomeError = document.getElementById("welcome-error");
  var elExitDialog = document.getElementById("exit-dialog");
  var elExitText = document.getElementById("exit-text");
  var elBootError = document.getElementById("boot-error");
  var elSettingsDialog = document.getElementById("settings-dialog");
  var elSettingsFont = document.getElementById("settings-font-family");
  var elSettingsCustom = document.getElementById("settings-font-custom");
  var elSettingsSize = document.getElementById("settings-font-size");
  var elSettingsSizeValue = document.getElementById("settings-size-value");
  var elSettingsError = document.getElementById("settings-error");
  var elSettingsPreview = document.getElementById("settings-font-preview");
  var elOpenSettings = document.getElementById("open-settings");
  var elFontBtn = document.getElementById("font-settings-btn");

  var FONT_STACK = '"JetBrainsMono Nerd Font Mono", "JetBrains Mono", Menlo, monospace';

  /* Curated picker list of common monospaced families across the three
   * platforms. Any family is used as the FIRST entry of the fallback
   * chain, with FONT_STACK trailing — an unknown/typo'd name degrades to
   * the bundled font instead of blank glyphs. Empty value = default
   * (FONT_STACK alone).
   *
   * The webview cannot enumerate system fonts itself, so the list is
   * broader than any single machine: get_available_fonts (bridge) names
   * the families the OS really has and populateFontSelect hides the
   * rest — an entry that survives filtering is guaranteed renderable,
   * so every visible item looks (and previews) differently. */
  var FONT_CHOICES = [
    { label: "Default (bundled Nerd Font)", value: "" },
    { label: "JetBrainsMono Nerd Font Mono", value: '"JetBrainsMono Nerd Font Mono"' },
    { label: "JetBrains Mono", value: '"JetBrains Mono"' },
    { label: "Menlo", value: "Menlo" },
    { label: "Monaco", value: "Monaco" },
    { label: "SF Mono", value: '"SF Mono"' },
    { label: "Courier New", value: '"Courier New"' },
    { label: "Consolas", value: "Consolas" },
    { label: "Cascadia Code", value: '"Cascadia Code"' },
    { label: "Fira Code", value: '"Fira Code"' },
    { label: "Source Code Pro", value: '"Source Code Pro"' },
    { label: "Hack", value: "Hack" },
    { label: "Inconsolata", value: "Inconsolata" },
    { label: "DejaVu Sans Mono", value: '"DejaVu Sans Mono"' },
    { label: "Ubuntu Mono", value: '"Ubuntu Mono"' },
    { label: "Liberation Mono", value: '"Liberation Mono"' },
  ];
  var FONT_CUSTOM_VALUE = "__custom__";

  var DEFAULT_FONT_SIZE = 13;
  var MIN_FONT_SIZE = 6;
  var MAX_FONT_SIZE = 40;
  var FONT_STEP = 1;

  /* ---- tab model ---- */
  /* tabs[] holds one object per tab, in tab-bar order:
   *   { id, kind: "welcome"|"terminal", project, started, closing,
   *     ws, term, fit, webgl, filter, pane, elTerm, tabEl, labelEl }
   * The welcome node (#welcome) is a SINGLE shared element moved into
   * whichever welcome-tab is active — only one tab is visible at a
   * time, so reparenting is enough. */
  var tabs = [];
  var activeTab = null;
  var nextTabSeq = 1;
  var dialogTab = null; // tab the crash dialog is about

  var savedSettings = null; // {font_family, font_size} from the bridge
  var currentFontFamily = null; // primary family (null = default FONT_STACK)
  /* Families the OS reports as installed (bridge get_available_fonts).
   * null = unknown (browser mode or probe failure) → NO filtering, the
   * full curated list is shown — fail-open, exactly the pre-filtering
   * behavior. An empty array (probe worked, nothing matched) does
   * filter: only Default and Custom… remain. */
  var availableFonts = null;

  /* Last recents payload from the bridge, kept so the list can be
   * re-rendered (with fresh occupied/open marks) whenever a welcome tab
   * is activated — tabs may have opened or closed since the last look. */
  var lastRecents = null;

  function api(name) {
    var bridge = window.pywebview && window.pywebview.api;
    if (!bridge || typeof bridge[name] !== "function") return null;
    return function () {
      return bridge[name].apply(bridge, arguments);
    };
  }

  function bootError(message) {
    elBootError.textContent = message;
    elBootError.classList.remove("hidden");
  }

  /* ---------------- OSC 52 interception ----------------
   * Byte-level state machine over the PTY output stream. Multibyte UTF-8
   * sequences never contain ASCII bytes, so scanning for ESC ] 5 2 ... is
   * safe and everything else is forwarded untouched (xterm.js keeps its
   * own UTF-8 state across chunks when fed Uint8Array).
   */
  var OSC52_MAX = 16 * 1024 * 1024; // sanity cap on the base64 payload

  function makeOsc52Filter(forward, copy) {
    var state = 0; // 0 normal, 1 saw ESC, 2 in OSC, 3 in OSC, saw ESC
    var osc = [];
    var ESC = 0x1b, BEL = 0x07, BRACKET = 0x5d, BACKSLASH = 0x5c;

    function finishOsc() {
      var body = String.fromCharCode.apply(null, osc);
      osc = [];
      var parts = body.split(";");
      if (parts[0] !== "52" || parts.length < 2) return;
      var b64 = parts.length >= 3 ? parts.slice(2).join(";") : parts[1];
      if (!b64) return; // empty payload = OSC 52 "clear/query" — ignore
      if (b64.length > OSC52_MAX) return;
      try {
        var bin = atob(b64);
        var bytes = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        var text = new TextDecoder("utf-8").decode(bytes);
        if (text) copy(text);
      } catch (err) {
        /* malformed OSC 52 — drop silently */
      }
    }

    return function process(bytes) {
      var out = [];
      var i = 0;
      while (i < bytes.length) {
        var b = bytes[i++];
        if (state === 0) {
          if (b === ESC) state = 1;
          else out.push(b);
        } else if (state === 1) {
          if (b === BRACKET) { state = 2; osc = []; }
          else { out.push(ESC, b); state = 0; }
        } else if (state === 2) {
          if (b === BEL) { finishOsc(); state = 0; }
          else if (b === ESC) {
            if (i < bytes.length && bytes[i] === BACKSLASH) { i++; finishOsc(); state = 0; }
            else if (i >= bytes.length) state = 3; // ST split across chunks
            else osc.push(ESC, b);
          } else {
            osc.push(b);
            if (osc.length > OSC52_MAX) { state = 0; osc = []; } // runaway OSC — bail
          }
        } else if (state === 3) {
          if (b === BACKSLASH) { finishOsc(); state = 0; }
          else { osc.push(ESC); state = b === ESC ? 3 : 2; if (state === 2) osc.push(b); }
        }
      }
      if (out.length) forward(new Uint8Array(out));
    };
  }

  function copyToClipboard(text) {
    var bridgeCopy = api("set_clipboard");
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(function () {
        if (bridgeCopy) bridgeCopy(text);
      });
    } else if (bridgeCopy) {
      bridgeCopy(text);
    }
  }

  /* ---------------- tabs ---------------- */

  function fontStackFor(family) {
    var primary = family && String(family).trim();
    return primary ? primary + ", " + FONT_STACK : FONT_STACK;
  }

  function projectTitle(path) {
    var parts = String(path || "").split("/").filter(Boolean);
    return parts.pop() || path || "DrCode";
  }

  function nextTabId() {
    return "t" + nextTabSeq++;
  }

  function createTab(kind, project) {
    var tab = {
      id: nextTabId(),
      kind: kind, // "welcome" (picker, no process yet) or "terminal"
      project: project || null,
      started: false,
      closing: false, // deliberate close in progress: suppress dialogs
      ws: null,
      term: null,
      fit: null,
      webgl: null,
      filter: null,
      pane: null,
      elTerm: null,
      tabEl: null,
      labelEl: null,
    };
    tabs.push(tab);

    var tabEl = document.createElement("li");
    tabEl.className = "tab";
    var label = document.createElement("span");
    label.className = "tab-label";
    tabEl.appendChild(label);
    var closeBtn = document.createElement("span");
    closeBtn.className = "tab-close";
    closeBtn.textContent = "×";
    closeBtn.title = "Close tab";
    tabEl.appendChild(closeBtn);
    closeBtn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      closeTab(tab);
    });
    tabEl.addEventListener("click", function () { activateTab(tab); });
    tab.tabEl = tabEl;
    tab.labelEl = label;
    elTabs.appendChild(tabEl);

    var pane = document.createElement("div");
    pane.className = "tab-pane hidden";
    tab.pane = pane;
    elPanes.appendChild(pane);

    if (kind === "terminal") createTerminalHost(tab);
    refreshTabLabels();
    activateTab(tab);
    return tab;
  }

  function createTerminalHost(tab) {
    // The xterm container: hidden until the session's "started" event
    // (the shared welcome node keeps the pane occupied until then).
    var elTerm = document.createElement("div");
    elTerm.className = "terminal hidden";
    tab.pane.appendChild(elTerm);
    tab.elTerm = elTerm;
    if (resizeObserver) resizeObserver.observe(elTerm);
  }

  function refreshTabLabels() {
    tabs.forEach(function (tab) {
      if (tab.kind === "welcome") {
        tab.labelEl.textContent = "New project";
        tab.tabEl.title = "New project";
      } else {
        tab.labelEl.textContent = projectTitle(tab.project);
        tab.tabEl.title = tab.project || "";
      }
    });
  }

  function activateTab(tab) {
    if (!tab) return;
    activeTab = tab;
    tabs.forEach(function (t) {
      t.pane.classList.toggle("hidden", t !== tab);
      t.tabEl.classList.toggle("active", t === tab);
    });
    if (tab.kind === "welcome") {
      // Move the shared welcome node into this pane (appendChild on an
      // already-attached node relocates it).
      tab.pane.appendChild(elWelcome);
      elWelcome.classList.remove("hidden");
      // Re-render the recents with current occupancy: tabs opened or
      // closed since the last render change which projects are open.
      if (lastRecents) renderRecents(lastRecents);
      updateFontBtnVisibility();
      setWindowTitle("DrCode");
      return;
    }
    if (tab.started) {
      if (tab.fit) {
        try { tab.fit.fit(); } catch (err) { /* container not measurable */ }
      }
      sendResize(tab);
      if (tab.term) tab.term.focus();
    }
    if (tab.project) setWindowTitle("DrCode — " + projectTitle(tab.project));
    updateFontBtnVisibility();
  }

  function updateFontBtnVisibility() {
    if (!elFontBtn) return;
    var show = !!(activeTab && activeTab.started);
    elFontBtn.classList.toggle("hidden", !show);
  }

  function setWindowTitle(title) {
    document.title = title;
    var setTitle = api("set_title");
    if (setTitle) setTitle(title);
  }

  function closeTab(tab) {
    if (!tab || tab.closing) return;
    tab.closing = true;
    // Kill the PTY explicitly: tab sessions deliberately outlive a ws
    // reconnect, so the server will not infer the close from the socket.
    if (tab.ws && tab.ws.readyState === WebSocket.OPEN) {
      try { tab.ws.send(JSON.stringify({ type: "kill", tab: tab.id })); } catch (err) { /* gone */ }
    }
    if (tab.ws) {
      try { tab.ws.close(); } catch (err) { /* gone */ }
    }
    if (dialogTab === tab) {
      dialogTab = null;
      hideDialog();
    }
    if (tab.term) {
      try { tab.term.dispose(); } catch (err) { /* already gone */ }
    }
    if (elWelcome.parentNode === tab.pane) elWelcome.remove();
    if (tab.pane) tab.pane.remove();
    if (tab.tabEl) tab.tabEl.remove();
    var idx = tabs.indexOf(tab);
    if (idx >= 0) tabs.splice(idx, 1);
    if (activeTab === tab) {
      activeTab = null;
      if (tabs.length) activateTab(tabs[Math.max(0, idx - 1)]);
    }
    if (!tabs.length) {
      // Last tab closed — the window has nothing left to show.
      var closeWindow = api("close_window");
      if (closeWindow) closeWindow();
      else window.close();
    }
  }

  /* Python-side hook (ctx.on_clean_exit → evaluate_js) closes the tab
   * through this; idempotent with the ws "exited"-event path. */
  window.__drcodeDesktop = {
    closeTab: function (id) {
      var tab = tabs.filter(function (t) { return t.id === id; })[0];
      if (tab) closeTab(tab);
    },
  };

  /* ---------------- terminal setup (per tab) ---------------- */

  function createTerminalFor(tab) {
    var settings = savedSettings || {};
    var fontSize = typeof settings.font_size === "number"
      ? settings.font_size
      : DEFAULT_FONT_SIZE;
    tab.term = new window.Terminal({
      fontFamily: fontStackFor(currentFontFamily),
      fontSize: fontSize,
      lineHeight: 1.0,
      cursorBlink: true,
      allowProposedApi: true,
      scrollback: 0, // alternate screen app: Textual scrolls its own content
      convertEol: false,
      macOptionIsMeta: true,
      theme: {
        background: "#16181d",
        foreground: "#d7dae0",
        cursor: "#4f9cf9",
        selectionBackground: "#2d4a6e",
      },
    });
    // Unicode 11 width tables: the built-in UnicodeV6 provider predates emoji,
    // so ✅ etc. render 1 cell wide while Textual (Python wcwidth) sends them
    // as 2 — the rest of the line then drifts one cell to the left.
    tab.term.loadAddon(new window.Unicode11Addon.Unicode11Addon());
    tab.term.unicode.activeVersion = "11";
    tab.fit = new window.FitAddon.FitAddon();
    tab.term.loadAddon(tab.fit);
    tab.term.open(tab.elTerm);
    try {
      tab.webgl = new window.WebglAddon.WebglAddon();
      tab.webgl.onContextLoss(function () {
        if (tab.webgl) { tab.webgl.dispose(); tab.webgl = null; }
      }); // xterm falls back to DOM renderer
      tab.term.loadAddon(tab.webgl);
    } catch (err) {
      tab.webgl = null;
      /* no WebGL — the default renderer keeps everything working */
    }
    tab.term.attachCustomKeyEventHandler(function (ev) {
      return terminalKeyHandler(tab, ev);
    });
    tab.term.onData(function (data) {
      if (tab.ws && tab.ws.readyState === WebSocket.OPEN) {
        tab.ws.send(new TextEncoder().encode(data));
      }
    });
  }

  /* ---------------- font settings (Ctrl+= / Ctrl+- / Ctrl+0 / Ctrl+,) -------
   * Without interception these combos would be forwarded to the PTY
   * through term.onData; the custom handler swallows them (return false)
   * and changes term.options instead. The WebGL renderer in xterm 5.x can
   * keep a stale glyph atlas across any font-metric change, so the addon
   * is re-created after each change; on failure the DOM renderer takes
   * over. fit.fit() recomputes cols/rows for the new cell metrics and
   * sendResize() propagates them to the PTY.
   *
   * Font settings are WINDOW-global (remembered for all projects), so a
   * change fans out to every live terminal, not just the active one.
   */

  function recreateWebgl(tab) {
    if (!tab.webgl) return;
    try { tab.webgl.dispose(); } catch (err) { /* already gone */ }
    tab.webgl = null;
    try {
      tab.webgl = new window.WebglAddon.WebglAddon();
      tab.webgl.onContextLoss(function () {
        if (tab.webgl) { tab.webgl.dispose(); tab.webgl = null; }
      });
      tab.term.loadAddon(tab.webgl);
    } catch (err) {
      tab.webgl = null; /* DOM renderer fallback */
    }
  }

  function refitAfterFontChange(tab) {
    if (tab.fit) {
      try { tab.fit.fit(); } catch (err) { /* container not measurable */ }
    }
    sendResize(tab);
  }

  /* Hotkey-driven size changes must persist like the settings dialog does.
   * A held Ctrl+-/Ctrl+= emits a burst of keydowns, so the bridge call is
   * debounced; at fire time the CURRENT term.options.fontSize is sent and
   * the family is always passed along — the python side rewrites BOTH
   * fields, and null family would wipe the stored font. Bridge-less
   * browser mode stays local (no persistence, matching its no-disk
   * nature). */
  var FONT_SAVE_DEBOUNCE_MS = 500;
  var fontSizeSaveTimer = null;

  function persistFontSize() {
    var ref = activeTab && activeTab.term;
    if (!ref) return;
    savedSettings = { font_family: currentFontFamily, font_size: ref.options.fontSize };
    var setFont = api("set_font_settings");
    if (!setFont) return; // browser mode: apply locally only
    if (fontSizeSaveTimer) clearTimeout(fontSizeSaveTimer);
    fontSizeSaveTimer = setTimeout(function () {
      fontSizeSaveTimer = null;
      Promise.resolve(setFont(currentFontFamily, ref.options.fontSize))
        .then(function (res) {
          if (res) savedSettings = res;
        })
        .catch(function () { /* best-effort persistence, like _save_config */ });
    }, FONT_SAVE_DEBOUNCE_MS);
  }

  function applyFontSize(size) {
    var clamped = Math.min(MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE, size));
    var changed = false;
    tabs.forEach(function (tab) {
      if (!tab.term || tab.term.options.fontSize === clamped) return;
      tab.term.options.fontSize = clamped;
      recreateWebgl(tab);
      refitAfterFontChange(tab);
      changed = true;
    });
    if (changed) persistFontSize();
  }

  /* Changing the family changes the cell metrics exactly like a size
   * change (glyph widths / advance), so the same WebGL re-creation +
   * refit sequence applies. The family rides as the first entry of the
   * fallback chain (fontStackFor) so unknown names degrade gracefully. */
  function applyFontFamily(family) {
    currentFontFamily = family || null;
    var stack = fontStackFor(family);
    tabs.forEach(function (tab) {
      if (!tab.term || tab.term.options.fontFamily === stack) return;
      tab.term.options.fontFamily = stack;
      recreateWebgl(tab);
      refitAfterFontChange(tab);
    });
  }

  /* Modified Enter variants (Shift+Enter etc.) must reach the PTY as a
   * kitty-style CSI u sequence: xterm.js encodes every Enter as plain "\r",
   * so the TUI would only ever see "enter" and submit instead of inserting
   * a newline. Textual's input parser decodes \x1b[13;<mods>u back into
   * shift+enter / ctrl+enter / alt+enter (kitty modifier encoding:
   * param = 1 + shift(1) + alt(2) + ctrl(4)). Plain Enter stays untouched.
   * Only intercepted while the tab's session is live — the welcome screen
   * has no PTY to talk to.
   */
  function encodeEnter(event) {
    var mods = 1;
    if (event.shiftKey) mods += 1;
    if (event.altKey) mods += 2;
    if (event.ctrlKey) mods += 4;
    return "\x1b[13;" + mods + "u";
  }

  function terminalKeyHandler(tab, ev) {
    // xterm.js invokes the custom handler for keydown, keyup AND keypress.
    // Reacting to keyup would re-fire every interception (a second CSI u
    // for modified Enter → a duplicated newline, a second font-size step),
    // so only keydown acts; keyup/keypress fall through to xterm's own
    // (no-op for our cases) handling.
    if (ev.type !== "keydown") return true;
    if (ev.key === "F11") return true; // handled by the window listener
    if (ev.key === "Enter" && (ev.shiftKey || ev.ctrlKey || ev.altKey)) {
      if (!tab.ws || tab.ws.readyState !== WebSocket.OPEN || !tab.started) return true;
      ev.preventDefault();
      tab.ws.send(new TextEncoder().encode(encodeEnter(ev)));
      return false;
    }
    if (!ev.ctrlKey || ev.metaKey || ev.altKey || ev.shiftKey) return true;
    var code = ev.code || "";
    if (ev.key === ",") {
      ev.preventDefault();
      openSettings();
      return false;
    }
    if (ev.key === "=" || ev.key === "+" || code === "NumpadAdd") {
      ev.preventDefault();
      applyFontSize(((activeTab && activeTab.term ? activeTab.term.options.fontSize : 0) || DEFAULT_FONT_SIZE) + FONT_STEP);
      return false;
    }
    if (ev.key === "-" || ev.key === "_" || code === "NumpadSubtract") {
      ev.preventDefault();
      applyFontSize(((activeTab && activeTab.term ? activeTab.term.options.fontSize : 0) || DEFAULT_FONT_SIZE) - FONT_STEP);
      return false;
    }
    if (ev.key === "0" || code === "Digit0" || code === "Numpad0") {
      ev.preventDefault();
      applyFontSize(DEFAULT_FONT_SIZE);
      return false;
    }
    return true;
  }

  function currentSize(tab) {
    var size = { cols: 80, rows: 24 };
    if (tab && tab.fit) {
      try { tab.fit.fit(); } catch (err) { /* container not measurable yet */ }
      var proposal = tab.fit.proposeDimensions();
      if (proposal && proposal.cols && proposal.rows) {
        size.cols = proposal.cols;
        size.rows = proposal.rows;
      }
    }
    return size;
  }

  function sendResize(tab) {
    tab = tab || activeTab;
    if (!tab || !tab.ws || tab.ws.readyState !== WebSocket.OPEN || !tab.started) return;
    var size = currentSize(tab);
    tab.ws.send(JSON.stringify({ type: "resize", cols: size.cols, rows: size.rows, tab: tab.id }));
  }

  /* ---------------- font settings dialog ---------------- */

  /* FONT_CHOICES values carry CSS quoting ("SF Mono"); the availability
   * list from the bridge holds bare family names. Matching is case-
   * insensitive and prefix-tolerant ("Consolas" vs the registry's
   * "Consolas Bold (TrueType)" → "Consolas Bold"). */
  function unquoteFamily(value) {
    return String(value || "").replace(/["']/g, "").trim();
  }

  function fontIsAvailable(value) {
    if (availableFonts === null) return true; // unknown → fail open
    var want = unquoteFamily(value).toLowerCase();
    if (!want) return true; // Default (empty) — always offered
    return availableFonts.some(function (name) {
      var n = String(name).toLowerCase();
      return n === want || n.indexOf(want + " ") === 0;
    });
  }

  /* Only the <select> is filtered — the apply path (fontStackFor: chosen
   * family first, FONT_STACK trailing) is untouched, so a family that is
   * missing on THIS machine still works when the project moves to one
   * where it exists. A saved font that is no longer offered falls into
   * the Custom… branch below: shown selected, never silently reset. */
  function populateFontSelect(selected) {
    elSettingsFont.innerHTML = "";
    var choices = FONT_CHOICES.filter(function (choice) {
      return fontIsAvailable(choice.value);
    });
    choices.forEach(function (choice) {
      var opt = document.createElement("option");
      opt.value = choice.value;
      opt.textContent = choice.label;
      elSettingsFont.appendChild(opt);
    });
    var custom = document.createElement("option");
    custom.value = FONT_CUSTOM_VALUE;
    custom.textContent = "Custom…";
    elSettingsFont.appendChild(custom);

    var known = choices.some(function (c) { return c.value === selected; });
    if (known) {
      elSettingsFont.value = selected;
      elSettingsCustom.value = "";
      elSettingsCustom.classList.add("hidden");
    } else {
      elSettingsFont.value = FONT_CUSTOM_VALUE;
      elSettingsCustom.value = selected || "";
      elSettingsCustom.classList.remove("hidden");
    }
  }

  function syncSizeLabel() {
    elSettingsSizeValue.textContent = elSettingsSize.value || String(DEFAULT_FONT_SIZE);
  }

  /* The dialog preview must show the font EXACTLY as the terminal will
   * draw it: the same fontStackFor() chain (chosen family first, bundled
   * FONT_STACK trailing — unknown names degrade to the bundled Nerd font,
   * which the @font-face in style.css provides to this page too) and the
   * same 6–40 clamp. The preview is especially useful on the welcome
   * screen, where the terminal itself is hidden. */
  function dialogFamily() {
    return elSettingsFont.value === FONT_CUSTOM_VALUE
      ? elSettingsCustom.value.trim()
      : elSettingsFont.value;
  }

  function updateFontPreview() {
    if (!elSettingsPreview) return;
    var size = parseInt(elSettingsSize.value, 10);
    if (isNaN(size)) size = DEFAULT_FONT_SIZE;
    size = Math.min(MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE, size));
    elSettingsPreview.style.fontFamily = fontStackFor(dialogFamily());
    elSettingsPreview.style.fontSize = size + "px";
  }

  function openSettings() {
    if (!elSettingsDialog) return;
    populateFontSelect(currentFontFamily);
    var size = activeTab && activeTab.term && activeTab.term.options.fontSize
      ? activeTab.term.options.fontSize
      : (savedSettings && typeof savedSettings.font_size === "number"
          ? savedSettings.font_size
          : DEFAULT_FONT_SIZE);
    elSettingsSize.value = String(size);
    syncSizeLabel();
    updateFontPreview();
    elSettingsError.classList.add("hidden");
    elSettingsDialog.classList.remove("hidden");
    elSettingsFont.focus();
  }

  function closeSettings() {
    if (elSettingsDialog) elSettingsDialog.classList.add("hidden");
    // The dialog (opened from the hotkey, the welcome button or the
    // floating button) moved focus to its first field — hand it back to
    // the active terminal so hotkeys and typing keep working without a
    // click. On the welcome screen there is no session to refocus.
    if (activeTab && activeTab.term && activeTab.started) activeTab.term.focus();
  }

  function showSettingsError(message) {
    elSettingsError.textContent = message;
    elSettingsError.classList.remove("hidden");
  }

  function saveSettings() {
    var choice = elSettingsFont.value;
    var family = choice === FONT_CUSTOM_VALUE
      ? elSettingsCustom.value.trim()
      : choice;
    var size = parseInt(elSettingsSize.value, 10);
    if (isNaN(size)) size = null;
    // Apply exactly what the python side validated and stored — it is the
    // clamp authority (6–40); a bridge-less browser mode applies locally.
    var setFont = api("set_font_settings");
    if (!setFont) {
      applyFontFamily(family);
      if (size !== null) applyFontSize(size);
      closeSettings();
      return;
    }
    Promise.resolve(setFont(family, size))
      .then(function (res) {
        if (res) {
          applyFontFamily(res.font_family);
          if (typeof res.font_size === "number") applyFontSize(res.font_size);
        }
        closeSettings();
      })
      .catch(function (err) {
        showSettingsError("Failed to save settings: " + err);
      });
  }

  /* ---------------- websocket bridge (one ws per tab) ---------------- */

  function connectTab(tab) {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    var ws = new WebSocket(
      proto + "//" + location.host + "/ws?token=" + encodeURIComponent(TOKEN) +
      "&tab=" + encodeURIComponent(tab.id)
    );
    ws.binaryType = "arraybuffer";
    tab.ws = ws;
    tab.filter = makeOsc52Filter(
      function (bytes) {
        if (tab.term) tab.term.write(bytes);
      },
      copyToClipboard
    );

    ws.onopen = function () {
      if (tab.project) {
        var size = currentSize(tab);
        ws.send(JSON.stringify({ type: "start", project: tab.project, cols: size.cols, rows: size.rows, tab: tab.id }));
      }
    };

    ws.onmessage = function (msg) {
      if (msg.data instanceof ArrayBuffer) {
        tab.filter(new Uint8Array(msg.data));
        return;
      }
      var event;
      try { event = JSON.parse(msg.data); } catch (err) { return; }
      handleTabEvent(tab, event);
    };

    ws.onclose = function () {
      if (tab.started && !tab.closing && activeTab === tab) {
        // The shell server itself went away before any exit event arrived.
        bootError("Connection to the DrCode shell server was lost.");
      }
    };
  }

  function handleTabEvent(tab, event) {
    if (event.event === "started") {
      tab.started = true;
      if (event.project) tab.project = event.project;
      if (elWelcome.parentNode === tab.pane) elWelcome.remove();
      if (tab.elTerm) tab.elTerm.classList.remove("hidden");
      refreshTabLabels();
      hideDialog();
      if (tab.fit) {
        try { tab.fit.fit(); } catch (err) { /* noop */ }
      }
      sendResize(tab);
      if (activeTab === tab) {
        if (tab.term) tab.term.focus();
        setWindowTitle("DrCode — " + projectTitle(tab.project));
        if (elFontBtn) elFontBtn.classList.remove("hidden");
      }
    } else if (event.event === "exited") {
      tab.started = false;
      onTabExited(tab, event.code);
    } else if (event.event === "error") {
      if (!tab.started) {
        // The start never happened (missing directory, spawn failure):
        // revert the tab instead of keeping a renamed phantom.
        revertTabToWelcome(tab);
        if (activeTab === tab) showWelcomeError(event.message);
      } else if (activeTab === tab) {
        bootError(event.message);
      }
    }
  }

  function onTabExited(tab, code) {
    if (code === 0 || tab.closing) {
      // Clean exit (or a deliberate kill): close THIS tab; the window
      // itself goes away when the last tab is gone (closeTab handles it).
      closeTab(tab);
      updateFontBtnVisibility();
      return;
    }
    // Crash (or external kill): offer a restart instead of closing.
    dialogTab = tab;
    elExitText.textContent =
      "The drcode process exited unexpectedly with code " + code +
      (tab.project ? " while working in " + tab.project + "." : ".");
    elExitDialog.classList.remove("hidden");
  }

  function hideDialog() {
    elExitDialog.classList.add("hidden");
  }

  function showWelcomeError(message) {
    elWelcomeError.textContent = message;
    elWelcomeError.classList.remove("hidden");
  }

  /* A failed start (e.g. a recents entry whose directory was deleted)
   * must not leave a phantom tab behind: launchProject has already
   * renamed the tab and converted it to "terminal", so the error
   * handler reverts it to the welcome picker it was launched from —
   * clicking the same bad entry again then just re-shows the error
   * instead of stacking another renamed tab. */
  function revertTabToWelcome(tab) {
    tab.project = null;
    tab.kind = "welcome";
    tab.started = false;
    if (tab.term) {
      try { tab.term.dispose(); } catch (err) { /* already gone */ }
      tab.term = null;
    }
    tab.fit = null;
    tab.webgl = null;
    tab.filter = null;
    if (tab.elTerm) {
      if (resizeObserver) {
        try { resizeObserver.unobserve(tab.elTerm); } catch (err) { /* gone */ }
      }
      tab.elTerm.remove();
      tab.elTerm = null;
    }
    refreshTabLabels();
    if (activeTab === tab) activateTab(tab);
  }

  /* One drcode instance per project: its state lives in the shared
   * <project>/.drcode/ (tasks, worktrees, search index), so a second
   * tab over the same path would race it. Best-effort pre-check on the
   * open tabs (the server re-checks with realpath normalization and
   * answers an error event — this only spares the round trip). */
  function projectOccupied(project, excludeTab) {
    if (!project) return false;
    return tabs.some(function (t) {
      return t !== excludeTab && t.kind === "terminal" && t.started && t.project === project;
    });
  }

  function duplicateMessage(project) {
    return "This project is already open in another tab: " + project +
      ". Close that tab first.";
  }

  function launchProject(tab, project) {
    if (!tab || !project) return;
    if (projectOccupied(project, tab)) {
      if (tab.started) bootError(duplicateMessage(project));
      else showWelcomeError(duplicateMessage(project));
      return;
    }
    if (activeTab !== tab) activateTab(tab);
    elWelcomeError.classList.add("hidden");
    tab.project = project;
    if (!tab.elTerm) {
      tab.kind = "terminal";
      createTerminalHost(tab);
      createTerminalFor(tab);
    }
    refreshTabLabels();
    if (!tab.ws) {
      connectTab(tab); // start message is sent from ws.onopen
    } else if (tab.ws.readyState === WebSocket.OPEN) {
      var size = currentSize(tab);
      tab.ws.send(JSON.stringify({ type: "start", project: project, cols: size.cols, rows: size.rows, tab: tab.id }));
    }
  }

  /* One-shot availability probe for the font dialog, retried with the
   * same 250 ms × 40 pattern as the other bridge calls (pywebview
   * injects asynchronously). The FIRST api() lookup must be inside the
   * retry loop too: boot() runs on DOMContentLoaded, which in the frozen
   * app regularly wins the race against the bridge injection — an early
   * miss used to bail out permanently (if (!getFonts) return) and the
   * dialog stayed unfiltered forever (the Python probe was never even
   * called). Failure or absence of the bridge leaves availableFonts
   * null → the dialog shows the full curated list. A dialog opened
   * before the probe lands is simply unfiltered — the next open is
   * filtered. */
  function fetchAvailableFonts() {
    function apply(list) {
      if (Array.isArray(list)) availableFonts = list;
    }
    var attempts = 0;
    var timer = setInterval(function () {
      attempts += 1;
      var getFonts = api("get_available_fonts");
      if (getFonts) {
        clearInterval(timer);
        Promise.resolve(getFonts()).then(apply).catch(function () {});
      } else if (attempts > 40) {
        clearInterval(timer); // bridge never arrives: stay unfiltered
      }
    }, 250);
  }

  /* ---------------- welcome screen ---------------- */

  function renderRecents(recents) {
    if (!recents || !recents.length) {
      elRecentsBlock.classList.add("hidden");
      return;
    }
    lastRecents = recents;
    elRecentsList.innerHTML = "";
    recents.forEach(function (item) {
      var li = document.createElement("li");
      var occupied = projectOccupied(item.project, null);
      if (occupied) {
        li.className = "occupied";
        li.title = "This project is already open in another tab";
      }
      var path = document.createElement("span");
      path.className = "path";
      path.textContent = item.project;
      var when = document.createElement("span");
      when.className = "when";
      when.textContent = item.last_used ? new Date(item.last_used * 1000).toLocaleDateString() : "";
      li.appendChild(path);
      li.appendChild(when);
      var remove = document.createElement("span");
      remove.className = "remove";
      remove.textContent = "×";
      remove.title = "Remove from recent projects";
      remove.setAttribute("aria-label", "Remove from recent projects");
      remove.addEventListener("click", function (ev) {
        // Delete the recents entry, not the project: stop the li's
        // launch/error handler before it fires.
        ev.stopPropagation();
        var doRemove = api("remove_recent");
        if (!doRemove) return; // no bridge (browser mode) — nothing to do
        Promise.resolve(doRemove(item.project))
          .then(function (recents) { renderRecents(recents); })
          .catch(function () { /* storage is best-effort; keep the list */ });
      });
      li.appendChild(remove);
      if (occupied) {
        var mark = document.createElement("span");
        mark.className = "open-mark";
        mark.textContent = "open";
        li.appendChild(mark);
        // Occupied entries are not launchable — clicking explains why.
        li.addEventListener("click", function () {
          showWelcomeError(duplicateMessage(item.project));
        });
      } else {
        li.addEventListener("click", function () { launchFromPicker(item.project); });
      }
      elRecentsList.appendChild(li);
    });
    elRecentsBlock.classList.remove("hidden");
  }

  /* Projects picked on the welcome screen open in the tab the welcome
   * node currently lives in — unless that tab is already a live session,
   * in which case a fresh tab takes it (the picker is only reachable
   * from a welcome tab in practice). */
  function launchFromPicker(project) {
    // Blocked BEFORE any tab is created/activated: an occupied project
    // must not leave an extra welcome tab behind.
    if (projectOccupied(project, null)) {
      showWelcomeError(duplicateMessage(project));
      return;
    }
    var tab = activeTab && activeTab.kind === "welcome"
      ? activeTab
      : createTab("welcome", null);
    launchProject(tab, project);
  }

  function loadRecents() {
    var getInfo = api("get_info");
    if (getInfo) {
      Promise.resolve(getInfo())
        .then(function (info) {
          if (info && info.recents) renderRecents(info.recents);
        })
        .catch(function () { /* recents are cosmetic */ });
      return;
    }
    // pywebview injects its bridge asynchronously; recents may need a retry.
    var attempts = 0;
    var timer = setInterval(function () {
      attempts += 1;
      var retry = api("get_info");
      if (retry) {
        clearInterval(timer);
        Promise.resolve(retry()).then(function (info) {
          if (info && info.recents) renderRecents(info.recents);
        });
      } else if (attempts > 40) {
        clearInterval(timer);
      }
    }, 250);
  }

  function boot() {
    wireUi();
    fetchAvailableFonts();

    /* The bridge injects asynchronously, so settings load with the same
     * retry pattern as recents — and the first Terminal constructor
     * (which bakes fontFamily/fontSize into its renderer) waits for
     * them. */
    function applyAndStart(settings) {
      savedSettings = settings || {};
      currentFontFamily = savedSettings.font_family || null;
      startShell();
    }

    var getSettings = api("get_settings");
    if (getSettings) {
      Promise.resolve(getSettings())
        .then(applyAndStart)
        .catch(function () { applyAndStart(null); });
      return;
    }
    var attempts = 0;
    var timer = setInterval(function () {
      attempts += 1;
      var retry = api("get_settings");
      if (retry) {
        clearInterval(timer);
        Promise.resolve(retry())
          .then(applyAndStart)
          .catch(function () { applyAndStart(null); });
      } else if (attempts > 40) {
        // ~10 s without the bridge — browser mode: defaults, no persistence.
        clearInterval(timer);
        applyAndStart(null);
      }
    }, 250);
  }

  function startShell() {
    loadRecents();
    // CLI-pinned project (--project at app start): tab #1 auto-launches,
    // skipping the welcome screen. Everything else opens from the UI.
    var projectArg = params.get("project");
    if (projectArg) {
      var tab = createTab("terminal", projectArg);
      launchProject(tab, projectArg);
    } else {
      createTab("welcome", null);
    }
  }

  function wireUi() {
    function runChooseProject(choose) {
      Promise.resolve(choose()).then(function (path) {
        if (path) launchFromPicker(path);
      });
    }

    elPick.addEventListener("click", function () {
      var choose = api("choose_project");
      if (choose) {
        runChooseProject(choose);
        return;
      }
      // pywebview injects its bridge asynchronously; an early click may
      // land before that — wait for the bridge (same budget as the
      // recents retry below) instead of reporting the dialog unavailable
      // on the first miss.
      if (elPick.dataset.waiting) return; // a retry is already running
      elPick.dataset.waiting = "1";
      var attempts = 0;
      var timer = setInterval(function () {
        attempts += 1;
        var retry = api("choose_project");
        if (retry) {
          clearInterval(timer);
          delete elPick.dataset.waiting;
          runChooseProject(retry);
        } else if (attempts > 40) {
          // ~10 s without the bridge — really no shell around the page.
          clearInterval(timer);
          delete elPick.dataset.waiting;
          showWelcomeError("Native folder dialog is unavailable in this environment (no pywebview bridge — browser mode?). Pass --project to the binary or pick a recent project.");
        }
      }, 250);
    });

    document.getElementById("btn-restart").addEventListener("click", function () {
      var tab = dialogTab;
      hideDialog();
      dialogTab = null;
      if (tab && tab.project) launchProject(tab, tab.project);
    });
    document.getElementById("btn-close").addEventListener("click", function () {
      var tab = dialogTab;
      hideDialog();
      dialogTab = null;
      if (tab) closeTab(tab); // last tab closed → the window follows
      else {
        var closeWindow = api("close_window");
        if (closeWindow) closeWindow();
        else window.close();
      }
    });

    // "New project" next to the tab bar: a fresh welcome tab.
    if (elNewTab) {
      elNewTab.addEventListener("click", function () {
        createTab("welcome", null);
      });
    }

    // Settings dialog wiring (open from the welcome button, the floating
    // session button or the Ctrl+, hotkey — intercepted in
    // terminalKeyHandler when the terminal has focus and in the window
    // keydown listener otherwise).
    if (elOpenSettings) {
      elOpenSettings.addEventListener("click", openSettings);
    }
    if (elFontBtn) {
      // Suppress mousedown's default (moving focus out of the terminal):
      // the click still fires and opens the dialog, but keys the user is
      // already typing are not stolen mid-word. The button itself needs
      // no keyboard focus — Ctrl+, reaches openSettings from anywhere.
      elFontBtn.addEventListener("mousedown", function (ev) {
        ev.preventDefault();
      });
      elFontBtn.addEventListener("click", openSettings);
    }
    if (elSettingsFont) {
      elSettingsFont.addEventListener("change", function () {
        var custom = elSettingsFont.value === FONT_CUSTOM_VALUE;
        elSettingsCustom.classList.toggle("hidden", !custom);
        if (custom) elSettingsCustom.focus();
        updateFontPreview();
      });
    }
    if (elSettingsCustom) {
      // "Custom…": the preview tracks the family as it is typed.
      elSettingsCustom.addEventListener("input", updateFontPreview);
    }
    if (elSettingsSize) {
      elSettingsSize.addEventListener("input", function () {
        syncSizeLabel();
        updateFontPreview();
      });
    }
    var btnSave = document.getElementById("btn-settings-save");
    var btnCancel = document.getElementById("btn-settings-cancel");
    if (btnSave) btnSave.addEventListener("click", saveSettings);
    if (btnCancel) btnCancel.addEventListener("click", closeSettings);

    window.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && elSettingsDialog && !elSettingsDialog.classList.contains("hidden")) {
        ev.preventDefault();
        closeSettings();
      } else if (ev.key === "F11") {
        ev.preventDefault();
        var fs = api("toggle_fullscreen");
        if (fs) fs();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.shiftKey && (ev.key === "r" || ev.key === "R")) {
        ev.preventDefault();
        // Restart the ACTIVE tab's project.
        if (activeTab && activeTab.project) launchProject(activeTab, activeTab.project);
      } else if ((ev.metaKey || ev.ctrlKey) && !ev.shiftKey && !ev.altKey && ev.key === ",") {
        // Welcome screen / focus outside the terminal: the in-terminal
        // case is handled by terminalKeyHandler before xterm forwards
        // the combo to the PTY.
        ev.preventDefault();
        openSettings();
      }
    });

    // Keep every PTY size in sync with the window size — but only the
    // ACTIVE tab sends resizes (hidden panes keep their stale geometry;
    // switching back re-fits and re-sends).
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(function (entries) {
        var touchesActive = entries.some(function (entry) {
          return activeTab && entry.target === activeTab.elTerm;
        });
        if (!touchesActive) return;
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function () { sendResize(activeTab); }, 80); // debounce live-drag resizes
      });
    }
    window.addEventListener("resize", function () { sendResize(activeTab); });
  }

  var resizeObserver = null;
  var resizeTimer = null;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
