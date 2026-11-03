/*
 * DrCode Desktop — terminal client.
 *
 * xterm.js (vendored UMD bundles, no frontend build) wired to the local
 * aiohttp server's /ws bridge:
 *   - binary frames: raw PTY output (server->client) / stdin (client->server)
 *   - text frames:   JSON control events (start/resize/exited/...)
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

  var elTerminal = document.getElementById("terminal");
  var elWelcome = document.getElementById("welcome");
  var elPick = document.getElementById("pick-project");
  var elRecentsBlock = document.getElementById("recents-block");
  var elRecentsList = document.getElementById("recents-list");
  var elWelcomeError = document.getElementById("welcome-error");
  var elExitDialog = document.getElementById("exit-dialog");
  var elExitText = document.getElementById("exit-text");
  var elBootError = document.getElementById("boot-error");

  var FONT_STACK = '"JetBrainsMono Nerd Font Mono", "JetBrains Mono", Menlo, monospace';

  var DEFAULT_FONT_SIZE = 13;
  var MIN_FONT_SIZE = 6;
  var MAX_FONT_SIZE = 40;
  var FONT_STEP = 1;

  var term = null;
  var fit = null;
  var webgl = null;
  var ws = null;
  var currentProject = null;
  var started = false;
  var closing = false; // clean exit in progress: suppress the crash dialog

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

  /* ---------------- terminal setup ---------------- */

  function createTerminal() {
    term = new window.Terminal({
      fontFamily: FONT_STACK,
      fontSize: 13,
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
    term.loadAddon(new window.Unicode11Addon.Unicode11Addon());
    term.unicode.activeVersion = "11";
    fit = new window.FitAddon.FitAddon();
    term.loadAddon(fit);
    term.open(elTerminal);
    try {
      webgl = new window.WebglAddon.WebglAddon();
      webgl.onContextLoss(function () {
        if (webgl) { webgl.dispose(); webgl = null; }
      }); // xterm falls back to DOM renderer
      term.loadAddon(webgl);
    } catch (err) {
      webgl = null;
      /* no WebGL — the default renderer keeps everything working */
    }
    term.attachCustomKeyEventHandler(terminalKeyHandler);
    term.onData(function (data) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(new TextEncoder().encode(data));
      }
    });
  }

  /* ---------------- font size (Ctrl+= / Ctrl+- / Ctrl+0) ----------------
   * Without interception these combos would be forwarded to the PTY
   * through term.onData; the custom handler swallows them (return false)
   * and changes term.options.fontSize instead. The WebGL renderer in
   * xterm 5.x can keep a stale glyph atlas across a font-size change, so
   * the addon is re-created after each change; on failure the DOM
   * renderer takes over. fit.fit() recomputes cols/rows for the new cell
   * metrics and sendResize() propagates them to the PTY.
   */

  function applyFontSize(size) {
    if (!term) return;
    var clamped = Math.min(MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE, size));
    if (clamped === term.options.fontSize) return;
    term.options.fontSize = clamped;
    if (webgl) {
      try { webgl.dispose(); } catch (err) { /* already gone */ }
      webgl = null;
      try {
        webgl = new window.WebglAddon.WebglAddon();
        webgl.onContextLoss(function () {
          if (webgl) { webgl.dispose(); webgl = null; }
        });
        term.loadAddon(webgl);
      } catch (err) {
        webgl = null; /* DOM renderer fallback */
      }
    }
    if (fit) {
      try { fit.fit(); } catch (err) { /* container not measurable */ }
    }
    sendResize();
  }

  function terminalKeyHandler(ev) {
    if (ev.key === "F11") return true; // handled by the window listener
    if (!ev.ctrlKey || ev.metaKey || ev.altKey || ev.shiftKey) return true;
    var code = ev.code || "";
    if (ev.key === "=" || ev.key === "+" || code === "NumpadAdd") {
      ev.preventDefault();
      applyFontSize((term.options.fontSize || DEFAULT_FONT_SIZE) + FONT_STEP);
      return false;
    }
    if (ev.key === "-" || ev.key === "_" || code === "NumpadSubtract") {
      ev.preventDefault();
      applyFontSize((term.options.fontSize || DEFAULT_FONT_SIZE) - FONT_STEP);
      return false;
    }
    if (ev.key === "0" || code === "Digit0" || code === "Numpad0") {
      ev.preventDefault();
      applyFontSize(DEFAULT_FONT_SIZE);
      return false;
    }
    return true;
  }

  function currentSize() {
    var size = { cols: 80, rows: 24 };
    if (fit) {
      try { fit.fit(); } catch (err) { /* container not measurable yet */ }
      var proposal = fit.proposeDimensions();
      if (proposal && proposal.cols && proposal.rows) {
        size.cols = proposal.cols;
        size.rows = proposal.rows;
      }
    }
    return size;
  }

  function sendResize() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !started) return;
    var size = currentSize();
    ws.send(JSON.stringify({ type: "resize", cols: size.cols, rows: size.rows }));
  }

  /* ---------------- websocket bridge ---------------- */

  function connect() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws?token=" + encodeURIComponent(TOKEN));
    ws.binaryType = "arraybuffer";

    var filter = makeOsc52Filter(
      function (bytes) {
        if (term) term.write(bytes);
      },
      copyToClipboard
    );

    ws.onopen = function () {
      var size = currentSize();
      if (currentProject) {
        ws.send(JSON.stringify({ type: "start", project: currentProject, cols: size.cols, rows: size.rows }));
      }
    };

    ws.onmessage = function (msg) {
      if (msg.data instanceof ArrayBuffer) {
        filter(new Uint8Array(msg.data));
        return;
      }
      var event;
      try { event = JSON.parse(msg.data); } catch (err) { return; }
      handleEvent(event);
    };

    ws.onclose = function () {
      if (started && !closing) {
        // The shell server itself went away before any exit event arrived.
        bootError("Connection to the DrCode shell server was lost.");
      }
    };
  }

  function handleEvent(event) {
    if (event.event === "started") {
      started = true;
      currentProject = event.project || currentProject;
      elWelcome.classList.add("hidden");
      elTerminal.classList.remove("hidden");
      hideDialog();
      if (fit) {
        try { fit.fit(); } catch (err) { /* noop */ }
      }
      sendResize();
      if (term) term.focus();
      var setTitle = api("set_title");
      if (setTitle) {
        var name = currentProject.split("/").filter(Boolean).pop() || currentProject;
        setTitle("DrCode — " + name);
      }
    } else if (event.event === "exited") {
      started = false;
      onProcessExited(event.code);
    } else if (event.event === "error") {
      if (!started) showWelcomeError(event.message);
      else bootError(event.message);
    }
  }

  function onProcessExited(code) {
    if (code === 0 || closing) {
      closing = true;
      var closeWindow = api("close_window");
      if (closeWindow) closeWindow();
      else window.close();
      return;
    }
    // Crash (or external kill): offer a restart instead of closing.
    elExitText.textContent =
      "The drcode process exited unexpectedly with code " + code +
      (currentProject ? " while working in " + currentProject + "." : ".");
    elExitDialog.classList.remove("hidden");
  }

  function hideDialog() {
    elExitDialog.classList.add("hidden");
  }

  function showWelcomeError(message) {
    elWelcomeError.textContent = message;
    elWelcomeError.classList.remove("hidden");
  }

  function launchProject(project) {
    if (!project) return;
    elWelcomeError.classList.add("hidden");
    currentProject = project;
    var size = currentSize();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "start", project: project, cols: size.cols, rows: size.rows }));
    }
  }

  /* ---------------- welcome screen ---------------- */

  function renderRecents(recents) {
    if (!recents || !recents.length) {
      elRecentsBlock.classList.add("hidden");
      return;
    }
    elRecentsList.innerHTML = "";
    recents.forEach(function (item) {
      var li = document.createElement("li");
      var path = document.createElement("span");
      path.className = "path";
      path.textContent = item.project;
      var when = document.createElement("span");
      when.className = "when";
      when.textContent = item.last_used ? new Date(item.last_used * 1000).toLocaleDateString() : "";
      li.appendChild(path);
      li.appendChild(when);
      li.addEventListener("click", function () { launchProject(item.project); });
      elRecentsList.appendChild(li);
    });
    elRecentsBlock.classList.remove("hidden");
  }

  function boot() {
    createTerminal();

    function runChooseProject(choose) {
      Promise.resolve(choose()).then(function (path) {
        if (path) launchProject(path);
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
      hideDialog();
      launchProject(currentProject);
    });
    document.getElementById("btn-close").addEventListener("click", function () {
      var closeWindow = api("close_window");
      if (closeWindow) closeWindow();
      else window.close();
    });

    window.addEventListener("keydown", function (ev) {
      if (ev.key === "F11") {
        ev.preventDefault();
        var fs = api("toggle_fullscreen");
        if (fs) fs();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.shiftKey && (ev.key === "r" || ev.key === "R")) {
        ev.preventDefault();
        if (currentProject) launchProject(currentProject);
      }
    });

    // Keep the PTY size in sync with the window size.
    if (typeof ResizeObserver !== "undefined") {
      var pending = null;
      var ro = new ResizeObserver(function () {
        if (pending) clearTimeout(pending);
        pending = setTimeout(sendResize, 80); // debounce live-drag resizes
      });
      ro.observe(elTerminal);
    }
    window.addEventListener("resize", sendResize);

    // Populate recents from the shell (pywebview bridge), then connect.
    var getInfo = api("get_info");
    var ready = getInfo
      ? Promise.resolve(getInfo()).then(function (info) {
          if (info && info.recents) renderRecents(info.recents);
        })
      : Promise.resolve();

    // CLI-pinned project (--project at app start): auto-launch, skip the welcome.
    var projectArg = params.get("project");
    if (projectArg) launchProject(projectArg);

    ready
      .then(function () { connect(); })
      .catch(function (err) {
        connect(); // still work, just without recents
        bootError("Failed to load shell info: " + err);
      });

    // pywebview injects its bridge asynchronously; recents may need a retry.
    if (!getInfo) {
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
