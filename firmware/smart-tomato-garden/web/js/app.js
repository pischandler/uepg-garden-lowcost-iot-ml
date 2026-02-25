(function () {
  const R = window.STGRender;
  const A = window.STGApi;
  const I = window.STGInference;
  const F = window.STGFmt;
  const T = window.STGI18n;
  function t(key, params) { return T.t(key, params); }

  const state = {
    streaming: true,
    feed: [],
    refreshInFlight: null,
    streamTick: null,
    snapshotTick: null,
    streamProfile: "auto",
    lastInferTs: 0,
    reconnectCount: 0,
    fpsFrameCount: 0,
    fpsLastTs: 0,
    fpsEstimate: 0,
    pollTimer: null,
    pollMs: 3500,
    hist: {
      temp: [],
      soil: [],
      pump: [],
      ts: []
    }
  };
  const HIGH_RES_STREAM_FS = 8; // VGA+
  const MID_RES_STREAM_FS = 6; // CIF+

  function pushHist(arr, value, max) {
    arr.push(value);
    while (arr.length > max) arr.shift();
  }

  function rangeToMs(v) {
    if (v === "5m") return 5 * 60 * 1000;
    if (v === "2h") return 2 * 60 * 60 * 1000;
    return 30 * 60 * 1000;
  }

  function filteredHistory() {
    const range = R.$("trendRange") ? R.$("trendRange").value : "30m";
    const windowMs = rangeToMs(range);
    const now = Date.now();
    const out = { temp: [], soil: [], pump: [] };
    for (let i = 0; i < state.hist.ts.length; i++) {
      if (now - state.hist.ts[i] <= windowMs) {
        out.temp.push(state.hist.temp[i]);
        out.soil.push(state.hist.soil[i]);
        out.pump.push(state.hist.pump[i]);
      }
    }
    return out;
  }

  function exportTrendCsv() {
    const h = filteredHistory();
    if (!h.temp.length) return;
    const rows = ["idx,temp_c,soil_pct,pump_on"];
    for (let i = 0; i < h.temp.length; i++) {
      rows.push([i, h.temp[i], h.soil[i], h.pump[i]].join(","));
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "stg_trends.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function pushFeed(kind, msg) {
    const time = new Date().toLocaleTimeString();
    state.feed.unshift({ kind: kind, msg: msg, time: time });
    state.feed = state.feed.slice(0, 20);
    const ul = R.$("feed");
    ul.innerHTML = "";
    if (!state.feed.length) {
      const empty = document.createElement("li");
      empty.className = "emptyFeed";
      empty.textContent = t("events.none");
      ul.appendChild(empty);
      return;
    }
    state.feed.forEach(function (line) {
      const li = document.createElement("li");
      const tag = document.createElement("span");
      tag.className = "feedTag";
      tag.textContent = line.kind;
      li.appendChild(tag);
      li.appendChild(document.createTextNode("[" + line.time + "] " + line.msg));
      ul.appendChild(li);
    });
  }

  function setBtnLoading(id, on, busyText) {
    const btn = R.$(id);
    if (!btn) return;
    if (!btn.dataset.idleText) btn.dataset.idleText = btn.textContent;
    if (on) {
      btn.disabled = true;
      btn.classList.add("is-loading");
      if (busyText) btn.textContent = t(busyText);
    } else {
      btn.disabled = false;
      btn.classList.remove("is-loading");
      btn.textContent = btn.dataset.idleText;
    }
  }

  function setHud() {
    const transport = shouldUseSnapshotLoop() ? "snapshot" : "mjpeg";
    R.$("hudTransport").textContent = "mode " + transport;
    R.$("hudFps").textContent = "fps " + (state.fpsEstimate ? state.fpsEstimate.toFixed(1) : "-");
    R.$("hudReconnect").textContent = "reconnect " + state.reconnectCount;
  }

  async function runAction(meta, fn) {
    const id = meta.id;
    try {
      setBtnLoading(id, true, meta.busyText);
      await fn();
      if (meta.toastOk) R.showToast(t(meta.toastOk), "ok");
    } catch (err) {
      const msg = err && err.message ? err.message : "erro inesperado";
      pushFeed(meta.feedKind || "sistema", t("generic.error", { error: msg }));
      R.showToast(meta.toastErr ? t(meta.toastErr) : t("generic.action_failed", { error: msg }), "bad", 3200);
      return false;
    } finally {
      setBtnLoading(id, false);
    }
    return true;
  }

  function updateStream(on) {
    state.streaming = on;
    stopStreamLoops();
    R.setCameraLoading(true, on ? t("loading.camera") : t("loading.capture"));
    if (on) startStreamTransport();
    else R.$("view").src = "/capture?ts=" + Date.now();
    R.$("btnStream").textContent = on ? t("action.pause_stream") : t("action.resume_stream");
    R.setBadge("streamState", on ? t("status.stream_active") : t("status.stream_paused"), on ? "ok" : "warn");
  }

  function isHighResStream() {
    const fs = Number(R.$("framesize").value || 0);
    return fs >= HIGH_RES_STREAM_FS;
  }

  function isMidOrHighResStream() {
    const fs = Number(R.$("framesize").value || 0);
    return fs >= MID_RES_STREAM_FS;
  }

  function shouldUseSnapshotLoop() {
    if (state.streamProfile === "stable") return true;
    if (state.streamProfile === "fast") return false;
    return isMidOrHighResStream();
  }

  function streamIntervalMs() {
    const fs = Number(R.$("framesize").value || 0);
    if (fs >= 10) return 1100;
    if (fs >= 9) return 900;
    if (fs >= 8) return 700;
    if (fs >= 6) return 550;
    return 420;
  }

  function setStreamHint() {
    const hint = R.$("streamHint");
    if (!hint) return;
    let key = "stream.mode_hint_auto_mjpeg";
    if (state.streamProfile === "stable") key = "stream.mode_hint_forced_stable";
    else if (state.streamProfile === "fast") key = "stream.mode_hint_forced_fast";
    else if (shouldUseSnapshotLoop()) key = "stream.mode_hint_auto_snapshot";
    hint.textContent = t(key);
    setHud();
  }

  function stopStreamLoops() {
    if (state.streamTick) clearInterval(state.streamTick);
    if (state.snapshotTick) clearTimeout(state.snapshotTick);
    state.streamTick = null;
    state.snapshotTick = null;
  }

  function snapshotFrameLoop() {
    if (!state.streaming) return;
    R.$("view").src = "/capture?ts=" + Date.now();
    state.snapshotTick = setTimeout(snapshotFrameLoop, streamIntervalMs());
  }

  function startStreamTransport() {
    setStreamHint();
    if (shouldUseSnapshotLoop()) {
      snapshotFrameLoop();
      return;
    }
    R.$("view").src = "/stream?ts=" + Date.now();
    streamWatchdogStart();
  }

  function streamWatchdogStart() {
    if (state.streamTick) clearInterval(state.streamTick);
    state.streamTick = setInterval(function () {
      if (!state.streaming) return;
      if (shouldUseSnapshotLoop()) return;
      if (!isHighResStream()) return;
      // Refresh MJPEG connection periodically in high-res mode to reduce stuck streams.
      R.$("view").src = "/stream?ts=" + Date.now();
    }, 12000);
  }

  async function cameraControl(name, value) {
    await A.textget("/control?var=" + encodeURIComponent(name) + "&val=" + encodeURIComponent(String(value)));
  }

  async function irrigate(ms) {
    const prevTxt = R.$("pumpState").textContent;
    const prevCls = R.$("pumpState").className;
    R.setBadge("pumpState", t("status.pump_on"), "ok");
    await A.jpost("/api/irrigation/start", { ms: ms });
    pushFeed("irrigacao", t("irrigation.started", { duration: F.fmtMs(ms) }));
    try {
      await refreshAll();
    } catch (err) {
      R.$("pumpState").textContent = prevTxt;
      R.$("pumpState").className = prevCls;
      throw err;
    }
  }

  async function stopIrrigation() {
    const prevTxt = R.$("pumpState").textContent;
    const prevCls = R.$("pumpState").className;
    R.setBadge("pumpState", t("status.pump_off"), "");
    await A.jpost("/api/irrigation/stop", {});
    pushFeed("irrigacao", t("irrigation.stopped"));
    try {
      await refreshAll();
    } catch (err) {
      R.$("pumpState").textContent = prevTxt;
      R.$("pumpState").className = prevCls;
      throw err;
    }
  }

  async function refreshAll(opts) {
    const options = opts || {};
    if (state.refreshInFlight) return state.refreshInFlight;
    R.setSyncState(t("status.syncing"), "");
    if (!options.silent) setBtnLoading("btnRefresh", true, "action.refresh_all");

    state.refreshInFlight = (async function () {
    const payload = await A.refreshPayload();
    R.setSystem(payload);
    const vm = I.toViewModel(payload.lastInfer);
    state.lastInferTs = Number(payload.lastInfer && payload.lastInfer.ts_ms) || state.lastInferTs;
    pushHist(state.hist.temp, Number(payload.sensors && payload.sensors.temp_c), 2200);
    pushHist(state.hist.soil, Number(payload.sensors && payload.sensors.soil_pct), 2200);
    pushHist(state.hist.ts, Date.now(), 2200);
    state.hist.pump.push(payload.irrigation && payload.irrigation.pump_on ? 1 : 0);
    while (state.hist.pump.length > 2200) state.hist.pump.shift();
    R.setInference(vm, payload.lastInfer);
    R.setGuidedAlert(vm);
    R.setDashboard(payload, vm, filteredHistory());
    R.setRaw(payload);
      R.setSyncState(t("status.updated_now"), "");
    })();

    try {
      await state.refreshInFlight;
    } finally {
      state.refreshInFlight = null;
      if (!options.silent) setBtnLoading("btnRefresh", false);
    }
  }

  function bindUi() {
    const tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));
    const panels = Array.prototype.slice.call(document.querySelectorAll(".panel"));
    function activateTab(name) {
      tabs.forEach(function (b) { b.classList.toggle("is-active", b.dataset.tab === name); });
      panels.forEach(function (p) {
        const opts = (p.dataset.panel || "overview").split(" ");
        p.classList.toggle("is-hidden", opts.indexOf(name) < 0);
      });
    }
    tabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        activateTab(btn.dataset.tab || "overview");
      });
    });
    activateTab("overview");

    const camera = R.$("view");
    camera.addEventListener("load", function () {
      const now = performance.now();
      state.fpsFrameCount++;
      if (!state.fpsLastTs) state.fpsLastTs = now;
      const dt = now - state.fpsLastTs;
      if (dt >= 1000) {
        state.fpsEstimate = (state.fpsFrameCount * 1000) / dt;
        state.fpsFrameCount = 0;
        state.fpsLastTs = now;
        setHud();
      }
      R.setCameraLoading(false);
    });
    camera.addEventListener("error", function () {
      state.reconnectCount++;
      R.setCameraLoading(true, t("camera.load_fail_reconnect"));
      R.showToast(t("camera.stream_fail"), "warn");
      pushFeed("camera", t("camera.stream_fail"));
      if (state.streaming) {
        stopStreamLoops();
        setTimeout(function () { if (state.streaming) startStreamTransport(); }, 900);
      }
      setHud();
    });

    R.$("btnStream").addEventListener("click", function () {
      updateStream(!state.streaming);
      pushFeed("camera", state.streaming ? t("camera.stream_resumed") : t("camera.stream_paused"));
    });

    R.$("btnSnap").addEventListener("click", async function () {
      await runAction(
        { id: "btnSnap", busyText: "action.capture_photo", toastOk: "camera.photo_captured", toastErr: "camera.photo_capture_failed", feedKind: "camera" },
        async function () {
          const r = await fetch("/capture", { cache: "no-store" });
          if (!r.ok) throw new Error("capture " + r.status);
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          window.open(url, "_blank");
          pushFeed("camera", t("camera.photo_captured"));
        }
      );
    });

    R.$("btnRefresh").addEventListener("click", async function () {
      await refreshAll({ silent: false });
      pushFeed("sistema", t("refresh.done"));
      R.showToast(t("refresh.done_toast"), "ok");
    });

    R.$("btnRetrySync").addEventListener("click", async function () {
      await refreshAll({ silent: false });
      pushFeed("sistema", t("connection.restored"));
      R.showToast(t("connection.restored_toast"), "ok");
    });

    R.$("btnIrrigate").addEventListener("click", async function () {
      await runAction(
        { id: "btnIrrigate", busyText: "action.irrigate_now", toastOk: "irrigation.started_toast", toastErr: "irrigation.start_failed", feedKind: "irrigacao" },
        async function () {
          const ms = Math.max(200, Math.min(30000, Number(R.$("irrigationMs").value || 1500)));
          R.$("irrigationMs").value = String(ms);
          await irrigate(ms);
        }
      );
    });

    Array.prototype.forEach.call(document.querySelectorAll("button.quick"), function (btn) {
      btn.addEventListener("click", async function () {
        const ms = Number(btn.dataset.ms || 1500);
        R.$("irrigationMs").value = String(ms);
        await runAction(
          { id: btn.id || "btnIrrigate", busyText: "action.irrigate_now", toastOk: "irrigation.started_toast", toastErr: "irrigation.start_failed", feedKind: "irrigacao" },
          async function () { await irrigate(ms); }
        );
      });
    });

    R.$("btnStop").addEventListener("click", async function () {
      await runAction(
        { id: "btnStop", busyText: "action.stop", toastOk: "irrigation.stopped_toast", toastErr: "irrigation.stop_failed", feedKind: "irrigacao" },
        stopIrrigation
      );
    });

    R.$("btnRunInference").addEventListener("click", async function () {
      await runAction(
        { id: "btnRunInference", busyText: "action.run_inference", toastOk: "inference.requested_toast", toastErr: "inference.request_failed", feedKind: "inferencia" },
        async function () {
          const wasStreaming = state.streaming;
          const prevTs = Number(state.lastInferTs || 0);
          if (wasStreaming) {
            updateStream(false);
            await sleep(220);
          }

          R.$("diagnosisMain").classList.add("loading");
          await A.jpost("/api/inference/run", {});
          pushFeed("inferencia", t("inference.requested"));

          // Wait for a fresh inference cycle before resuming the stream.
          const start = Date.now();
          while (Date.now() - start < 5000) {
            await sleep(380);
            let last = null;
            try {
              last = await A.jget("/api/inference/last");
            } catch (e) {
              continue;
            }
            const ts = Number(last && last.ts_ms) || 0;
            if (ts > prevTs) break;
          }

          await refreshAll({ silent: true });
          if (wasStreaming) {
            updateStream(true);
          }
        }
      );
    });

    R.$("quality").addEventListener("input", function () {
      R.$("qualityV").textContent = R.$("quality").value;
    });
    R.$("led").addEventListener("input", function () {
      R.$("ledV").textContent = R.$("led").value;
    });

    R.$("quality").addEventListener("change", async function () {
      try {
        await cameraControl("quality", R.$("quality").value);
        pushFeed("camera", t("camera.quality_set", { value: R.$("quality").value }));
      } catch (err) {
        R.showToast(t("camera.quality_set_fail"), "bad");
      }
    });

    R.$("framesize").addEventListener("change", async function () {
      try {
        await cameraControl("framesize", R.$("framesize").value);
        pushFeed("camera", t("camera.resolution_set", { value: R.$("framesize").selectedOptions[0].textContent }));
        if (state.streaming) {
          stopStreamLoops();
          startStreamTransport();
        }
        if (isHighResStream()) {
          R.showToast(t("camera.high_res_warning"), "warn", 3800);
          pushFeed("camera", t("camera.high_res_warning"));
        }
      } catch (err) {
        R.showToast(t("camera.resolution_set_fail"), "bad");
      }
    });

    R.$("led").addEventListener("change", async function () {
      try {
        await cameraControl("led_intensity", R.$("led").value);
        pushFeed("camera", t("camera.led_set", { value: R.$("led").value }));
      } catch (err) {
        R.showToast(t("camera.led_set_fail"), "bad");
      }
    });

    R.$("langSelect").addEventListener("change", function () {
      T.setLang(R.$("langSelect").value);
      updateStream(state.streaming);
      pushFeed("sistema", "LANG " + T.getLang());
      refreshAll({ silent: true }).catch(function () {});
    });
    R.$("streamProfile").addEventListener("change", function () {
      state.streamProfile = R.$("streamProfile").value || "auto";
      setStreamHint();
      if (state.streaming) {
        stopStreamLoops();
        startStreamTransport();
      }
    });
    R.$("safeMode").addEventListener("change", function () {
      const on = Boolean(R.$("safeMode").checked);
      if (on) {
        state.pollMs = 6000;
        state.streamProfile = "stable";
        R.$("streamProfile").value = "stable";
        R.showToast(t("stream.safe_mode_on"), "warn");
      } else {
        state.pollMs = 3500;
        state.streamProfile = "auto";
        R.$("streamProfile").value = "auto";
        R.showToast(t("stream.safe_mode_off"), "ok");
      }
      if (state.streaming) {
        stopStreamLoops();
        startStreamTransport();
      }
      startPolling();
    });
    R.$("trendRange").addEventListener("change", function () {
      refreshAll({ silent: true }).catch(function () {});
    });
    R.$("btnExportTrends").addEventListener("click", exportTrendCsv);
    document.addEventListener("stg:lang-changed", function () {
      R.$("langSelect").value = T.getLang();
      setStreamHint();
      Array.prototype.forEach.call(document.querySelectorAll("button"), function (btn) {
        delete btn.dataset.idleText;
      });
    });
  }

  function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(function () {
      refreshAll({ silent: true }).catch(function (err) {
        R.setError(err);
        pushFeed("sistema", t("sync.error", { error: err.message }));
        R.showToast(t("sync.error_toast"), "warn");
      });
    }, state.pollMs);
  }

  async function boot() {
    T.init();
    R.$("langSelect").value = T.getLang();
    state.streamProfile = R.$("streamProfile").value || "auto";
    bindUi();
    setStreamHint();
    setHud();
    R.setCameraLoading(true, t("loading.camera"));
    updateStream(true);
    await refreshAll({ silent: false });
    pushFeed("sistema", t("boot.started"));
    R.showToast(t("boot.started_toast"), "ok", 1700);

    startPolling();
  }

  boot().catch(function (err) {
    R.setError(err);
    pushFeed("sistema", t("boot.fail", { error: err.message }));
    R.$("rawJson").textContent = String(err);
  });
})();
