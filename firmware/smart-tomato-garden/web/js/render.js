(function () {
  function $(id) { return document.getElementById(id); }
  function t(key, params) { return window.STGI18n.t(key, params); }

  function setBadge(elId, text, cls) {
    const node = $(elId);
    node.textContent = text;
    node.className = ("status " + (cls || "")).trim();
  }

  function gaugeColor(pct) {
    if (pct < 33) return "#c9743f";
    if (pct < 66) return "#c3a53c";
    return "#58b576";
  }

  function setGauge(id, pct) {
    const node = $(id);
    const safe = Math.max(0, Math.min(100, Number(pct) || 0));
    const color = gaugeColor(safe);
    node.style.background = "conic-gradient(" + color + " " + (safe * 3.6) + "deg, #e7efe4 0deg)";
  }

  function drawSparkline(canvasId, values, color) {
    const cvs = $(canvasId);
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    const w = cvs.width;
    const h = cvs.height;
    ctx.clearRect(0, 0, w, h);

    const clean = (values || []).map(function (v) { return Number(v); }).filter(function (n) { return Number.isFinite(n); });
    if (clean.length < 2) {
      ctx.strokeStyle = "rgba(90,120,100,.45)";
      ctx.beginPath();
      ctx.moveTo(0, h * .6);
      ctx.lineTo(w, h * .6);
      ctx.stroke();
      return;
    }

    let min = Math.min.apply(null, clean);
    let max = Math.max.apply(null, clean);
    if (max - min < 0.0001) max = min + 1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < clean.length; i++) {
      const x = (i / (clean.length - 1)) * (w - 1);
      const norm = (clean[i] - min) / (max - min);
      const y = h - 12 - norm * (h - 24);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function setSyncState(text, cls) {
    const node = $("syncState");
    node.textContent = text;
    node.className = ("tiny syncState " + (cls || "")).trim();
  }

  function setCameraLoading(on, text) {
    const overlay = $("cameraOverlay");
    overlay.classList.toggle("loading", Boolean(on));
    if (text) {
      const span = overlay.querySelector("span:last-child");
      if (span) span.textContent = text;
    }
  }

  function showToast(text, kind, ttlMs) {
    const stack = $("toastStack");
    if (!stack) return;
    const el = document.createElement("div");
    el.className = ("toast " + (kind || "")).trim();
    el.textContent = text;
    stack.appendChild(el);
    setTimeout(function () {
      if (el && el.parentNode) el.parentNode.removeChild(el);
    }, ttlMs || 2400);
  }

  function setSystem(payload) {
    const F = window.STGFmt;
    const health = payload.health;
    const sensors = payload.sensors;
    const irrigation = payload.irrigation;
    const camera = payload.camera;
    const metrics = payload.metrics;
    const config = payload.config;

    $("device").textContent = health.device_id || "-";
    $("ip").textContent = health.ip || "-";
    $("rssi").textContent = health.rssi + " dBm";
    $("uptime").textContent = F.fmtMs(health.uptime_ms);
    $("heap").textContent = String(health.heap || "-");
    $("psram").textContent = String(health.psram || "-");
    $("inferOk").textContent = String(metrics.infer_ok || 0);

    setBadge("onlineState", health.online ? t("status.online") : t("status.offline"), health.online ? "ok" : "bad");
    if (!health.online) setSyncState(t("status.offline"), "bad");

    const age = Number(sensors.age_ms || 0);
    if (age <= 4000) setBadge("sensorFresh", t("sensor.updated_ago", { age: F.fmtMs(age) }), "ok");
    else if (age <= 10000) setBadge("sensorFresh", t("sensor.delayed", { age: F.fmtMs(age) }), "warn");
    else setBadge("sensorFresh", t("sensor.stale", { age: F.fmtMs(age) }), "bad");

    $("soilRaw").textContent = String(sensors.soil_raw ?? "-");
    $("soilPct").textContent = F.fmtPct(sensors.soil_pct, 1);
    $("luxRaw").textContent = String(sensors.lux_raw ?? "-");
    $("tempC").textContent = F.fmtNum(sensors.temp_c, 1) + " " + t("unit.celsius");
    $("humPct").textContent = F.fmtPct(sensors.hum_pct, 1);
    $("dhtOk").textContent = sensors.dht_ok ? t("status.ok") : t("status.fail");

    $("autoMode").textContent = irrigation.auto_enabled ? t("state.active") : t("state.disabled");
    $("pumpRemaining").textContent = F.fmtMs(irrigation.remaining_ms || 0);
    $("cooldownRemaining").textContent = F.fmtMs(irrigation.cooldown_remaining_ms || 0);
    $("dryThreshold").textContent = (irrigation.soil_dry_threshold_pct ?? config.soil_dry_threshold_pct) + "%";

    if (irrigation.pump_on) setBadge("pumpState", t("status.pump_on"), "ok");
    else if ((irrigation.cooldown_remaining_ms || 0) > 0) setBadge("pumpState", t("status.cooldown_wait"), "warn");
    else setBadge("pumpState", t("status.pump_off"), "");

    $("quality").value = camera.quality;
    $("qualityV").textContent = String(camera.quality);
    $("framesize").value = camera.framesize;
    $("led").value = camera.led_intensity;
    $("ledV").textContent = String(camera.led_intensity);

    $("streamClients").textContent = String(metrics.stream_clients || 0);
    $("captureCount").textContent = String(metrics.capture || 0);
    $("httpCount").textContent = String(metrics.http || 0);
    $("mqttPub").textContent = String(metrics.mqtt_pub || 0);
    $("mqttFail").textContent = String(metrics.mqtt_fail || 0);
    $("lastSync").textContent = t("status.synced_at", { time: new Date().toLocaleTimeString() });
  }

  function setInference(vm, rawLastInfer) {
    const F = window.STGFmt;
    const diagCard = $("diagnosisCard");
    const main = $("diagnosisMain");
    main.classList.remove("loading");
    diagCard.classList.remove("ok", "warn", "bad");
    $("btnRetrySync").classList.add("hidden");

    let cls = "";
    if (vm.status === "ok") cls = "ok";
    if (vm.status === "low_confidence") cls = "warn";
    if (vm.status === "fail") cls = "bad";
    if (cls) diagCard.classList.add(cls);

    if (vm.status === "ok") setBadge("diagnosisStatus", vm.confidenceBadge, "ok");
    else if (vm.status === "low_confidence") setBadge("diagnosisStatus", vm.confidenceBadge, "warn");
    else if (vm.status === "fail") setBadge("diagnosisStatus", vm.confidenceBadge, "bad");
    else setBadge("diagnosisStatus", vm.confidenceBadge, "");

    $("diagLabel").textContent = vm.labelFriendly || t("diagnosis.no_recent");
    if (vm.confidencePct != null) {
      $("diagSub").textContent = t("diagnosis.confidence", { value: F.fmtNum(vm.confidencePct, 1) });
    } else if (vm.reasons.length) {
      $("diagSub").textContent = t("diagnosis.reasons", { value: vm.reasons.join(", ") });
    } else {
      $("diagSub").textContent = t("diagnosis.awaiting");
    }
    $("diagMeta").textContent = t("diagnosis.meta_line", {
      latency: F.fmtMs(vm.latencyMs),
      http: Number.isFinite(vm.httpStatus) ? vm.httpStatus : "-",
      time: F.fmtClock(vm.tsMs)
    });

    const topkList = $("topkList");
    topkList.innerHTML = "";
    if (!vm.topk.length) {
      topkList.innerHTML = '<p class="tiny">' + t("state.no_data_show") + "</p>";
    } else {
      vm.topk.forEach(function (row) {
        const pctText = row.scorePct == null ? "-" : F.fmtNum(row.scorePct, 1) + "%";
        const pctWidth = row.scorePct == null ? 0 : row.scorePct;
        const el = document.createElement("div");
        el.innerHTML =
          '<div class="barRow"><span>' + row.labelFriendly + '</span><strong>' + pctText + '</strong></div>' +
          '<div class="bar"><i style="width:' + pctWidth + '%"></i></div>';
        topkList.appendChild(el);
      });
    }

    const chips = $("contextChips");
    chips.innerHTML = "";
    const ctx = vm.context || {};
    const items = [
      t("chip.temp", { value: Number.isFinite(ctx.tempC) ? F.fmtNum(ctx.tempC, 1) + " " + t("unit.celsius") : "-" }),
      t("chip.air_humidity", { value: Number.isFinite(ctx.humPct) ? F.fmtNum(ctx.humPct, 1) + "%" : "-" }),
      t("chip.light", { value: Number.isFinite(ctx.luxRaw) ? String(ctx.luxRaw) : "-" }),
      t("chip.soil", { value: Number.isFinite(ctx.soilPct) ? F.fmtNum(ctx.soilPct, 1) + "%" : "-" }),
      t("chip.pump", { value: ctx.pumpOn == null ? "-" : (ctx.pumpOn ? t("pump.on") : t("pump.off")) })
    ];
    items.forEach(function (txt) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = txt;
      chips.appendChild(chip);
    });

    $("inferenceJson").textContent = JSON.stringify(rawLastInfer || {}, null, 2);
  }

  function setRaw(payload) {
    $("rawJson").textContent = JSON.stringify(payload, null, 2);
  }

  function setGuidedAlert(vm) {
    const t = window.STGI18n.t;
    const box = $("guidedAlert");
    if (!box) return;
    box.classList.add("hidden");

    if (!vm) return;
    if (vm.status === "fail") {
      box.textContent = t("guided.infer_fail");
      box.classList.remove("hidden");
      return;
    }
    if (vm.status === "low_confidence" && vm.reasons && vm.reasons.length) {
      box.textContent = t("guided.low_conf", { reasons: vm.reasons.join(", ") });
      box.classList.remove("hidden");
      return;
    }
  }

  function setDashboard(payload, vm, hist) {
    const F = window.STGFmt;
    const t = window.STGI18n.t;
    const sensors = payload.sensors || {};
    const metrics = payload.metrics || {};

    const soilPct = Number(sensors.soil_pct);
    const humPct = Number(sensors.hum_pct);
    const tempC = Number(sensors.temp_c);

    setGauge("gaugeSoil", soilPct);
    setGauge("gaugeHum", humPct);
    // temp mapped to 10..40 C for a horticulture-friendly visual scale
    const tempNorm = ((tempC - 10) / 30) * 100;
    setGauge("gaugeTemp", tempNorm);

    $("gaugeSoilV").textContent = Number.isFinite(soilPct) ? F.fmtNum(soilPct, 1) + "%" : "-";
    $("gaugeHumV").textContent = Number.isFinite(humPct) ? F.fmtNum(humPct, 1) + "%" : "-";
    $("gaugeTempV").textContent = Number.isFinite(tempC) ? F.fmtNum(tempC, 1) + " " + t("unit.celsius") : "-";

    drawSparkline("trendTemp", hist.temp, "#4eaa68");
    drawSparkline("trendSoil", hist.soil, "#2f8f48");
    const cleanTemp = (hist.temp || []).map(function (n) { return Number(n); }).filter(function (n) { return Number.isFinite(n); });
    const cleanSoil = (hist.soil || []).map(function (n) { return Number(n); }).filter(function (n) { return Number.isFinite(n); });
    if (cleanTemp.length) {
      const tMin = Math.min.apply(null, cleanTemp);
      const tMax = Math.max.apply(null, cleanTemp);
      $("trendTempMeta").textContent = F.fmtNum(tMin, 1) + " - " + F.fmtNum(tMax, 1) + " " + t("unit.celsius");
    } else {
      $("trendTempMeta").textContent = "-";
    }
    if (cleanSoil.length) {
      const sMin = Math.min.apply(null, cleanSoil);
      const sMax = Math.max.apply(null, cleanSoil);
      $("trendSoilMeta").textContent = F.fmtNum(sMin, 1) + "% - " + F.fmtNum(sMax, 1) + "%";
    } else {
      $("trendSoilMeta").textContent = "-";
    }

    const attempts = Number(metrics.infer_attempt || 0);
    const ok = Number(metrics.infer_ok || 0);
    const fail = Number(metrics.infer_fail || 0);
    const rate = attempts > 0 ? (ok / attempts) * 100 : 0;
    const ring = $("inferRing");
    ring.style.background = "conic-gradient(#6fc082 " + (rate * 3.6) + "deg, #e7efe4 0deg)";
    $("inferRate").textContent = attempts > 0 ? F.fmtNum(rate, 1) + "%" : "-";
    $("inferAttempt").textContent = String(attempts);
    $("inferFail").textContent = String(fail);
    $("inferLatency").textContent = Number.isFinite(vm.latencyMs) ? F.fmtMs(vm.latencyMs) : "-";

    const timeline = $("pumpTimeline");
    timeline.innerHTML = "";
    const pumpHist = hist.pump.slice(-30);
    for (let i = 0; i < 30; i++) {
      const v = pumpHist[i] || 0;
      const bar = document.createElement("i");
      if (v) bar.className = "on";
      timeline.appendChild(bar);
    }
    const onCount = pumpHist.reduce(function (acc, n) { return acc + (n ? 1 : 0); }, 0);
    const duty = pumpHist.length ? (onCount / pumpHist.length) * 100 : 0;
    $("pumpDuty").textContent = F.fmtNum(duty, 1) + "%";
  }

  function setError(err) {
    setBadge("onlineState", t("status.sync_fail"), "bad");
    setSyncState(t("status.no_connection"), "bad");
    setBadge("diagnosisStatus", t("diagnosis.error_connection"), "bad");
    $("diagLabel").textContent = t("diagnosis.load_failed");
    $("diagSub").textContent = err && err.message ? err.message : "erro desconhecido";
    $("btnRetrySync").classList.remove("hidden");
    $("diagnosisCard").classList.remove("ok", "warn");
    $("diagnosisCard").classList.add("bad");
    $("diagnosisMain").classList.remove("loading");
    setCameraLoading(false);
  }

  window.STGRender = {
    $,
    setBadge,
    setSyncState,
    setCameraLoading,
    showToast,
    setSystem,
    setInference,
    setRaw,
    setError
    ,
    setDashboard,
    setGuidedAlert
  };
})();
