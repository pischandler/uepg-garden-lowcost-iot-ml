(function () {
  function normalizeReasons(rawReasons) {
    if (Array.isArray(rawReasons)) return rawReasons;
    if (typeof rawReasons === "string" && rawReasons.trim()) {
      return rawReasons.split("|").map(function (s) { return s.trim(); }).filter(Boolean);
    }
    return [];
  }

  function parseRaw(raw) {
    if (!raw) return null;
    if (typeof raw === "object") return raw;
    if (typeof raw !== "string") return null;
    try {
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  function toViewModel(lastInfer) {
    const F = window.STGFmt;
    const M = window.STGMap;
    const t = window.STGI18n.t;
    const empty = {
      status: "empty",
      labelFriendly: t("diagnosis.no_recent"),
      labelRaw: "",
      confidencePct: null,
      confidenceBadge: t("diagnosis.no_data"),
      latencyMs: null,
      httpStatus: null,
      reasons: [],
      topk: [],
      context: { tempC: null, humPct: null, luxRaw: null, soilPct: null, pumpOn: null },
      rawPayload: null,
      tsMs: null
    };

    if (!lastInfer || typeof lastInfer !== "object" || Object.keys(lastInfer).length === 0) return empty;

    const rawPayload = parseRaw(lastInfer.raw);
    const predictedRaw =
      (rawPayload && (rawPayload.classe_predita || rawPayload.predicted || rawPayload.label || rawPayload.class)) ||
      lastInfer.predicted ||
      "";
    const confidenceRaw =
      (rawPayload && (rawPayload.score != null ? rawPayload.score : rawPayload.confidence)) != null
        ? (rawPayload.score != null ? rawPayload.score : rawPayload.confidence)
        : lastInfer.confidence;
    const confidencePct = F.toPctNum(confidenceRaw);
    const confident = Boolean((rawPayload && rawPayload.confident) != null ? rawPayload.confident : lastInfer.confident);
    const minConf = Number((rawPayload && rawPayload.min_confidence) != null ? rawPayload.min_confidence : 0.6);

    const reasonBase = normalizeReasons((rawPayload && rawPayload.reasons) != null ? rawPayload.reasons : lastInfer.reasons);
    const reasons = M.mapReasons(reasonBase);

    let status = "empty";
    let confidenceBadge = t("diagnosis.no_data");
    if (lastInfer.ok === false) {
      status = "fail";
      confidenceBadge = t("diagnosis.fail");
    } else if (predictedRaw || confidencePct != null) {
      if (confident && Number(confidenceRaw) >= minConf) {
        status = "ok";
        confidenceBadge = t("diagnosis.confident");
      } else {
        status = "low_confidence";
        confidenceBadge = t("diagnosis.low_confidence");
      }
    }

    const topkRaw = rawPayload && Array.isArray(rawPayload.topk) ? rawPayload.topk.slice(0, 3) : [];
    let topk = topkRaw.map(function (it) {
      const rawLabel = it.classe || it.class || it.label || "";
      const scorePct = F.toPctNum(it.score);
      return {
        labelFriendly: M.labelClass(rawLabel),
        labelRaw: rawLabel,
        scorePct: scorePct
      };
    });

    if (!topk.length && predictedRaw) {
      topk = [{
        labelFriendly: M.labelClass(predictedRaw),
        labelRaw: predictedRaw,
        scorePct: confidencePct
      }];
    }

    const meta = rawPayload && rawPayload.meta ? rawPayload.meta : {};
    const context = {
      tempC: meta.temp_c != null ? Number(meta.temp_c) : null,
      humPct: meta.hum_pct != null ? Number(meta.hum_pct) : null,
      luxRaw: meta.lux_raw != null ? Number(meta.lux_raw) : null,
      soilPct: meta.soil_pct != null ? Number(meta.soil_pct) : null,
      pumpOn: meta.pump_on != null ? Number(meta.pump_on) === 1 : null
    };

    return {
      status: status,
      labelFriendly: M.labelClass(predictedRaw),
      labelRaw: predictedRaw,
      confidencePct: confidencePct,
      confidenceBadge: confidenceBadge,
      latencyMs: Number(lastInfer.latency_ms),
      httpStatus: Number(lastInfer.http_status),
      reasons: reasons,
      topk: topk,
      context: context,
      rawPayload: rawPayload,
      tsMs: Number(lastInfer.ts_ms),
      ok: lastInfer.ok !== false
    };
  }

  window.STGInference = { toViewModel };
})();
