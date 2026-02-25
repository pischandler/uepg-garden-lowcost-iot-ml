(function () {
  function fmtMs(ms) {
    const n = Number(ms || 0);
    if (!Number.isFinite(n)) return "-";
    if (n < 1000) return n + " ms";
    const s = Math.round(n / 100) / 10;
    if (s < 60) return s + "s";
    const m = Math.floor(s / 60);
    const rs = Math.round(s % 60);
    return m + "m " + rs + "s";
  }

  function fmtPct(v, digits) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits == null ? 1 : digits) + "%";
  }

  function toPctNum(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return Math.max(0, Math.min(100, n * 100));
  }

  function fmtNum(v, digits) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits == null ? 0 : digits);
  }

  function fmtClock(tsMs) {
    if (!Number.isFinite(Number(tsMs))) return "-";
    const msNow = Date.now();
    const uptimeNow = performance.now();
    const approxEpoch = msNow - uptimeNow + Number(tsMs);
    return new Date(approxEpoch).toLocaleTimeString();
  }

  window.STGFmt = { fmtMs, fmtPct, toPctNum, fmtNum, fmtClock };
})();
