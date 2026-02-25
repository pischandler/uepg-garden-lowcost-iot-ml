(function () {
  async function jget(path) {
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) throw new Error(path + " " + r.status);
    return r.json();
  }

  async function jpost(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    if (!r.ok) throw new Error(path + " " + r.status);
    return r.json();
  }

  async function textget(path) {
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) throw new Error(path + " " + r.status);
    return r.text();
  }

  async function refreshPayload() {
    const paths = [
      jget("/health"),
      jget("/api/sensors"),
      jget("/api/irrigation"),
      jget("/status"),
      jget("/metrics"),
      jget("/api/config"),
      jget("/api/inference/last")
    ];
    const [health, sensors, irrigation, camera, metrics, config, lastInfer] = await Promise.all(paths);
    return { health, sensors, irrigation, camera, metrics, config, lastInfer };
  }

  window.STGApi = { jget, jpost, textget, refreshPayload };
})();
