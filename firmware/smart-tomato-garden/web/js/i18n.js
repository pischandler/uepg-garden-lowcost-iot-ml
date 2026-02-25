(function () {
  const dicts = window.__STG_I18N_DICTS__ || {};
  let lang = "pt";

  function interpolate(text, params) {
    if (!params) return text;
    return String(text).replace(/\{(\w+)\}/g, function (_, k) {
      return params[k] != null ? String(params[k]) : "";
    });
  }

  function t(key, params) {
    const active = dicts[lang] || {};
    const fallback = dicts.pt || {};
    const raw = active[key] != null ? active[key] : (fallback[key] != null ? fallback[key] : key);
    return interpolate(raw, params);
  }

  function apply(root) {
    const node = root || document;
    const list = node.querySelectorAll("[data-i18n]");
    list.forEach(function (el) {
      const key = el.getAttribute("data-i18n");
      if (!key) return;
      el.textContent = t(key);
    });
  }

  function setLang(next) {
    if (!dicts[next]) return;
    lang = next;
    try { localStorage.setItem("stg_lang", lang); } catch (e) {}
    apply(document);
    document.documentElement.setAttribute("lang", next === "cn" ? "zh-CN" : next);
    document.dispatchEvent(new CustomEvent("stg:lang-changed", { detail: { lang: lang } }));
  }

  function getLang() {
    return lang;
  }

  function init() {
    let preferred = "pt";
    try {
      preferred = localStorage.getItem("stg_lang") || "pt";
    } catch (e) {}
    if (!dicts[preferred]) preferred = "pt";
    lang = preferred;
    apply(document);
  }

  window.STGI18n = { init, t, apply, setLang, getLang, dicts };
})();
