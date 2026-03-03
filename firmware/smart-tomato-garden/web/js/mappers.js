(function () {
  var CLASS_MAP = {
    Tomato_Leaf_Mold: "class.Tomato_Leaf_Mold",
    Tomato_Early_blight: "class.Tomato_Early_blight",
    Tomato__Target_Spot: "class.Tomato__Target_Spot",
    Tomato_Late_blight: "class.Tomato_Late_blight",
    Tomato_Septoria_leaf_spot: "class.Tomato_Septoria_leaf_spot",
    Tomato_Bacterial_spot: "class.Tomato_Bacterial_spot",
    Tomato_healthy: "class.Tomato_healthy"
  };

  var REASON_MAP = {
    low_light: "reason.low_light",
    blurry: "reason.blurry",
    low_lux_skip: "reason.low_lux_skip",
    infer_host_empty: "reason.infer_host_empty",
    camera_fb_get_fail: "reason.camera_fb_get_fail",
    http_fail: "reason.http_fail",
    wifi_disconnected: "reason.wifi_disconnected",
    unknown: "reason.unknown"
  };

  function init(schema) {
    if (schema && Array.isArray(schema.classes)) {
      var m = {};
      schema.classes.forEach(function (c) {
        m[c] = "class." + c;
      });
      CLASS_MAP = m;
    }
    if (schema && Array.isArray(schema.reasons)) {
      var m = {};
      schema.reasons.forEach(function (r) {
        m[r] = "reason." + r;
      });
      REASON_MAP = m;
    }
  }

  function labelClass(raw) {
    var t = window.STGI18n.t;
    if (!raw) return t("diagnosis.no_recent");
    var key = CLASS_MAP[raw];
    return key ? t(key) : raw;
  }

  function mapReasons(list) {
    var t = window.STGI18n.t;
    if (!Array.isArray(list)) return [];
    return list.filter(Boolean).map(function (r) {
      var key = REASON_MAP[r];
      return key ? t(key) : r;
    });
  }

  window.STGMap = { init, labelClass, mapReasons };
})();
