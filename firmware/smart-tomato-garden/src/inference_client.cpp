// src/inference_client.cpp
#include "inference_client.h"
#include "config.h"
#include "logger.h"
#include "metrics.h"
#include "storage.h"
#include "networking.h"
#include "sensors.h"
#include "irrigation.h"
#include "camera_server.h"

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <esp_camera.h>
#include <esp_task_wdt.h>
#include <time.h>
#include <cstring>

namespace
{
  static constexpr uint8_t INFER_FB_GET_RETRIES = 3;
  static constexpr uint16_t INFER_FB_GET_RETRY_DELAY_MS = 45;
  static constexpr uint8_t INFER_FAIL_RING_SIZE = 16;
  static constexpr uint32_t INFER_BREAKER_BASE_MS = 2000;
  static constexpr uint32_t INFER_BREAKER_MAX_MS = 60000;

  String g_lastJson = "{}";
  volatile bool g_runRequested = false;
  uint32_t g_nextRunMs = 0;
  uint32_t g_lastRunRequestTsMs = 0;
  uint32_t g_lastCompletedTsMs = 0;
  bool g_running = false;
  bool g_lastRunOk = false;
  uint32_t g_consecutiveFails = 0;
  uint32_t g_breakerOpenUntilMs = 0;
  String g_modelVersion = "";
  uint8_t g_consecutiveCamFbFails = 0;

  struct InferDiagEntry
  {
    uint32_t ts_ms = 0;
    int http_status = 0;
    uint32_t latency_ms = 0;
    char reason[32] = {0};
    uint16_t lux_raw = 0;
    float soil_pct = 0.0f;
    float temp_c = 0.0f;
    float hum_pct = 0.0f;
    bool pump_on = false;
  };

  InferDiagEntry g_diagRing[INFER_FAIL_RING_SIZE];
  uint8_t g_diagWriteIdx = 0;
  uint8_t g_diagCount = 0;

  static inline void wdtKick()
  {
    esp_task_wdt_reset();
  }

  static inline uint8_t clampDuty(int v)
  {
    if (v < 0)
      return 0;
    if (v > (int)LED_FLASH_MAX_DUTY)
      return LED_FLASH_MAX_DUTY;
    return (uint8_t)v;
  }

  void ledSet(uint8_t duty)
  {
    if (LED_FLASH_GPIO < 0)
      return;
    ledcWrite(LED_FLASH_PWM_CH, clampDuty(duty));
  }

  void ledAssistOn(uint8_t duty)
  {
    if (LED_FLASH_GPIO < 0)
      return;
    ledSet(duty);
    delay(LED_FLASH_ON_DELAY_MS);
  }

  void ledRestoreAfterCapture()
  {
    auto cfg = ConfigStore::get();
    bool keep = cfg.led_on_stream && (Metrics::streamClients() > 0);
    if (keep)
      ledSet(cfg.led_duty);
    else
      ledSet(0);
  }

  String normalizePath(const char *p)
  {
    if (!p || !*p)
      return "/";
    if (p[0] == '/')
      return String(p);
    return String("/") + String(p);
  }

  String csvEscape(const String &s)
  {
    String out = s;
    out.replace("\"", "\"\"");
    return out;
  }

  void writeCsvLine(bool ok, int httpCode, uint32_t latencyMs,
                    const String &predicted, float confidence,
                    bool confident, const String &reasons)
  {
    time_t now = time(nullptr);
    uint32_t tsUnix = (now > 100000) ? (uint32_t)now : 0;
    uint32_t tsMs = (uint32_t)millis();

    SensorsSnapshot sn = Sensors::latest();
    IrrigationState ir = Irrigation::state();

    String line;
    line.reserve(360);

    line += String(tsUnix);
    line += ",";
    line += String(tsMs);
    line += ",";
    line += "\"";
    line += csvEscape(Networking::deviceId());
    line += "\"";
    line += ",";
    line += "\"";
    line += csvEscape(Networking::ip());
    line += "\"";
    line += ",";
    line += String(Networking::rssi());
    line += ",";
    line += ok ? "1" : "0";
    line += ",";
    line += String(httpCode);
    line += ",";
    line += String(latencyMs);
    line += ",";
    line += "\"";
    line += csvEscape(predicted);
    line += "\"";
    line += ",";
    line += String(confidence, 6);
    line += ",";
    line += confident ? "1" : "0";
    line += ",";
    line += "\"";
    line += csvEscape(reasons);
    line += "\"";
    line += ",";
    line += String(sn.soil_pct, 2);
    line += ",";
    line += String(sn.soil_raw);
    line += ",";
    line += String(sn.lux_raw);
    line += ",";
    line += String(sn.temp_c, 2);
    line += ",";
    line += String(sn.hum_pct, 2);
    line += ",";
    line += sn.dht_ok ? "1" : "0";
    line += ",";
    line += ir.pump_on ? "1" : "0";

    Storage::appendInferenceCsv(line.c_str());
  }

  void setLastJson(bool ok, int httpCode, uint32_t latencyMs,
                   const String &predicted, float confidence,
                   bool confident, const String &reasons,
                   const String &raw)
  {
    StaticJsonDocument<1400> doc;
    doc["ok"] = ok;
    doc["http_status"] = httpCode;
    doc["latency_ms"] = latencyMs;
    doc["ts_ms"] = (uint32_t)millis();

    doc["predicted"] = predicted;
    doc["confidence"] = confidence;

    doc["confident"] = confident;
    doc["reasons"] = reasons;
    doc["model_version"] = g_modelVersion;

    doc["raw"] = raw;

    String out;
    serializeJson(doc, out);
    g_lastJson = out;
  }

  struct ParsedResp
  {
    String predicted;
    float score = 0.0f;
    bool confident = false;
    String modelVersion;
    String reasons; // "low_light|blurry|..."
    bool hasLowLight = false;
    bool hasBlurry = false;
  };

  bool parseApiResponse(const String &raw, ParsedResp &out)
  {
    out = ParsedResp{};

    StaticJsonDocument<4096> r;
    auto err = deserializeJson(r, raw);
    if (err)
      return false;

    if (r["classe_predita"].is<const char *>())
      out.predicted = String((const char *)r["classe_predita"]);
    else if (r["predicted"].is<const char *>())
      out.predicted = String((const char *)r["predicted"]);
    else if (r["class"].is<const char *>())
      out.predicted = String((const char *)r["class"]);
    else if (r["label"].is<const char *>())
      out.predicted = String((const char *)r["label"]);
    else
      out.predicted = "";

    if (r["score"].is<float>() || r["score"].is<double>() || r["score"].is<int>())
      out.score = r["score"].as<float>();
    else if (r["confidence"].is<float>() || r["confidence"].is<double>() || r["confidence"].is<int>())
      out.score = r["confidence"].as<float>();
    else if (r["probability"].is<float>() || r["probability"].is<double>() || r["probability"].is<int>())
      out.score = r["probability"].as<float>();

    if (r["confident"].is<bool>())
      out.confident = r["confident"].as<bool>();
    else
      out.confident = (out.predicted.length() > 0);

    if (r["model_version"].is<const char *>())
      out.modelVersion = String((const char *)r["model_version"]);
    else if (r["model"].is<const char *>())
      out.modelVersion = String((const char *)r["model"]);

    // API pode mandar reasons como array OU string
    if (r["reasons"].is<JsonArray>())
    {
      JsonArray a = r["reasons"].as<JsonArray>();
      bool first = true;
      for (JsonVariant v : a)
      {
        const char *s = v.as<const char *>();
        if (!s || !*s)
          continue;
        if (!first)
          out.reasons += "|";
        out.reasons += s;
        first = false;

        if (strcmp(s, "low_light") == 0)
          out.hasLowLight = true;
        if (strcmp(s, "blurry") == 0)
          out.hasBlurry = true;
      }
    }
    else if (r["reasons"].is<const char *>())
    {
      const char *s = r["reasons"].as<const char *>();
      if (s && *s)
        out.reasons = s;
    }

    if (!out.confident)
      out.predicted = "";

    return true;
  }

  camera_fb_t *captureFrameForInference()
  {
    for (uint8_t i = 0; i <= INFER_FB_GET_RETRIES; i++)
    {
      wdtKick();
      camera_fb_t *fb = CameraAccess::fbGet(120);
      if (fb && fb->buf && fb->len > 0)
        return fb;

      CameraAccess::fbReturn(fb);

      if (i < INFER_FB_GET_RETRIES)
      {
        delay(INFER_FB_GET_RETRY_DELAY_MS);
        wdtKick();
      }
    }
    return nullptr;
  }

  void pushDiagFail(int httpCode, uint32_t latencyMs, const char *reason)
  {
    InferDiagEntry &d = g_diagRing[g_diagWriteIdx];
    g_diagWriteIdx = (uint8_t)((g_diagWriteIdx + 1) % INFER_FAIL_RING_SIZE);
    if (g_diagCount < INFER_FAIL_RING_SIZE)
      g_diagCount++;

    SensorsSnapshot sn = Sensors::latest();
    IrrigationState ir = Irrigation::state();

    d.ts_ms = (uint32_t)millis();
    d.http_status = httpCode;
    d.latency_ms = latencyMs;
    strncpy(d.reason, (reason && *reason) ? reason : "unknown", sizeof(d.reason) - 1);
    d.reason[sizeof(d.reason) - 1] = 0;
    d.lux_raw = sn.lux_raw;
    d.soil_pct = sn.soil_pct;
    d.temp_c = sn.temp_c;
    d.hum_pct = sn.hum_pct;
    d.pump_on = ir.pump_on;
  }

  void registerFailureForBreaker(bool endpointFailure)
  {
    if (!endpointFailure)
      return;
    g_consecutiveFails++;
    uint32_t shift = g_consecutiveFails > 5 ? 5 : g_consecutiveFails;
    uint32_t cool = INFER_BREAKER_BASE_MS << shift;
    if (cool > INFER_BREAKER_MAX_MS)
      cool = INFER_BREAKER_MAX_MS;
    g_breakerOpenUntilMs = (uint32_t)millis() + cool;
  }

  void resetBreaker()
  {
    g_consecutiveFails = 0;
    g_breakerOpenUntilMs = 0;
  }

  void maybeAdaptiveCameraDegrade()
  {
    if (g_consecutiveCamFbFails < 3)
      return;

    g_consecutiveCamFbFails = 0;

    auto cfg = ConfigStore::get();
    uint8_t newFs = cfg.cam_framesize;
    uint8_t newQ = cfg.cam_quality;

    if (newFs > 5)
      newFs = (uint8_t)(newFs - 1);
    if (newQ < 58)
      newQ = (uint8_t)(newQ + 5);

    StaticJsonDocument<128> doc;
    doc["cam_framesize"] = newFs;
    doc["cam_quality"] = newQ;
    String js;
    serializeJson(doc, js);
    ConfigStore::setPartialJson(js.c_str());

    sensor_t *s = esp_camera_sensor_get();
    if (s)
    {
      s->set_framesize(s, (framesize_t)newFs);
      s->set_quality(s, newQ);
    }

    Log::event("infer_cam_adapt", {LI("fs", newFs), LI("q", newQ)});
  }

  bool doInferenceOnce()
  {
    g_running = true;
    g_lastRunOk = false;

    Metrics::incInferAttempt();

    if (g_breakerOpenUntilMs > (uint32_t)millis())
    {
      Metrics::incInferFail();
      setLastJson(false, -31, 0, "", 0.0f, false, "breaker_open", "{\"err\":\"breaker_open\"}");
      writeCsvLine(false, -31, 0, "", 0.0f, false, "breaker_open");
      pushDiagFail(-31, 0, "breaker_open");
      g_running = false;
      g_lastCompletedTsMs = (uint32_t)millis();
      return false;
    }

    if (WiFi.status() != WL_CONNECTED)
    {
      Metrics::incInferFail();
      setLastJson(false, -1, 0, "", 0.0f, false, "wifi_disconnected", "{\"err\":\"wifi_disconnected\"}");
      writeCsvLine(false, -1, 0, "", 0.0f, false, "wifi_disconnected");
      pushDiagFail(-1, 0, "wifi_disconnected");
      registerFailureForBreaker(true);
      g_running = false;
      g_lastCompletedTsMs = (uint32_t)millis();
      Log::event("infer_fail", {LS("err", "wifi_disconnected")});
      return false;
    }

    auto cfg = ConfigStore::get();
    SensorsSnapshot sn = Sensors::latest();

    // valida host
    if (strlen(cfg.infer_host) == 0)
    {
      Metrics::incInferFail();
      setLastJson(false, -10, 0, "", 0.0f, false, "infer_host_empty", "{\"err\":\"infer_host_empty\"}");
      writeCsvLine(false, -10, 0, "", 0.0f, false, "infer_host_empty");
      pushDiagFail(-10, 0, "infer_host_empty");
      registerFailureForBreaker(false);
      g_running = false;
      g_lastCompletedTsMs = (uint32_t)millis();
      Log::event("infer_fail", {LS("err", "infer_host_empty")});
      return false;
    }

    // gate por lux (se habilitado)
    if (cfg.infer_min_lux_raw > 0 && sn.lux_raw < cfg.infer_min_lux_raw && !cfg.infer_use_led)
    {
      Metrics::incInferFail();
      setLastJson(false, -20, 0, "", 0.0f, false, "low_lux_skip", "{\"err\":\"low_lux_skip\"}");
      writeCsvLine(false, -20, 0, "", 0.0f, false, "low_lux_skip");
      pushDiagFail(-20, 0, "low_lux_skip");
      registerFailureForBreaker(false);
      g_running = false;
      g_lastCompletedTsMs = (uint32_t)millis();
      Log::event("infer_skip", {LS("err", "low_lux_skip")});
      return false;
    }

    String host = String(cfg.infer_host);
    uint16_t port = (uint16_t)cfg.infer_port;
    String path = normalizePath(cfg.infer_path);
    String url = String("http://") + host + ":" + String(port) + path;

    ParsedResp lastParsed;

    uint8_t maxRetries = cfg.infer_max_retries;
    for (uint8_t attempt = 0; attempt <= maxRetries; attempt++)
    {
      wdtKick();

      bool forceLed = false;

      if (attempt == 0)
      {
        if (cfg.infer_use_led && cfg.infer_min_lux_raw > 0 && sn.lux_raw < cfg.infer_min_lux_raw)
          forceLed = true;
      }
      else
      {
        if (cfg.infer_use_led && (lastParsed.hasLowLight || lastParsed.hasBlurry))
          forceLed = true;

        if (cfg.infer_retry_delay_ms > 0)
        {
          delay(cfg.infer_retry_delay_ms);
          wdtKick();
        }
      }

      bool keepLed = cfg.led_on_stream && (Metrics::streamClients() > 0);
      if (forceLed && !keepLed && LED_FLASH_GPIO >= 0)
        ledAssistOn(cfg.led_duty);

      camera_fb_t *fb = captureFrameForInference();

      if (!keepLed)
        ledRestoreAfterCapture();

      if (!fb)
      {
        bool canRetryCapture = (attempt < maxRetries);
        if (canRetryCapture)
        {
          g_consecutiveCamFbFails++;
          maybeAdaptiveCameraDegrade();
          Log::event("infer_retry", {LI("attempt", (int)attempt)}, {LS("reasons", "camera_fb_get_fail")});
          continue;
        }

        Metrics::incInferFail();
        setLastJson(false, -2, 0, "", 0.0f, false, "camera_fb_get_fail", "{\"err\":\"camera_fb_get_fail\"}");
        writeCsvLine(false, -2, 0, "", 0.0f, false, "camera_fb_get_fail");
        pushDiagFail(-2, 0, "camera_fb_get_fail");
        g_consecutiveCamFbFails++;
        maybeAdaptiveCameraDegrade();
        registerFailureForBreaker(true);
        g_running = false;
        g_lastCompletedTsMs = (uint32_t)millis();
        Log::event("infer_fail", {LS("err", "camera_fb_get_fail")});
        return false;
      }
      g_consecutiveCamFbFails = 0;

      uint32_t t0 = millis();

      HTTPClient http;
      http.begin(url);
      http.setTimeout(INFER_HTTP_TIMEOUT_MS);
      http.addHeader("Content-Type", "image/jpeg");

      // headers de telemetria (API pode salvar/debugar)
      http.addHeader("X-Device-Id", Networking::deviceId());
      http.addHeader("X-Lux-Raw", String(sn.lux_raw));
      http.addHeader("X-Soil-Raw", String(sn.soil_raw));
      http.addHeader("X-Soil-Pct", String(sn.soil_pct, 2));
      http.addHeader("X-Temp-C", String(sn.temp_c, 2));
      http.addHeader("X-Hum-Pct", String(sn.hum_pct, 2));
      http.addHeader("X-Pump-On", (Irrigation::state().pump_on ? "1" : "0"));

      wdtKick();
      int code = http.POST(fb->buf, fb->len);
      wdtKick();

      String resp = (code > 0) ? http.getString() : String("");

      http.end();
      CameraAccess::fbReturn(fb);

      uint32_t latencyMs = millis() - t0;

      ParsedResp parsed;
      bool parsedOk = resp.length() > 0 ? parseApiResponse(resp, parsed) : false;

      lastParsed = parsed;

      if (code >= 200 && code < 300 && resp.length() > 0)
      {
        if (parsedOk && parsed.confident)
        {
          Metrics::incInferOk();
          if (parsed.modelVersion.length())
            g_modelVersion = parsed.modelVersion;
          setLastJson(true, code, latencyMs, parsed.predicted, parsed.score, true, parsed.reasons, resp);
          writeCsvLine(true, code, latencyMs, parsed.predicted, parsed.score, true, parsed.reasons);
          resetBreaker();
          g_running = false;
          g_lastCompletedTsMs = (uint32_t)millis();
          g_lastRunOk = true;
          Log::event("infer_ok", {LI("code", code), LI("lat_ms", (int)latencyMs)});
          return true;
        }

        bool canRetry = (attempt < maxRetries) && parsedOk && (parsed.hasLowLight || parsed.hasBlurry);
        if (canRetry)
        {
          Log::event("infer_retry", {LI("attempt", (int)attempt), LI("code", code)}, {LS("reasons", parsed.reasons.c_str())});
          continue;
        }

        Metrics::incInferOk();
        if (parsed.modelVersion.length())
          g_modelVersion = parsed.modelVersion;
        setLastJson(true, code, latencyMs, parsed.predicted, parsed.score, false, parsed.reasons, resp);
        writeCsvLine(true, code, latencyMs, parsed.predicted, parsed.score, false, parsed.reasons);
        resetBreaker();
        g_running = false;
        g_lastCompletedTsMs = (uint32_t)millis();
        g_lastRunOk = true;
        Log::event("infer_not_confident", {LI("code", code), LI("lat_ms", (int)latencyMs)}, {LS("reasons", parsed.reasons.c_str())});
        return true;
      }

      Metrics::incInferFail();
      String errRaw = resp.length() ? resp : (String("{\"err\":\"http_fail\",\"code\":") + String(code) + "}");
      setLastJson(false, code, latencyMs, parsed.predicted, parsed.score, false, "http_fail", errRaw);
      writeCsvLine(false, code, latencyMs, parsed.predicted, parsed.score, false, "http_fail");
      pushDiagFail(code, latencyMs, "http_fail");
      registerFailureForBreaker(true);
      g_running = false;
      g_lastCompletedTsMs = (uint32_t)millis();
      Log::event("infer_fail", {LI("code", code), LI("lat_ms", (int)latencyMs)});
      return false;
    }

    Metrics::incInferFail();
    setLastJson(false, -99, 0, "", 0.0f, false, "unknown", "{\"err\":\"unknown\"}");
    writeCsvLine(false, -99, 0, "", 0.0f, false, "unknown");
    pushDiagFail(-99, 0, "unknown");
    registerFailureForBreaker(true);
    g_running = false;
    g_lastCompletedTsMs = (uint32_t)millis();
    return false;
  }
}

void InferenceClient::begin()
{
  g_lastJson = "{}";
  g_runRequested = false;

  auto cfg = ConfigStore::get();
  uint32_t now = millis();
  g_nextRunMs = now + (cfg.infer_period_ms ? cfg.infer_period_ms : 0);
}

void InferenceClient::loop()
{
  auto cfg = ConfigStore::get();
  if (!cfg.infer_enabled)
    return;

  uint32_t now = millis();

  if (!g_runRequested && cfg.infer_skip_when_streaming && Metrics::streamClients() > 0)
    return;

  bool timeDue = (cfg.infer_period_ms > 0) && (now >= g_nextRunMs);

  if (g_runRequested || timeDue)
  {
    g_runRequested = false;
    if (cfg.infer_period_ms > 0)
      g_nextRunMs = now + cfg.infer_period_ms;

    doInferenceOnce();
  }
}

void InferenceClient::requestRun()
{
  g_runRequested = true;
  g_lastRunRequestTsMs = (uint32_t)millis();
}

String InferenceClient::lastJson()
{
  return g_lastJson;
}

String InferenceClient::statusJson()
{
  StaticJsonDocument<384> doc;
  doc["api_version"] = "1.2.0";
  doc["queued"] = g_runRequested;
  doc["running"] = g_running;
  doc["last_ok"] = g_lastRunOk;
  doc["last_request_ts_ms"] = g_lastRunRequestTsMs;
  doc["last_completed_ts_ms"] = g_lastCompletedTsMs;
  doc["breaker_open"] = g_breakerOpenUntilMs > (uint32_t)millis();
  doc["breaker_open_until_ms"] = g_breakerOpenUntilMs;
  doc["consecutive_fails"] = g_consecutiveFails;
  doc["model_version"] = g_modelVersion;
  String out;
  serializeJson(doc, out);
  return out;
}

String InferenceClient::diagnosticsJson()
{
  StaticJsonDocument<2048> doc;
  doc["api_version"] = "1.2.0";
  doc["count"] = g_diagCount;
  JsonArray arr = doc.createNestedArray("items");
  for (uint8_t i = 0; i < g_diagCount; i++)
  {
    uint8_t pos = (uint8_t)((g_diagWriteIdx + INFER_FAIL_RING_SIZE - g_diagCount + i) % INFER_FAIL_RING_SIZE);
    const InferDiagEntry &d = g_diagRing[pos];
    JsonObject o = arr.createNestedObject();
    o["ts_ms"] = d.ts_ms;
    o["http_status"] = d.http_status;
    o["latency_ms"] = d.latency_ms;
    o["reason"] = d.reason;
    o["lux_raw"] = d.lux_raw;
    o["soil_pct"] = d.soil_pct;
    o["temp_c"] = d.temp_c;
    o["hum_pct"] = d.hum_pct;
    o["pump_on"] = d.pump_on;
  }
  String out;
  serializeJson(doc, out);
  return out;
}
