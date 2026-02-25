// src/inference_client.cpp
#include "inference_client.h"
#include "config.h"
#include "logger.h"
#include "metrics.h"
#include "storage.h"
#include "networking.h"
#include "sensors.h"
#include "irrigation.h"

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <esp_camera.h>
#include <esp_task_wdt.h>
#include <time.h>
#include <cstring>

namespace
{
  String g_lastJson = "{}";
  volatile bool g_runRequested = false;
  uint32_t g_nextRunMs = 0;

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

  bool doInferenceOnce()
  {
    Metrics::incInferAttempt();

    if (WiFi.status() != WL_CONNECTED)
    {
      Metrics::incInferFail();
      setLastJson(false, -1, 0, "", 0.0f, false, "wifi_disconnected", "{\"err\":\"wifi_disconnected\"}");
      writeCsvLine(false, -1, 0, "", 0.0f, false, "wifi_disconnected");
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
      Log::event("infer_fail", {LS("err", "infer_host_empty")});
      return false;
    }

    // gate por lux (se habilitado)
    if (cfg.infer_min_lux_raw > 0 && sn.lux_raw < cfg.infer_min_lux_raw && !cfg.infer_use_led)
    {
      Metrics::incInferFail();
      setLastJson(false, -20, 0, "", 0.0f, false, "low_lux_skip", "{\"err\":\"low_lux_skip\"}");
      writeCsvLine(false, -20, 0, "", 0.0f, false, "low_lux_skip");
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

      wdtKick();
      camera_fb_t *fb = esp_camera_fb_get();

      if (!keepLed)
        ledRestoreAfterCapture();

      if (!fb || !fb->buf || fb->len == 0)
      {
        if (fb)
          esp_camera_fb_return(fb);
        Metrics::incInferFail();
        setLastJson(false, -2, 0, "", 0.0f, false, "camera_fb_get_fail", "{\"err\":\"camera_fb_get_fail\"}");
        writeCsvLine(false, -2, 0, "", 0.0f, false, "camera_fb_get_fail");
        Log::event("infer_fail", {LS("err", "camera_fb_get_fail")});
        return false;
      }

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
      esp_camera_fb_return(fb);

      uint32_t latencyMs = millis() - t0;

      ParsedResp parsed;
      bool parsedOk = resp.length() > 0 ? parseApiResponse(resp, parsed) : false;

      lastParsed = parsed;

      if (code >= 200 && code < 300 && resp.length() > 0)
      {
        if (parsedOk && parsed.confident)
        {
          Metrics::incInferOk();
          setLastJson(true, code, latencyMs, parsed.predicted, parsed.score, true, parsed.reasons, resp);
          writeCsvLine(true, code, latencyMs, parsed.predicted, parsed.score, true, parsed.reasons);
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
        setLastJson(true, code, latencyMs, parsed.predicted, parsed.score, false, parsed.reasons, resp);
        writeCsvLine(true, code, latencyMs, parsed.predicted, parsed.score, false, parsed.reasons);
        Log::event("infer_not_confident", {LI("code", code), LI("lat_ms", (int)latencyMs)}, {LS("reasons", parsed.reasons.c_str())});
        return true;
      }

      Metrics::incInferFail();
      String errRaw = resp.length() ? resp : (String("{\"err\":\"http_fail\",\"code\":") + String(code) + "}");
      setLastJson(false, code, latencyMs, parsed.predicted, parsed.score, false, "http_fail", errRaw);
      writeCsvLine(false, code, latencyMs, parsed.predicted, parsed.score, false, "http_fail");
      Log::event("infer_fail", {LI("code", code), LI("lat_ms", (int)latencyMs)});
      return false;
    }

    Metrics::incInferFail();
    setLastJson(false, -99, 0, "", 0.0f, false, "unknown", "{\"err\":\"unknown\"}");
    writeCsvLine(false, -99, 0, "", 0.0f, false, "unknown");
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
}

String InferenceClient::lastJson()
{
  return g_lastJson;
}
