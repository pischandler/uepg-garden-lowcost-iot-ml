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
#include <time.h>

namespace
{
  String g_lastJson = "{}";
  volatile bool g_runRequested = false;
  uint32_t g_nextRunMs = 0;

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

  void writeCsvLine(bool ok, int httpCode, uint32_t latencyMs, const String &predicted, float confidence)
  {
    time_t now = time(nullptr);
    uint32_t tsUnix = (now > 100000) ? (uint32_t)now : 0;
    uint32_t tsMs = (uint32_t)millis();

    SensorsSnapshot sn = Sensors::latest();
    IrrigationState ir = Irrigation::state();

    String line;
    line.reserve(256);

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

  void setLastJson(bool ok, int httpCode, uint32_t latencyMs, const String &predicted, float confidence, const String &raw)
  {
    StaticJsonDocument<1024> doc;
    doc["ok"] = ok;
    doc["http_status"] = httpCode;
    doc["latency_ms"] = latencyMs;
    doc["ts_ms"] = (uint32_t)millis();
    doc["predicted"] = predicted;
    doc["confidence"] = confidence;
    doc["raw"] = raw;
    String out;
    serializeJson(doc, out);
    g_lastJson = out;
  }

  void extractPrediction(const String &raw, String &predicted, float &confidence)
  {
    predicted = "";
    confidence = 0.0f;

    StaticJsonDocument<1024> r;
    auto err = deserializeJson(r, raw);
    if (err)
      return;

    if (r["predicted"].is<const char *>())
      predicted = String((const char *)r["predicted"]);
    else if (r["class"].is<const char *>())
      predicted = String((const char *)r["class"]);
    else if (r["label"].is<const char *>())
      predicted = String((const char *)r["label"]);

    if (r["confidence"].is<float>() || r["confidence"].is<double>() || r["confidence"].is<int>())
      confidence = r["confidence"].as<float>();
    else if (r["probability"].is<float>() || r["probability"].is<double>() || r["probability"].is<int>())
      confidence = r["probability"].as<float>();
    else if (r["score"].is<float>() || r["score"].is<double>() || r["score"].is<int>())
      confidence = r["score"].as<float>();
  }

  bool doInferenceOnce()
  {
    Metrics::incInferAttempt();

    if (WiFi.status() != WL_CONNECTED)
    {
      Metrics::incInferFail();
      setLastJson(false, -1, 0, "", 0.0f, "{\"err\":\"wifi_disconnected\"}");
      writeCsvLine(false, -1, 0, "", 0.0f);
      Log::event("infer_fail", {LS("err", "wifi_disconnected")});
      return false;
    }

    auto cfg = ConfigStore::get();
    String host = String(cfg.infer_host);
    uint16_t port = (uint16_t)cfg.infer_port;
    String path = normalizePath(cfg.infer_path);
    String url = String("http://") + host + ":" + String(port) + path;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb || !fb->buf || fb->len == 0)
    {
      if (fb)
        esp_camera_fb_return(fb);
      Metrics::incInferFail();
      setLastJson(false, -2, 0, "", 0.0f, "{\"err\":\"camera_fb_get_fail\"}");
      writeCsvLine(false, -2, 0, "", 0.0f);
      Log::event("infer_fail", {LS("err", "camera_fb_get_fail")});
      return false;
    }

    uint32_t t0 = millis();

    HTTPClient http;
    http.begin(url);
    http.addHeader("Content-Type", "image/jpeg");
    http.setTimeout(INFER_HTTP_TIMEOUT_MS);

    int code = http.POST(fb->buf, fb->len);
    String resp = (code > 0) ? http.getString() : String("");

    http.end();
    esp_camera_fb_return(fb);

    uint32_t latencyMs = millis() - t0;

    String predicted;
    float confidence = 0.0f;
    if (resp.length())
      extractPrediction(resp, predicted, confidence);

    if (code >= 200 && code < 300 && resp.length() > 0)
    {
      Metrics::incInferOk();
      setLastJson(true, code, latencyMs, predicted, confidence, resp);
      writeCsvLine(true, code, latencyMs, predicted, confidence);
      Log::event("infer_ok", {LI("code", code), LI("lat_ms", (int)latencyMs)});
      return true;
    }

    Metrics::incInferFail();
    if (resp.length() == 0)
      resp = String("{\"err\":\"http_fail\",\"code\":") + String(code) + "}";

    setLastJson(false, code, latencyMs, predicted, confidence, resp);
    writeCsvLine(false, code, latencyMs, predicted, confidence);
    Log::event("infer_fail", {LI("code", code), LI("lat_ms", (int)latencyMs)});
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
