#include "camera_server.h"
#include "camera_pins.h"
#include "config.h"
#include "hash_dispatch.h"
#include "logger.h"
#include "metrics.h"
#include "networking.h"
#include "sensors.h"
#include "irrigation.h"
#include "camera_index.h"
#include "inference_client.h"
#include <esp_camera.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <LittleFS.h>
#include <cstring>

static AsyncWebServer server(HTTP_PORT);

static void ledInit()
{
  if (LED_FLASH_GPIO < 0)
    return;
  ledcSetup(LED_FLASH_PWM_CH, LED_FLASH_PWM_FREQ, LED_FLASH_PWM_BITS);
  ledcAttachPin(LED_FLASH_GPIO, LED_FLASH_PWM_CH);
  ledcWrite(LED_FLASH_PWM_CH, 0);
}

static void ledSetDuty(uint8_t duty)
{
  uint8_t d = duty > LED_FLASH_MAX_DUTY ? LED_FLASH_MAX_DUTY : duty;
  ledcWrite(LED_FLASH_PWM_CH, d);
}

static void ledOn(uint8_t duty) { ledSetDuty(duty); }
static void ledOff() { ledSetDuty(0); }

static bool cameraInit(uint8_t quality, uint8_t framesize)
{
  camera_config_t c;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = CAM_PIN_D0;
  c.pin_d1 = CAM_PIN_D1;
  c.pin_d2 = CAM_PIN_D2;
  c.pin_d3 = CAM_PIN_D3;
  c.pin_d4 = CAM_PIN_D4;
  c.pin_d5 = CAM_PIN_D5;
  c.pin_d6 = CAM_PIN_D6;
  c.pin_d7 = CAM_PIN_D7;
  c.pin_xclk = CAM_PIN_XCLK;
  c.pin_pclk = CAM_PIN_PCLK;
  c.pin_vsync = CAM_PIN_VSYNC;
  c.pin_href = CAM_PIN_HREF;
  c.pin_sccb_sda = CAM_PIN_SIOD;
  c.pin_sccb_scl = CAM_PIN_SIOC;
  c.pin_pwdn = CAM_PIN_PWDN;
  c.pin_reset = CAM_PIN_RESET;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;

  c.frame_size = (framesize_t)framesize;
  c.jpeg_quality = quality;
  c.fb_count = psramFound() ? 2 : 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK)
  {
    Log::event("camera_init_fail", {LI("err", (int)err)});
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s)
  {
    s->set_quality(s, quality);
    s->set_framesize(s, (framesize_t)framesize);
  }

  return true;
}

class AsyncFrameResponse : public AsyncAbstractResponse
{
public:
  explicit AsyncFrameResponse(camera_fb_t *frame) : fb(frame), idx(0)
  {
    _code = 200;
    _contentType = "image/jpeg";
    _contentLength = fb ? fb->len : 0;
    _sendContentLength = true;
    _chunked = false;
  }

  ~AsyncFrameResponse() override
  {
    if (fb)
    {
      esp_camera_fb_return(fb);
      fb = nullptr;
    }
  }

  bool _sourceValid() const override
  {
    return fb && fb->buf && fb->len > 0;
  }

  size_t _fillBuffer(uint8_t *buf, size_t maxLen) override
  {
    if (!fb)
      return 0;
    size_t remaining = fb->len - idx;
    if (remaining == 0)
      return 0;
    size_t toCopy = remaining < maxLen ? remaining : maxLen;
    std::memcpy(buf, fb->buf + idx, toCopy);
    idx += toCopy;
    return toCopy;
  }

private:
  camera_fb_t *fb;
  size_t idx;
};

class AsyncJpegStreamResponse : public AsyncAbstractResponse
{
public:
  AsyncJpegStreamResponse() : fb(nullptr), fbIdx(0), headIdx(0), needCrlf(false)
  {
    _code = 200;
    _contentType = "multipart/x-mixed-replace;boundary=frame";
    _sendContentLength = false;
    _chunked = true;
  }

  ~AsyncJpegStreamResponse() override
  {
    if (fb)
    {
      esp_camera_fb_return(fb);
      fb = nullptr;
    }
  }

  bool _sourceValid() const override { return true; }

  size_t _fillBuffer(uint8_t *buf, size_t maxLen) override
  {
    size_t out = 0;

    while (out < maxLen)
    {
      if (!fb)
      {
        fb = esp_camera_fb_get();
        if (!fb)
          break;

        head = String("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ") + String(fb->len) + String("\r\n\r\n");
        headIdx = 0;
        fbIdx = 0;
        needCrlf = true;
      }

      if (headIdx < (size_t)head.length())
      {
        size_t rem = (size_t)head.length() - headIdx;
        size_t n = rem < (maxLen - out) ? rem : (maxLen - out);
        std::memcpy(buf + out, head.c_str() + headIdx, n);
        headIdx += n;
        out += n;
        continue;
      }

      if (fbIdx < fb->len)
      {
        size_t rem = fb->len - fbIdx;
        size_t n = rem < (maxLen - out) ? rem : (maxLen - out);
        std::memcpy(buf + out, fb->buf + fbIdx, n);
        fbIdx += n;
        out += n;
        continue;
      }

      if (needCrlf)
      {
        const char *crlf = "\r\n";
        size_t n = (maxLen - out) >= 2 ? 2 : (maxLen - out);
        std::memcpy(buf + out, crlf, n);
        out += n;
        if (n == 2)
          needCrlf = false;
        continue;
      }

      esp_camera_fb_return(fb);
      fb = nullptr;
    }

    return out;
  }

private:
  camera_fb_t *fb;
  size_t fbIdx;
  String head;
  size_t headIdx;
  bool needCrlf;
};

static void sendJson(AsyncWebServerRequest *request, const String &payload)
{
  AsyncWebServerResponse *r = request->beginResponse(200, "application/json", payload);
  r->addHeader("Cache-Control", "no-store");
  request->send(r);
}

static void handleJsonBody(AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total, void (*onDone)(AsyncWebServerRequest *, const String &))
{
  if (!request->_tempObject)
    request->_tempObject = new String();
  String *body = static_cast<String *>(request->_tempObject);
  if (index == 0)
  {
    body->reserve(total);
    body->clear();
  }
  body->concat(reinterpret_cast<const char *>(data), len);
  if (index + len == total)
  {
    String payload = *body;
    delete body;
    request->_tempObject = nullptr;
    onDone(request, payload);
  }
}

static void applyCameraFromConfig()
{
  auto cfg = ConfigStore::get();
  sensor_t *s = esp_camera_sensor_get();
  if (!s)
    return;
  s->set_quality(s, cfg.cam_quality);
  s->set_framesize(s, (framesize_t)cfg.cam_framesize);
}

static void maybeLedForStream()
{
  auto cfg = ConfigStore::get();
  if (cfg.led_on_stream && Metrics::streamClients() > 0)
    ledOn(cfg.led_duty);
  else
    ledOff();
}

void CameraServer::begin()
{
  ledInit();

  auto cfg = ConfigStore::get();
  bool ok = cameraInit(cfg.cam_quality, cfg.cam_framesize);
  if (!ok)
  {
    Log::event("camera_fail", {LI("ok", 0)});
  }
  else
  {
    Log::event("camera_ok", {LI("ok", 1)});
  }

  DefaultHeaders::Instance().addHeader("Access-Control-Allow-Origin", "*");

  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    AsyncWebServerResponse* r = request->beginResponse_P(200, "text/html", INDEX_HTML_GZ, INDEX_HTML_GZ_LEN);
    r->addHeader("Content-Encoding", "gzip");
    r->addHeader("Cache-Control", "no-store");
    r->addHeader("ETag", INDEX_HTML_SHA1);
    request->send(r); });

  server.on("/health", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    StaticJsonDocument<448> doc;
    doc["ip"] = Networking::ip();
    doc["rssi"] = Networking::rssi();
    doc["online"] = Networking::online();
    doc["device_id"] = Networking::deviceId();
    doc["heap"] = (int)ESP.getFreeHeap();
    doc["psram"] = (int)ESP.getFreePsram();
    doc["uptime_ms"] = (uint32_t)millis();
    doc["stream_clients"] = Metrics::streamClients();
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/metrics", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    StaticJsonDocument<384> doc;
    doc["http"] = Metrics::http();
    doc["capture"] = Metrics::capture();
    doc["stream_clients"] = Metrics::streamClients();
    doc["mqtt_pub"] = Metrics::mqttPub();
    doc["mqtt_fail"] = Metrics::mqttFail();
    doc["logs"] = Metrics::logs();
    doc["infer_attempt"] = Metrics::inferAttempt();
    doc["infer_ok"] = Metrics::inferOk();
    doc["infer_fail"] = Metrics::inferFail();
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    auto cfg = ConfigStore::get();
    StaticJsonDocument<256> doc;
    doc["quality"] = cfg.cam_quality;
    doc["framesize"] = cfg.cam_framesize;
    doc["led_intensity"] = cfg.led_duty;
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/control", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    if (!request->hasParam("var") || !request->hasParam("val")) {
      request->send(400, "text/plain", "missing var/val");
      return;
    }

    String var = request->getParam("var")->value();
    String valS = request->getParam("val")->value();
    int val = valS.toInt();

    uint32_t h = fnv1a32(var.c_str());
    bool changed = false;

    if (h == HKEY("quality")) {
      int q = val;
      if (q < 10) q = 10;
      if (q > 63) q = 63;
      StaticJsonDocument<96> doc;
      doc["cam_quality"] = (uint8_t)q;
      String js;
      serializeJson(doc, js);
      ConfigStore::setPartialJson(js.c_str());
      applyCameraFromConfig();
      changed = true;
    } else if (h == HKEY("framesize")) {
      int fs = val;
      if (fs < 0) fs = 0;
      if (fs > 13) fs = 13;
      StaticJsonDocument<96> doc;
      doc["cam_framesize"] = (uint8_t)fs;
      String js;
      serializeJson(doc, js);
      ConfigStore::setPartialJson(js.c_str());
      applyCameraFromConfig();
      changed = true;
    } else if (h == HKEY("led_intensity")) {
      int d = val;
      if (d < 0) d = 0;
      if (d > 255) d = 255;
      StaticJsonDocument<96> doc;
      doc["led_duty"] = (uint8_t)d;
      String js;
      serializeJson(doc, js);
      ConfigStore::setPartialJson(js.c_str());
      maybeLedForStream();
      changed = true;
    }

    if (!changed) {
      request->send(400, "text/plain", "unknown var");
      return;
    }

    request->send(200, "text/plain", "OK"); });

  server.on("/api/sensors", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    SensorsSnapshot s = Sensors::latest();
    uint32_t now = millis();
    StaticJsonDocument<320> doc;
    doc["ts_ms"] = s.ts_ms;
    doc["age_ms"] = now - s.ts_ms;
    doc["soil_raw"] = s.soil_raw;
    doc["lux_raw"] = s.lux_raw;
    doc["soil_pct"] = s.soil_pct;
    doc["temp_c"] = s.temp_c;
    doc["hum_pct"] = s.hum_pct;
    doc["dht_ok"] = s.dht_ok;
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/api/irrigation", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    auto cfg = ConfigStore::get();
    IrrigationState s = Irrigation::state();
    uint32_t now = millis();
    uint32_t remaining = 0;
    if (s.pump_on && (int32_t)(s.pump_until_ms - now) > 0)
      remaining = s.pump_until_ms - now;

    uint32_t cooldownRemaining = 0;
    uint32_t sinceLastRun = now - s.last_run_ms;
    if (sinceLastRun < cfg.pump_cooldown_ms)
      cooldownRemaining = cfg.pump_cooldown_ms - sinceLastRun;

    StaticJsonDocument<256> doc;
    doc["pump_on"] = s.pump_on;
    doc["pump_until_ms"] = s.pump_until_ms;
    doc["remaining_ms"] = remaining;
    doc["last_run_ms"] = s.last_run_ms;
    doc["auto_enabled"] = s.auto_enabled;
    doc["cooldown_remaining_ms"] = cooldownRemaining;
    doc["soil_dry_threshold_pct"] = cfg.soil_dry_threshold_pct;
    doc["pump_on_ms"] = cfg.pump_on_ms;
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/api/irrigation/start", HTTP_POST, [](AsyncWebServerRequest *request)
            { Metrics::incHttp(); }, nullptr, [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total)
            { handleJsonBody(request, data, len, index, total, [](AsyncWebServerRequest *req, const String &body)
                             {
                StaticJsonDocument<128> doc;
                auto err = deserializeJson(doc, body);
                uint32_t ms = 0;
                if (!err) ms = (uint32_t)(doc["ms"] | 0);
                bool ok = Irrigation::start(ms);
                StaticJsonDocument<128> out;
                out["ok"] = ok;
                out["ms"] = ms;
                String js;
                serializeJson(out, js);
                sendJson(req, js); }); });

  server.on("/api/irrigation/stop", HTTP_POST, [](AsyncWebServerRequest *request)
            { Metrics::incHttp(); }, nullptr, [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total)
            { handleJsonBody(request, data, len, index, total, [](AsyncWebServerRequest *req, const String &)
                             {
                bool ok = Irrigation::stop();
                StaticJsonDocument<64> out;
                out["ok"] = ok;
                String js;
                serializeJson(out, js);
                sendJson(req, js); }); });

  server.on("/api/config", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    String out = ConfigStore::toJson();
    sendJson(request, out); });

  server.on("/api/config", HTTP_POST, [](AsyncWebServerRequest *request)
            { Metrics::incHttp(); }, nullptr, [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total)
            { handleJsonBody(request, data, len, index, total, [](AsyncWebServerRequest *req, const String &body)
                             {
                ConfigStore::setPartialJson(body.c_str());
                applyCameraFromConfig();
                maybeLedForStream();
                String out = ConfigStore::toJson();
                sendJson(req, out); }); });

  server.on("/api/inference/config", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    auto cfg = ConfigStore::get();
    StaticJsonDocument<512> doc;
    doc["infer_enabled"] = cfg.infer_enabled;
    doc["infer_skip_when_streaming"] = cfg.infer_skip_when_streaming;
    doc["infer_period_ms"] = cfg.infer_period_ms;
    doc["infer_host"] = cfg.infer_host;
    doc["infer_port"] = cfg.infer_port;
    doc["infer_path"] = cfg.infer_path;
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/api/inference/config", HTTP_POST, [](AsyncWebServerRequest *request)
            { Metrics::incHttp(); }, nullptr, [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total)
            { handleJsonBody(request, data, len, index, total, [](AsyncWebServerRequest *req, const String &body)
                             {
                ConfigStore::setPartialJson(body.c_str());
                String out = ConfigStore::toJson();
                sendJson(req, out);
                Log::event("infer_cfg_set", {LI("ok", 1)}); }); });

  server.on("/api/inference/run", HTTP_POST, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    InferenceClient::requestRun();
    StaticJsonDocument<128> doc;
    doc["ok"] = true;
    doc["queued"] = true;
    String out;
    serializeJson(doc, out);
    sendJson(request, out); });

  server.on("/api/inference/last", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    sendJson(request, InferenceClient::lastJson()); });

  server.on("/api/inference/log", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    if (!LittleFS.begin(true)) {
      request->send(500, "text/plain", "fs_fail");
      return;
    }
    if (!LittleFS.exists("/inference.csv")) {
      request->send(404, "text/plain", "no_log");
      return;
    }
    request->send(LittleFS, "/inference.csv", "text/csv"); });

  server.on("/capture", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    Metrics::incCapture();

    auto cfg = ConfigStore::get();

    bool keepLed = cfg.led_on_stream && Metrics::streamClients() > 0;
    if (!keepLed && LED_FLASH_GPIO >= 0) {
      ledOn(cfg.led_duty);
      delay(LED_FLASH_ON_DELAY_MS);
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!keepLed) maybeLedForStream();

    if (!fb) {
      request->send(503, "text/plain", "camera_fb_get_fail");
      return;
    }

    SensorsSnapshot s = Sensors::latest();
    IrrigationState ir = Irrigation::state();

    AsyncFrameResponse* resp = new AsyncFrameResponse(fb);
    resp->addHeader("Cache-Control", "no-store");
    resp->addHeader("X-Device-Id", Networking::deviceId());
    resp->addHeader("X-Lux-Raw", String(s.lux_raw));
    resp->addHeader("X-Soil-Raw", String(s.soil_raw));
    resp->addHeader("X-Soil-Pct", String(s.soil_pct, 2));
    resp->addHeader("X-Temp-C", String(s.temp_c, 2));
    resp->addHeader("X-Hum-Pct", String(s.hum_pct, 2));
    resp->addHeader("X-Pump-On", ir.pump_on ? "1" : "0");

    request->send(resp); });

  server.on("/stream", HTTP_GET, [](AsyncWebServerRequest *request)
            {
    Metrics::incHttp();
    Metrics::incStreamClient();

    auto cfg = ConfigStore::get();
    if (cfg.led_on_stream) ledOn(cfg.led_duty);

    if (request->client()) {
      request->client()->onDisconnect([](void*, AsyncClient*) {
        Metrics::decStreamClient();
        maybeLedForStream();
      }, nullptr);
    }

    request->send(new AsyncJpegStreamResponse()); });

  server.onNotFound([](AsyncWebServerRequest *request)
                    {
    Metrics::incHttp();
    request->send(404, "text/plain", "not found"); });

  server.begin();
}
