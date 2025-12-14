#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClient.h>
#include <DHT.h>
#include "esp_http_server.h"

#define CAMERA_MODEL_CUSTOM_ESP32S3_CAM
#include "camera_pins.h"

static const char *WIFI_SSID = "CLARO_2G07ECA7";
static const char *WIFI_PASSWORD = "Victor3894";

static const char *INFER_HOST = "192.168.0.10";
static const uint16_t INFER_PORT = 5000;
static const char *INFER_PATH = "/analisar";

static const int SOIL_SENSOR_PIN = 1;
static const int LDR_PIN = 14;
static const int DHT_PIN = 21;
static const int RELAY_PUMP_PIN = 47;
static const int RELAY_FAN_PIN = 46;

static const int ADC_MAX = 4095;
static const int SOIL_WET_ADC = 900;
static const int SOIL_DRY_ADC = 3000;
static const bool LDR_INVERT = true;

static const int SOIL_ON_ADC = 2600;
static const int SOIL_REARM_ADC = 2350;
static const uint32_t IRRIGATION_PULSE_MS = 1000;
static const uint32_t IRRIGATION_COOLDOWN_MS = 60UL * 1000UL;

static const bool ENABLE_TIME_CYCLE_MODE = false;
static const uint32_t CYCLE_PERIOD_MS = 15UL * 60UL * 1000UL;
static const uint32_t CYCLE_PULSE_MS = 1000;

static const float FAN_ON_TEMP_C = 28.0f;
static const float FAN_OFF_TEMP_C = 26.0f;
static const uint32_t FAN_MIN_SWITCH_MS = 10UL * 1000UL;

static const uint32_t SENSOR_PERIOD_MS = 5000;
static const uint32_t INFER_PERIOD_MS = 60UL * 1000UL;

static const size_t MAX_HTTP_BODY = 4096;

DHT dht(DHT_PIN, DHT22);

static httpd_handle_t httpServer = nullptr;

static bool pumpOn = false;
static bool fanOn = false;

static uint32_t pumpStartedAt = 0;
static uint32_t lastIrrigationAt = 0;
static bool soilRearmed = true;

static uint32_t lastFanSwitchAt = 0;
static uint32_t lastCycleAt = 0;

static uint32_t lastSensorAt = 0;
static uint32_t lastInferAt = 0;

static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static float percentFromRange(int x, int in_min, int in_max, bool invert) {
  if (in_max == in_min) return 0.0f;
  float t = (float)(x - in_min) / (float)(in_max - in_min);
  t = clampf(t, 0.0f, 1.0f);
  if (invert) t = 1.0f - t;
  return t * 100.0f;
}

static float soilPercent(int soilAdc) {
  return percentFromRange(soilAdc, SOIL_WET_ADC, SOIL_DRY_ADC, true);
}

static float ldrPercent(int ldrAdc) {
  return percentFromRange(ldrAdc, 0, ADC_MAX, LDR_INVERT);
}

static void setPump(bool on) {
  digitalWrite(RELAY_PUMP_PIN, on ? HIGH : LOW);
  pumpOn = on;
  if (on) pumpStartedAt = millis();
}

static void setFan(bool on) {
  digitalWrite(RELAY_FAN_PIN, on ? HIGH : LOW);
  fanOn = on;
  lastFanSwitchAt = millis();
}

static bool wifiEnsureConnected(uint32_t timeoutMs) {
  if (WiFi.status() == WL_CONNECTED) return true;

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - t0) < timeoutMs) {
    delay(250);
  }
  return WiFi.status() == WL_CONNECTED;
}

static String httpReadAll(WiFiClient &client, uint32_t timeoutMs) {
  String out;
  uint32_t t0 = millis();
  while ((millis() - t0) < timeoutMs) {
    while (client.available()) {
      char c = (char)client.read();
      out += c;
      t0 = millis();
    }
    if (!client.connected()) break;
    delay(5);
  }
  return out;
}

static String extractBody(const String &raw) {
  int idx = raw.indexOf("\r\n\r\n");
  if (idx < 0) return String();
  return raw.substring(idx + 4);
}

static String postImageToInference(const uint8_t *jpeg, size_t jpegLen, uint32_t &httpCodeOut) {
  httpCodeOut = 0;

  WiFiClient client;
  if (!client.connect(INFER_HOST, INFER_PORT)) {
    return String("{\"erro\":\"connect_failed\"}");
  }

  const String boundary = "----esp32s3boundary7MA4YWxkTrZu0gW";
  const String head =
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";

  const String tail = "\r\n--" + boundary + "--\r\n";
  const size_t contentLength = head.length() + jpegLen + tail.length();

  client.print(String("POST ") + INFER_PATH + " HTTP/1.1\r\n");
  client.print(String("Host: ") + INFER_HOST + ":" + String(INFER_PORT) + "\r\n");
  client.print("Connection: close\r\n");
  client.print(String("Content-Type: multipart/form-data; boundary=") + boundary + "\r\n");
  client.print(String("Content-Length: ") + String(contentLength) + "\r\n\r\n");

  client.print(head);

  const uint8_t *p = jpeg;
  size_t remaining = jpegLen;
  while (remaining > 0) {
    size_t chunk = remaining > 1460 ? 1460 : remaining;
    client.write(p, chunk);
    p += chunk;
    remaining -= chunk;
    delay(0);
  }

  client.print(tail);

  String raw = httpReadAll(client, 8000);

  int sp = raw.indexOf(' ');
  if (sp > 0) {
    int sp2 = raw.indexOf(' ', sp + 1);
    if (sp2 > sp) httpCodeOut = (uint32_t)raw.substring(sp + 1, sp2).toInt();
  }

  String body = extractBody(raw);
  body.trim();
  if (body.length() == 0) body = String("{\"erro\":\"empty_response\"}");
  return body;
}

static camera_fb_t *captureJpeg() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return nullptr;
  if (fb->format != PIXFORMAT_JPEG) {
    esp_camera_fb_return(fb);
    return nullptr;
  }
  return fb;
}

static esp_err_t handleRoot(httpd_req_t *req) {
  const char *html =
      "<!doctype html><html><head><meta charset='utf-8'/>"
      "<meta name='viewport' content='width=device-width,initial-scale=1'/>"
      "<title>ESP32-S3 Smart Tomato Garden</title></head><body>"
      "<h2>ESP32-S3 Smart Tomato Garden</h2>"
      "<ul>"
      "<li><a href='/stream'>/stream</a></li>"
      "<li><a href='/capture'>/capture</a></li>"
      "<li><a href='/infer'>/infer</a></li>"
      "<li><a href='/status'>/status</a></li>"
      "</ul>"
      "</body></html>";
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t handleCapture(httpd_req_t *req) {
  camera_fb_t *fb = captureJpeg();
  if (!fb) {
    httpd_resp_set_status(req, "500 Internal Server Error");
    return httpd_resp_send(req, "capture_failed", HTTPD_RESP_USE_STRLEN);
  }
  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t handleStream(httpd_req_t *req) {
  static const char *contentType = "multipart/x-mixed-replace;boundary=frame";
  static const char *boundary = "\r\n--frame\r\n";
  static const char *part = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  httpd_resp_set_type(req, contentType);

  while (true) {
    camera_fb_t *fb = captureJpeg();
    if (!fb) break;

    if (httpd_resp_send_chunk(req, boundary, strlen(boundary)) != ESP_OK) {
      esp_camera_fb_return(fb);
      break;
    }

    char hdr[64];
    int hlen = snprintf(hdr, sizeof(hdr), part, (unsigned)fb->len);
    if (httpd_resp_send_chunk(req, hdr, hlen) != ESP_OK) {
      esp_camera_fb_return(fb);
      break;
    }

    if (httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len) != ESP_OK) {
      esp_camera_fb_return(fb);
      break;
    }

    esp_camera_fb_return(fb);
    delay(50);
  }

  httpd_resp_send_chunk(req, nullptr, 0);
  return ESP_OK;
}

static esp_err_t handleInfer(httpd_req_t *req) {
  if (WiFi.status() != WL_CONNECTED) {
    httpd_resp_set_status(req, "503 Service Unavailable");
    return httpd_resp_send(req, "{\"erro\":\"wifi_disconnected\"}", HTTPD_RESP_USE_STRLEN);
  }

  camera_fb_t *fb = captureJpeg();
  if (!fb) {
    httpd_resp_set_status(req, "500 Internal Server Error");
    return httpd_resp_send(req, "{\"erro\":\"capture_failed\"}", HTTPD_RESP_USE_STRLEN);
  }

  uint32_t code = 0;
  uint32_t t0 = millis();
  String body = postImageToInference(fb->buf, fb->len, code);
  uint32_t dt = millis() - t0;

  esp_camera_fb_return(fb);

  httpd_resp_set_type(req, "application/json; charset=utf-8");

  if (code < 200 || code >= 300) {
    String err = String("{\"erro\":\"inference_http_error\",\"http_code\":") + String(code) +
                 String(",\"elapsed_ms\":") + String(dt) +
                 String(",\"body\":") + String("\"") + body + String("\"}") ;
    return httpd_resp_send(req, err.c_str(), HTTPD_RESP_USE_STRLEN);
  }

  return httpd_resp_send(req, body.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t handleStatus(httpd_req_t *req) {
  int soilAdc = analogRead(SOIL_SENSOR_PIN);
  int ldrAdc = analogRead(LDR_PIN);
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  bool dhtOk = !(isnan(t) || isnan(h));

  String json = "{";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
  json += "\"soil_adc\":" + String(soilAdc) + ",";
  json += "\"soil_pct\":" + String(soilPercent(soilAdc), 1) + ",";
  json += "\"ldr_adc\":" + String(ldrAdc) + ",";
  json += "\"light_pct\":" + String(ldrPercent(ldrAdc), 1) + ",";
  json += "\"temp_c\":" + String(dhtOk ? t : -1.0f, 1) + ",";
  json += "\"humidity_pct\":" + String(dhtOk ? h : -1.0f, 1) + ",";
  json += "\"pump_on\":" + String(pumpOn ? "true" : "false") + ",";
  json += "\"fan_on\":" + String(fanOn ? "true" : "false");
  json += "}";

  httpd_resp_set_type(req, "application/json; charset=utf-8");
  return httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
}

static void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.stack_size = 8192;

  if (httpd_start(&httpServer, &config) != ESP_OK) {
    httpServer = nullptr;
    return;
  }

  httpd_uri_t uriRoot = { .uri = "/", .method = HTTP_GET, .handler = handleRoot, .user_ctx = nullptr };
  httpd_uri_t uriStream = { .uri = "/stream", .method = HTTP_GET, .handler = handleStream, .user_ctx = nullptr };
  httpd_uri_t uriCapture = { .uri = "/capture", .method = HTTP_GET, .handler = handleCapture, .user_ctx = nullptr };
  httpd_uri_t uriInfer = { .uri = "/infer", .method = HTTP_GET, .handler = handleInfer, .user_ctx = nullptr };
  httpd_uri_t uriStatus = { .uri = "/status", .method = HTTP_GET, .handler = handleStatus, .user_ctx = nullptr };

  httpd_register_uri_handler(httpServer, &uriRoot);
  httpd_register_uri_handler(httpServer, &uriStream);
  httpd_register_uri_handler(httpServer, &uriCapture);
  httpd_register_uri_handler(httpServer, &uriInfer);
  httpd_register_uri_handler(httpServer, &uriStatus);
}

static void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  if (!psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    delay(2000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 1);
  }
}

static void initIO() {
  dht.begin();

  pinMode(RELAY_PUMP_PIN, OUTPUT);
  pinMode(RELAY_FAN_PIN, OUTPUT);

  digitalWrite(RELAY_PUMP_PIN, LOW);
  digitalWrite(RELAY_FAN_PIN, LOW);

  setAnalogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  pinMode(LDR_PIN, INPUT);
  pinMode(SOIL_SENSOR_PIN, INPUT);
}

static void controlPumpThreshold(int soilAdc, uint32_t now) {
  if (pumpOn) {
    if ((now - pumpStartedAt) >= IRRIGATION_PULSE_MS) {
      setPump(false);
      lastIrrigationAt = now;
    }
    return;
  }

  if (soilAdc < SOIL_REARM_ADC) soilRearmed = true;

  if (!soilRearmed) return;
  if ((now - lastIrrigationAt) < IRRIGATION_COOLDOWN_MS) return;

  if (soilAdc >= SOIL_ON_ADC) {
    soilRearmed = false;
    setPump(true);
  }
}

static void controlPumpCyclic(uint32_t now) {
  if (pumpOn) {
    if ((now - pumpStartedAt) >= CYCLE_PULSE_MS) {
      setPump(false);
    }
    return;
  }

  if ((now - lastCycleAt) >= CYCLE_PERIOD_MS) {
    lastCycleAt = now;
    setPump(true);
  }
}

static void controlFan(float tempC, bool tempValid, uint32_t now) {
  if (!tempValid) return;
  if ((now - lastFanSwitchAt) < FAN_MIN_SWITCH_MS) return;

  if (!fanOn && tempC >= FAN_ON_TEMP_C) {
    setFan(true);
    return;
  }
  if (fanOn && tempC <= FAN_OFF_TEMP_C) {
    setFan(false);
    return;
  }
}

static void logStatus(uint32_t now, int soilAdc, int ldrAdc, float tempC, float humPct, bool dhtOk) {
  Serial.print("t=");
  Serial.print(now / 1000UL);
  Serial.print("s,soil_adc=");
  Serial.print(soilAdc);
  Serial.print(",soil_pct=");
  Serial.print(soilPercent(soilAdc), 1);
  Serial.print(",ldr_adc=");
  Serial.print(ldrAdc);
  Serial.print(",light_pct=");
  Serial.print(ldrPercent(ldrAdc), 1);
  Serial.print(",temp_c=");
  Serial.print(dhtOk ? String(tempC, 1) : String("nan"));
  Serial.print(",hum_pct=");
  Serial.print(dhtOk ? String(humPct, 1) : String("nan"));
  Serial.print(",pump=");
  Serial.print(pumpOn ? "1" : "0");
  Serial.print(",fan=");
  Serial.println(fanOn ? "1" : "0");
}

static void periodicInference(uint32_t now) {
  if (WiFi.status() != WL_CONNECTED) return;
  if ((now - lastInferAt) < INFER_PERIOD_MS) return;

  camera_fb_t *fb = captureJpeg();
  if (!fb) return;

  uint32_t code = 0;
  uint32_t t0 = millis();
  String body = postImageToInference(fb->buf, fb->len, code);
  uint32_t dt = millis() - t0;

  esp_camera_fb_return(fb);

  Serial.print("infer_http=");
  Serial.print(code);
  Serial.print(",elapsed_ms=");
  Serial.print(dt);
  Serial.print(",body=");
  Serial.println(body);

  lastInferAt = now;
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  initIO();
  initCamera();

  wifiEnsureConnected(20000);

  startCameraServer();

  lastSensorAt = millis();
  lastInferAt = millis();
  lastIrrigationAt = millis();
  lastCycleAt = millis();
  lastFanSwitchAt = millis();
}

void loop() {
  uint32_t now = millis();

  if (WiFi.status() != WL_CONNECTED) {
    wifiEnsureConnected(5000);
  }

  if ((now - lastSensorAt) >= SENSOR_PERIOD_MS) {
    lastSensorAt = now;

    int soilAdc = analogRead(SOIL_SENSOR_PIN);
    int ldrAdc = analogRead(LDR_PIN);

    float tempC = dht.readTemperature();
    float humPct = dht.readHumidity();
    bool dhtOk = !(isnan(tempC) || isnan(humPct));

    if (ENABLE_TIME_CYCLE_MODE) {
      controlPumpCyclic(now);
    } else {
      controlPumpThreshold(soilAdc, now);
    }

    controlFan(tempC, dhtOk, now);
    logStatus(now, soilAdc, ldrAdc, tempC, humPct, dhtOk);
  }

  periodicInference(now);

  delay(10);
}
