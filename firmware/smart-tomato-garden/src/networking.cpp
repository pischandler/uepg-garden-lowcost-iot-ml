// src/networking.cpp
#include "networking.h"
#include "config.h"
#include "logger.h"
#include "metrics.h"
#include "storage.h"
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <AsyncMqttClient.h>

static AsyncMqttClient mqtt;
static uint32_t lastWifiAttempt = 0;
static uint32_t lastTelemetryFlush = 0;
static uint32_t lastMqttAttempt = 0;
static uint32_t wifiAttempts = 0;
static bool wifiOk = false;
static bool mqttOk = false;
static bool otaOk = false;

static String g_ip = "0.0.0.0";
static String g_mac = "";
static String g_deviceId = "";

static String macFromEfuseWithColons()
{
  uint64_t chip = ESP.getEfuseMac(); // 48 bits úteis
  uint8_t m[6];
  for (int i = 0; i < 6; i++)
  {
    m[5 - i] = (uint8_t)(chip & 0xFF);
    chip >>= 8;
  }
  char buf[18];
  snprintf(buf, sizeof(buf), "%02X:%02X:%02X:%02X:%02X:%02X",
           m[0], m[1], m[2], m[3], m[4], m[5]);
  return String(buf);
}

static String topicEvents()
{
  String base = MQTT_BASE_TOPIC ? String(MQTT_BASE_TOPIC) : String("smart-tomato");
  return base + "/" + g_deviceId + "/events";
}

static void ntpInit()
{
  configTzTime("UTC0", NTP_SERVERS[0], NTP_SERVERS[1], NTP_SERVERS[2]);
  Log::event("ntp_init", {LI("ok", 1)});
}

static void otaInitOnce()
{
  if (otaOk)
    return;
  ArduinoOTA.setHostname(g_deviceId.c_str());
  ArduinoOTA.begin();
  otaOk = true;
  Log::event("ota_ready", {LI("ok", 1)});
}

static void mqttInit()
{
  mqtt.onConnect([](bool)
                 { mqttOk = true; Log::event("mqtt_connected", {LB("ok", true)}); });
  mqtt.onDisconnect([](AsyncMqttClientDisconnectReason)
                    { mqttOk = false; Log::event("mqtt_disconnected", {LB("ok", false)}); });

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  if (MQTT_USER && strlen(MQTT_USER))
    mqtt.setCredentials(MQTT_USER, MQTT_PASSW);
}

static void wifiConnect()
{
  if (!WIFI_SSID || !strlen(WIFI_SSID))
  {
    Log::event("wifi_no_ssid", {LI("ok", 0)});
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  wifiAttempts++;
  Log::event("wifi_connect_attempt", {LI("n", (int)wifiAttempts)});
}

static void mqttConnectIfNeeded()
{
  auto cfg = ConfigStore::get();
  if (!cfg.telemetry_enabled)
    return;
  if (!MQTT_ENABLED)
    return;
  if (!wifiOk)
    return;
  if (mqttOk)
    return;
  if (!MQTT_HOST || !strlen(MQTT_HOST))
    return;

  uint32_t now = millis();
  if (now - lastMqttAttempt < 5000)
    return;
  lastMqttAttempt = now;

  Log::event("mqtt_connect_attempt", {LI("ok", 1)});
  mqtt.connect();
}

void Networking::begin()
{
  // MAC/deviceId SEM depender do WiFi estar pronto
  g_mac = macFromEfuseWithColons();
  g_deviceId = String("stg-") + g_mac;
  g_deviceId.replace(":", "");

  WiFi.onEvent([](WiFiEvent_t ev, WiFiEventInfo_t info)
               {
    if (ev == ARDUINO_EVENT_WIFI_STA_GOT_IP)
    {
      wifiOk = true;
      g_ip = WiFi.localIP().toString();
      Log::event("wifi_ip", {LI("rssi", WiFi.RSSI())}, {LS("ip", g_ip.c_str())});
      ntpInit();
      otaInitOnce();
      mqttConnectIfNeeded();
    }

    if (ev == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
    {
      wifiOk = false;
      mqttOk = false;
      g_ip = "0.0.0.0";
      uint8_t reason = info.wifi_sta_disconnected.reason;
      Log::event("wifi_down", {LI("ok", 0), LI("reason", (int)reason)});
    } });

  mqttInit();
  wifiConnect();
}

void Networking::loop()
{
  if (otaOk)
    ArduinoOTA.handle();

  uint32_t now = millis();
  if (!wifiOk && now - lastWifiAttempt > 8000)
  {
    lastWifiAttempt = now;
    Log::event("wifi_retry", {LI("n", (int)wifiAttempts)});
    wifiConnect();
  }

  mqttConnectIfNeeded();

  auto cfg = ConfigStore::get();
  if (!cfg.telemetry_enabled)
    return;

  if (wifiOk && mqttOk && now - lastTelemetryFlush > 12000)
  {
    lastTelemetryFlush = now;
    String buffered = Storage::drainEvents(24 * 1024);
    if (buffered.length())
    {
      uint16_t pid = mqtt.publish(topicEvents().c_str(), 0, false, buffered.c_str(), buffered.length());
      if (pid)
        Metrics::incMqttPub();
      else
        Metrics::incMqttFail();
    }
  }
}

String Networking::ip() { return g_ip; }
String Networking::mac() { return g_mac; }
String Networking::deviceId() { return g_deviceId; }
int Networking::rssi() { return wifiOk ? WiFi.RSSI() : -127; }
bool Networking::online() { return wifiOk; }

void Networking::publishEvent(const String &jsonLine) { publishEvent(jsonLine.c_str()); }

void Networking::publishEvent(const char *jsonLine)
{
  auto cfg = ConfigStore::get();
  if (!cfg.telemetry_enabled)
    return;
  if (!MQTT_ENABLED)
    return;
  if (!mqttOk)
  {
    Metrics::incMqttFail();
    return;
  }

  uint16_t pid = mqtt.publish(topicEvents().c_str(), 0, false, jsonLine);
  if (pid)
    Metrics::incMqttPub();
  else
    Metrics::incMqttFail();
}
