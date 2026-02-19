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
static bool wifiOk = false;
static bool mqttOk = false;

static String g_ip = "0.0.0.0";
static String g_mac = "";
static String g_deviceId = "";

static String topicEvents()
{
  String base = MQTT_BASE_TOPIC ? String(MQTT_BASE_TOPIC) : String("smart-tomato");
  return base + "/" + g_deviceId + "/events";
}

static void ntpInit()
{
  configTzTime("UTC0", NTP_SERVERS[0], NTP_SERVERS[1], NTP_SERVERS[2]);
}

static void otaInit()
{
  ArduinoOTA.setHostname(g_deviceId.c_str());
  ArduinoOTA.begin();
}

static void mqttInit()
{
  mqtt.onConnect([](bool)
                 {
    mqttOk = true;
    Log::event("mqtt_connected", {{"ok", true}}); });

  mqtt.onDisconnect([](AsyncMqttClientDisconnectReason)
                    {
    mqttOk = false;
    Log::event("mqtt_disconnected", {{"ok", false}}); });

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  if (MQTT_USER && strlen(MQTT_USER))
    mqtt.setCredentials(MQTT_USER, MQTT_PASSW);
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
  mqtt.connect();
}

static void wifiConnect()
{
  if (!WIFI_SSID || !strlen(WIFI_SSID))
    return;
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void Networking::begin()
{
  g_mac = WiFi.macAddress();
  g_deviceId = String("stg-") + g_mac;
  g_deviceId.replace(":", "");

  WiFi.onEvent([](WiFiEvent_t ev)
               {
    if (ev == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
      wifiOk = true;
      g_ip = WiFi.localIP().toString();
      Log::event("wifi_ip", {{"rssi", WiFi.RSSI()}}, {{"ip", g_ip.c_str()}});
      ntpInit();
      mqttConnectIfNeeded();
    }
    if (ev == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
      wifiOk = false;
      mqttOk = false;
      g_ip = "0.0.0.0";
      Log::event("wifi_down", {{"ok", false}});
    } });

  wifiConnect();
  otaInit();
  mqttInit();
}

void Networking::loop()
{
  ArduinoOTA.handle();

  uint32_t now = millis();
  if (!wifiOk && now - lastWifiAttempt > 8000)
  {
    lastWifiAttempt = now;
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
      mqtt.publish(topicEvents().c_str(), 0, false, buffered.c_str(), buffered.length());
      Metrics::incMqttPub();
    }
  }
}

String Networking::ip() { return g_ip; }
String Networking::mac() { return g_mac; }
String Networking::deviceId() { return g_deviceId; }
int Networking::rssi() { return wifiOk ? WiFi.RSSI() : -127; }
bool Networking::online() { return wifiOk; }

void Networking::publishEvent(const String &jsonLine)
{
  publishEvent(jsonLine.c_str());
}

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
  mqtt.publish(topicEvents().c_str(), 0, false, jsonLine);
  Metrics::incMqttPub();
}
