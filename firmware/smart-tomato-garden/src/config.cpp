#include "config.h"
#include "hash_dispatch.h"
#include <Preferences.h>
#include <ArduinoJson.h>
#include <cstring>

static Preferences prefs;
static RuntimeConfig cfg;

static void setStr(char *dst, size_t cap, const char *src)
{
  if (!dst || cap == 0)
    return;
  if (!src)
    src = "";
  std::strncpy(dst, src, cap - 1);
  dst[cap - 1] = 0;
}

static RuntimeConfig defaults()
{
  RuntimeConfig d = {}; // <-- IMPORTANTÍSSIMO: zera tudo

  d.soil_dry_threshold_pct = 35;
  d.pump_on_ms = 1500;
  d.pump_cooldown_ms = 120000;

  d.soil_raw_dry = 3200;
  d.soil_raw_wet = 1400;

  d.led_duty = 160;
  d.led_on_stream = true;

  d.cam_quality = CAMERA_JPEG_QUALITY;
  d.cam_framesize = 6;

  d.store_events = true;
  d.telemetry_enabled = true;

  d.infer_enabled = true;
  d.infer_skip_when_streaming = true;
  d.infer_period_ms = 10UL * 60UL * 1000UL;

  setStr(d.infer_host, sizeof(d.infer_host), ML_API_HOST);
  d.infer_port = ML_API_PORT;
  setStr(d.infer_path, sizeof(d.infer_path), ML_API_PATH);

  d.infer_min_lux_raw = 0;
  d.infer_use_led = false;
  d.infer_max_retries = 1;
  d.infer_retry_delay_ms = 250;

  return d;
}

void ConfigStore::begin()
{
  cfg = defaults();
  prefs.begin("stg", false);

  cfg.soil_dry_threshold_pct = prefs.getUShort("soil_thr", cfg.soil_dry_threshold_pct);
  cfg.pump_on_ms = prefs.getUShort("pump_ms", cfg.pump_on_ms);
  cfg.pump_cooldown_ms = prefs.getULong("pump_cd", cfg.pump_cooldown_ms);

  cfg.soil_raw_dry = prefs.getUShort("soil_dry", cfg.soil_raw_dry);
  cfg.soil_raw_wet = prefs.getUShort("soil_wet", cfg.soil_raw_wet);

  cfg.led_duty = prefs.getUChar("led_duty", cfg.led_duty);
  cfg.led_on_stream = prefs.getBool("led_strm", cfg.led_on_stream);

  cfg.cam_quality = prefs.getUChar("cam_q", cfg.cam_quality);
  cfg.cam_framesize = prefs.getUChar("cam_fs", cfg.cam_framesize);

  cfg.store_events = prefs.getBool("store_ev", cfg.store_events);
  cfg.telemetry_enabled = prefs.getBool("tele_en", cfg.telemetry_enabled);

  cfg.infer_enabled = prefs.getBool("inf_en", cfg.infer_enabled);
  cfg.infer_skip_when_streaming = prefs.getBool("inf_sws", cfg.infer_skip_when_streaming);
  cfg.infer_period_ms = prefs.getULong("inf_per", cfg.infer_period_ms);

  String host = prefs.getString("inf_h", cfg.infer_host);
  String path = prefs.getString("inf_p", cfg.infer_path);
  setStr(cfg.infer_host, sizeof(cfg.infer_host), host.c_str());
  setStr(cfg.infer_path, sizeof(cfg.infer_path), path.c_str());
  cfg.infer_port = prefs.getUShort("inf_pt", cfg.infer_port);

  cfg.infer_min_lux_raw = prefs.getUShort("inf_lux", cfg.infer_min_lux_raw);
  cfg.infer_use_led = prefs.getBool("inf_led", cfg.infer_use_led);
  cfg.infer_max_retries = prefs.getUChar("inf_rt", cfg.infer_max_retries);
  cfg.infer_retry_delay_ms = prefs.getUShort("inf_rdel", cfg.infer_retry_delay_ms);
}

RuntimeConfig ConfigStore::get()
{
  return cfg;
}

static void persistKeyU16(const char *key, uint16_t v) { prefs.putUShort(key, v); }
static void persistKeyU32(const char *key, uint32_t v) { prefs.putULong(key, v); }
static void persistKeyU8(const char *key, uint8_t v) { prefs.putUChar(key, v); }
static void persistKeyB(const char *key, bool v) { prefs.putBool(key, v); }
static void persistKeyS(const char *key, const char *v) { prefs.putString(key, v ? v : ""); }

static bool applyKey(const char *k, JsonVariant v)
{
  uint32_t h = fnv1a32(k);

  switch (h)
  {
  case HKEY("soil_dry_threshold_pct"):
    cfg.soil_dry_threshold_pct = (uint16_t)v.as<int>();
    persistKeyU16("soil_thr", cfg.soil_dry_threshold_pct);
    return true;

  case HKEY("pump_on_ms"):
    cfg.pump_on_ms = (uint16_t)v.as<int>();
    persistKeyU16("pump_ms", cfg.pump_on_ms);
    return true;

  case HKEY("pump_cooldown_ms"):
    cfg.pump_cooldown_ms = (uint32_t)v.as<uint32_t>();
    persistKeyU32("pump_cd", cfg.pump_cooldown_ms);
    return true;

  case HKEY("soil_raw_dry"):
    cfg.soil_raw_dry = (uint16_t)v.as<int>();
    persistKeyU16("soil_dry", cfg.soil_raw_dry);
    return true;

  case HKEY("soil_raw_wet"):
    cfg.soil_raw_wet = (uint16_t)v.as<int>();
    persistKeyU16("soil_wet", cfg.soil_raw_wet);
    return true;

  case HKEY("led_duty"):
    cfg.led_duty = (uint8_t)v.as<int>();
    persistKeyU8("led_duty", cfg.led_duty);
    return true;

  case HKEY("led_on_stream"):
    cfg.led_on_stream = v.as<bool>();
    persistKeyB("led_strm", cfg.led_on_stream);
    return true;

  case HKEY("cam_quality"):
    cfg.cam_quality = (uint8_t)v.as<int>();
    persistKeyU8("cam_q", cfg.cam_quality);
    return true;

  case HKEY("cam_framesize"):
    cfg.cam_framesize = (uint8_t)v.as<int>();
    persistKeyU8("cam_fs", cfg.cam_framesize);
    return true;

  case HKEY("store_events"):
    cfg.store_events = v.as<bool>();
    persistKeyB("store_ev", cfg.store_events);
    return true;

  case HKEY("telemetry_enabled"):
    cfg.telemetry_enabled = v.as<bool>();
    persistKeyB("tele_en", cfg.telemetry_enabled);
    return true;

  case HKEY("infer_enabled"):
    cfg.infer_enabled = v.as<bool>();
    persistKeyB("inf_en", cfg.infer_enabled);
    return true;

  case HKEY("infer_skip_when_streaming"):
    cfg.infer_skip_when_streaming = v.as<bool>();
    persistKeyB("inf_sws", cfg.infer_skip_when_streaming);
    return true;

  case HKEY("infer_period_ms"):
    cfg.infer_period_ms = (uint32_t)v.as<uint32_t>();
    persistKeyU32("inf_per", cfg.infer_period_ms);
    return true;

  case HKEY("infer_host"):
  {
    const char *s = v.as<const char *>();
    setStr(cfg.infer_host, sizeof(cfg.infer_host), s);
    persistKeyS("inf_h", cfg.infer_host);
    return true;
  }

  case HKEY("infer_port"):
    cfg.infer_port = (uint16_t)v.as<int>();
    persistKeyU16("inf_pt", cfg.infer_port);
    return true;

  case HKEY("infer_path"):
  {
    const char *s = v.as<const char *>();
    setStr(cfg.infer_path, sizeof(cfg.infer_path), s);
    persistKeyS("inf_p", cfg.infer_path);
    return true;
  }

  case HKEY("infer_min_lux_raw"):
    cfg.infer_min_lux_raw = (uint16_t)v.as<int>();
    persistKeyU16("inf_lux", cfg.infer_min_lux_raw);
    return true;

  case HKEY("infer_use_led"):
    cfg.infer_use_led = v.as<bool>();
    persistKeyB("inf_led", cfg.infer_use_led);
    return true;

  case HKEY("infer_max_retries"):
    cfg.infer_max_retries = (uint8_t)v.as<int>();
    persistKeyU8("inf_rt", cfg.infer_max_retries);
    return true;

  case HKEY("infer_retry_delay_ms"):
    cfg.infer_retry_delay_ms = (uint16_t)v.as<int>();
    persistKeyU16("inf_rdel", cfg.infer_retry_delay_ms);
    return true;

  default:
    return false;
  }
}

void ConfigStore::setPartialJson(const char *json)
{
  StaticJsonDocument<1024> doc;
  auto err = deserializeJson(doc, json);
  if (err)
    return;
  if (!doc.is<JsonObject>())
    return;

  JsonObject o = doc.as<JsonObject>();
  for (auto kv : o)
    applyKey(kv.key().c_str(), kv.value());
}

String ConfigStore::toJson()
{
  StaticJsonDocument<1024> doc;

  doc["soil_dry_threshold_pct"] = cfg.soil_dry_threshold_pct;
  doc["pump_on_ms"] = cfg.pump_on_ms;
  doc["pump_cooldown_ms"] = cfg.pump_cooldown_ms;

  doc["soil_raw_dry"] = cfg.soil_raw_dry;
  doc["soil_raw_wet"] = cfg.soil_raw_wet;

  doc["led_duty"] = cfg.led_duty;
  doc["led_on_stream"] = cfg.led_on_stream;

  doc["cam_quality"] = cfg.cam_quality;
  doc["cam_framesize"] = cfg.cam_framesize;

  doc["store_events"] = cfg.store_events;
  doc["telemetry_enabled"] = cfg.telemetry_enabled;

  doc["infer_enabled"] = cfg.infer_enabled;
  doc["infer_skip_when_streaming"] = cfg.infer_skip_when_streaming;
  doc["infer_period_ms"] = cfg.infer_period_ms;

  doc["infer_host"] = cfg.infer_host;
  doc["infer_port"] = cfg.infer_port;
  doc["infer_path"] = cfg.infer_path;

  doc["infer_min_lux_raw"] = cfg.infer_min_lux_raw;
  doc["infer_use_led"] = cfg.infer_use_led;
  doc["infer_max_retries"] = cfg.infer_max_retries;
  doc["infer_retry_delay_ms"] = cfg.infer_retry_delay_ms;

  String out;
  serializeJson(doc, out);
  return out;
}
