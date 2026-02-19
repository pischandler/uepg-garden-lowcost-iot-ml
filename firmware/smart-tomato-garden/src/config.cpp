#include "config.h"
#include "hash_dispatch.h"
#include <Preferences.h>
#include <ArduinoJson.h>

static Preferences prefs;
static RuntimeConfig cfg;

static RuntimeConfig defaults()
{
  RuntimeConfig d;
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
}

RuntimeConfig ConfigStore::get()
{
  return cfg;
}

static void persistKeyU16(const char *key, uint16_t v) { prefs.putUShort(key, v); }
static void persistKeyU32(const char *key, uint32_t v) { prefs.putULong(key, v); }
static void persistKeyU8(const char *key, uint8_t v) { prefs.putUChar(key, v); }
static void persistKeyB(const char *key, bool v) { prefs.putBool(key, v); }

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
  default:
    return false;
  }
}

void ConfigStore::setPartialJson(const char *json)
{
  StaticJsonDocument<768> doc;
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
  StaticJsonDocument<512> doc;
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
  String out;
  serializeJson(doc, out);
  return out;
}
