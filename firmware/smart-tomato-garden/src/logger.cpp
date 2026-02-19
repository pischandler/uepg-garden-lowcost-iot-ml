#include "logger.h"
#include "networking.h"
#include "metrics.h"
#include "storage.h"

static void emitDoc(JsonDocument &doc)
{
  String out;
  serializeJson(doc, out);
  Serial.println(out);
  Metrics::incLog();
  if (Storage::shouldBuffer())
    Storage::appendEvent(out.c_str());
  Networking::publishEvent(out);
}

void Log::begin() {}

void Log::rawJson(const char *jsonLine)
{
  Serial.println(jsonLine);
  Metrics::incLog();
  if (Storage::shouldBuffer())
    Storage::appendEvent(jsonLine);
  Networking::publishEvent(jsonLine);
}

static void fillBase(JsonObject o, const char *name)
{
  o["ts_ms"] = (uint32_t)millis();
  o["event"] = name;
  o["heap"] = (int)ESP.getFreeHeap();
  o["uptime_ms"] = (uint32_t)millis();
  o["rssi"] = Networking::rssi();
  o["ip"] = Networking::ip();
  o["device_id"] = Networking::deviceId();
}

void Log::event(const char *name, std::initializer_list<Log::KVI> ints,
                std::initializer_list<Log::KVU> uints,
                std::initializer_list<Log::KV> strs,
                std::initializer_list<Log::KVB> bools,
                std::initializer_list<Log::KVF> floats)
{
  StaticJsonDocument<640> doc;
  auto o = doc.to<JsonObject>();
  fillBase(o, name);
  for (auto &kv : ints)
    o[kv.k] = kv.v;
  for (auto &kv : uints)
    o[kv.k] = kv.v;
  for (auto &kv : strs)
    o[kv.k] = kv.v;
  for (auto &kv : bools)
    o[kv.k] = kv.v;
  for (auto &kv : floats)
    o[kv.k] = kv.v;
  emitDoc(doc);
}

void Log::event(const char *name, std::initializer_list<Log::KVI> ints)
{
  event(name, ints, {}, {}, {}, {});
}
void Log::event(const char *name, std::initializer_list<Log::KVU> uints)
{
  event(name, {}, uints, {}, {}, {});
}
void Log::event(const char *name, std::initializer_list<Log::KV> strs)
{
  event(name, {}, {}, strs, {}, {});
}
void Log::event(const char *name, std::initializer_list<Log::KVB> bools)
{
  event(name, {}, {}, {}, bools, {});
}
void Log::event(const char *name, std::initializer_list<Log::KVF> floats)
{
  event(name, {}, {}, {}, {}, floats);
}

void Log::event(const char *name, std::initializer_list<Log::KVI> ints, std::initializer_list<Log::KV> strs)
{
  event(name, ints, {}, strs, {}, {});
}
