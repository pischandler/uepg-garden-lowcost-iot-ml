#include "storage.h"
#include "config.h"
#include <LittleFS.h>

static const char *EVENTS_PATH = "/events.log";
static const char *INFER_CSV_PATH = "/inference.csv";
static uint32_t lastFlushMs = 0;

void Storage::begin()
{
  LittleFS.begin(true);
}

bool Storage::shouldBuffer()
{
  auto cfg = ConfigStore::get();
  return cfg.store_events;
}

void Storage::appendEvent(const char *line)
{
  File f = LittleFS.open(EVENTS_PATH, "a");
  if (!f)
    return;
  f.println(line);
  f.close();
}

String Storage::drainEvents(size_t maxBytes)
{
  File f = LittleFS.open(EVENTS_PATH, "r");
  if (!f)
    return "";

  String out;
  out.reserve(maxBytes);

  while (f.available() && out.length() < maxBytes)
  {
    String line = f.readStringUntil('\n');
    if (line.length() == 0)
      continue;
    out += line;
    if (!out.endsWith("\n"))
      out += "\n";
  }

  f.close();
  LittleFS.remove(EVENTS_PATH);
  return out;
}

static void ensureCsvHeader()
{
  if (LittleFS.exists(INFER_CSV_PATH))
    return;
  File f = LittleFS.open(INFER_CSV_PATH, "w");
  if (!f)
    return;

  f.println("ts_unix,ts_ms,device_id,ip,rssi,ok,http_status,latency_ms,predicted,confidence,confident,reasons,soil_pct,soil_raw,lux_raw,temp_c,hum_pct,dht_ok,pump_on");
  f.close();
}

void Storage::appendInferenceCsv(const char *line)
{
  ensureCsvHeader();

  File f = LittleFS.open(INFER_CSV_PATH, "a");
  if (!f)
    return;

  if (line && strlen(line))
    f.println(line);

  size_t sz = f.size();
  f.close();

  // evita estourar flash
  if (sz > (2UL * 1024UL * 1024UL))
    LittleFS.remove(INFER_CSV_PATH);
}

void Storage::loop()
{
  uint32_t now = millis();
  if (now - lastFlushMs < 15000)
    return;
  lastFlushMs = now;
}
