#include "storage.h"
#include "config.h"
#include <LittleFS.h>

static const char *EVENTS_PATH = "/events.log";
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

void Storage::loop()
{
  uint32_t now = millis();
  if (now - lastFlushMs < 15000)
    return;
  lastFlushMs = now;
}
