#include <Arduino.h>
#include "config.h"
#include "logger.h"
#include "metrics.h"
#include "storage.h"
#include "networking.h"
#include "sensors.h"
#include "irrigation.h"
#include "camera_server.h"
#include <esp_task_wdt.h>

static uint32_t bootCount = 0;

static void wdtInit()
{
  esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
  esp_task_wdt_add(NULL);
}

static void wdtFeed()
{
  esp_task_wdt_reset();
}

void setup()
{
  Serial.begin(115200);
  delay(200);

  bootCount++;
  Log::begin();
  Metrics::begin();
  Storage::begin();
  ConfigStore::begin();

  wdtInit();

  Log::event("boot", {{"reason", (int)esp_reset_reason()},
                      {"boot_count", (int)bootCount},
                      {"chip_rev", (int)ESP.getChipRevision()},
                      {"heap", (int)ESP.getFreeHeap()},
                      {"psram", (int)ESP.getFreePsram()}});

  Networking::begin();
  Sensors::begin();
  Irrigation::begin();
  CameraServer::begin();

  Log::event("ready", {{"ip", Networking::ip().c_str()},
                       {"mac", Networking::mac().c_str()},
                       {"device_id", Networking::deviceId().c_str()}});
}

void loop()
{
  wdtFeed();
  Networking::loop();
  Sensors::loop();
  Irrigation::loop();
  Storage::loop();
  delay(5);
}
