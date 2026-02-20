#include <Arduino.h>
#include "config.h"
#include "logger.h"
#include "metrics.h"
#include "storage.h"
#include "networking.h"
#include "sensors.h"
#include "irrigation.h"
#include "camera_server.h"
#include "inference_client.h"
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

  Log::event("boot",
             {LI("reason", (int)esp_reset_reason()),
              LI("boot_count", (int)bootCount),
              LI("chip_rev", (int)ESP.getChipRevision()),
              LI("heap", (int)ESP.getFreeHeap()),
              LI("psram", (int)ESP.getFreePsram())});

  Networking::begin();
  Sensors::begin();
  Irrigation::begin();
  CameraServer::begin();
  InferenceClient::begin();

  Log::event("ready", {LS("ip", Networking::ip().c_str()),
                       LS("mac", Networking::mac().c_str()),
                       LS("device_id", Networking::deviceId().c_str())});
}

void loop()
{
  wdtFeed();
  Networking::loop();
  Sensors::loop();
  Irrigation::loop();
  InferenceClient::loop();
  Storage::loop();
  delay(5);
}
