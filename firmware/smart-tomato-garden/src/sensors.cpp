#include "sensors.h"
#include "config.h"
#include "logger.h"
#include <DHTesp.h>

static DHTesp dht;
static SensorsSnapshot snap;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static uint32_t lastReadMs = 0;

static float clampf(float v, float a, float b)
{
  if (v < a)
    return a;
  if (v > b)
    return b;
  return v;
}

static float soilPct(uint16_t raw, uint16_t dry, uint16_t wet)
{
  if (dry == wet)
    return 0.0f;
  float pct = 100.0f * (float)(dry - raw) / (float)(dry - wet);
  return clampf(pct, 0.0f, 100.0f);
}

void Sensors::begin()
{
  pinMode(SOIL_ADC_GPIO, INPUT);
  pinMode(LUX_ADC_GPIO, INPUT);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  dht.setup(DHT_GPIO, DHTesp::DHT22);

  snap = {};
  snap.ts_ms = millis();
  snap.dht_ok = false;
}

void Sensors::loop()
{
  uint32_t now = millis();
  if (now - lastReadMs < SENSORS_PERIOD_MS)
    return;
  lastReadMs = now;

  auto cfg = ConfigStore::get();
  uint16_t soil = (uint16_t)analogRead(SOIL_ADC_GPIO);
  uint16_t lux = (uint16_t)analogRead(LUX_ADC_GPIO);

  TempAndHumidity th = dht.getTempAndHumidity();
  bool ok = !isnan(th.temperature) && !isnan(th.humidity);

  portENTER_CRITICAL(&mux);
  snap.ts_ms = now;
  snap.soil_raw = soil;
  snap.lux_raw = lux;
  snap.soil_pct = soilPct(soil, cfg.soil_raw_dry, cfg.soil_raw_wet);
  snap.temp_c = ok ? th.temperature : snap.temp_c;
  snap.hum_pct = ok ? th.humidity : snap.hum_pct;
  snap.dht_ok = ok;
  portEXIT_CRITICAL(&mux);
}

SensorsSnapshot Sensors::latest()
{
  SensorsSnapshot out;
  portENTER_CRITICAL(&mux);
  out = snap;
  portEXIT_CRITICAL(&mux);
  return out;
}
