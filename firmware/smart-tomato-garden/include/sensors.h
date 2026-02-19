#pragma once

#include <Arduino.h>

struct SensorsSnapshot
{
  uint32_t ts_ms;
  uint16_t soil_raw;
  uint16_t lux_raw;
  float soil_pct;
  float temp_c;
  float hum_pct;
  bool dht_ok;
};

class Sensors
{
public:
  static void begin();
  static void loop();
  static SensorsSnapshot latest();
};
