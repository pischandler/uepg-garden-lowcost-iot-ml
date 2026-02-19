#pragma once

#include <Arduino.h>

struct IrrigationState
{
  bool pump_on;
  uint32_t pump_until_ms;
  uint32_t last_run_ms;
  bool auto_enabled;
};

class Irrigation
{
public:
  static void begin();
  static void loop();
  static IrrigationState state();
  static bool start(uint32_t ms);
  static bool stop();
};
