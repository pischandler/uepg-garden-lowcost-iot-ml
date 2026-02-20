#include "irrigation.h"
#include "config.h"
#include "sensors.h"
#include "logger.h"

static IrrigationState st;
static portMUX_TYPE muxI = portMUX_INITIALIZER_UNLOCKED;

static void pumpWrite(bool on)
{
  bool level = PUMP_RELAY_ACTIVE_LOW ? !on : on;
  digitalWrite(PUMP_RELAY_GPIO, level ? HIGH : LOW);
}

void Irrigation::begin()
{
  pinMode(PUMP_RELAY_GPIO, OUTPUT);
  pumpWrite(false);
  st = {};
  st.auto_enabled = true;
}

static void setState(const IrrigationState &s)
{
  portENTER_CRITICAL(&muxI);
  st = s;
  portEXIT_CRITICAL(&muxI);
}

IrrigationState Irrigation::state()
{
  IrrigationState o;
  portENTER_CRITICAL(&muxI);
  o = st;
  portEXIT_CRITICAL(&muxI);
  return o;
}

bool Irrigation::start(uint32_t ms)
{
  if (ms == 0)
    return false;
  IrrigationState s = state();
  s.pump_on = true;
  s.pump_until_ms = millis() + ms;
  s.last_run_ms = millis();
  setState(s);
  pumpWrite(true);
  Log::event("pump_start", {LI("ms", (int)ms)});
  return true;
}

bool Irrigation::stop()
{
  IrrigationState s = state();
  if (!s.pump_on)
    return true;
  s.pump_on = false;
  s.pump_until_ms = 0;
  setState(s);
  pumpWrite(false);
  Log::event("pump_stop", {LI("ok", 1)});
  return true;
}

void Irrigation::loop()
{
  uint32_t now = millis();
  auto cfg = ConfigStore::get();
  IrrigationState s = state();

  if (s.pump_on && (int32_t)(now - s.pump_until_ms) >= 0)
  {
    stop();
    s = state();
  }

  if (!s.auto_enabled)
    return;
  if (s.pump_on)
    return;

  SensorsSnapshot sn = Sensors::latest();
  if (sn.ts_ms == 0)
    return;

  if (sn.soil_pct < (float)cfg.soil_dry_threshold_pct)
  {
    if (now - s.last_run_ms >= cfg.pump_cooldown_ms)
    {
      start(cfg.pump_on_ms);
    }
  }
}
