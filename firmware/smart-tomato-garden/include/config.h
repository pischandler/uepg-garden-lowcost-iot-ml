#pragma once

#include <Arduino.h>

#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#endif

static const uint32_t WDT_TIMEOUT_SECONDS = 10;

static const uint16_t HTTP_PORT = 80;

static const int LED_FLASH_GPIO = 22;
static const uint32_t LED_FLASH_PWM_FREQ = 5000;
static const uint8_t LED_FLASH_PWM_BITS = 8;
static const uint8_t LED_FLASH_PWM_CH = 7;
static const uint16_t LED_FLASH_ON_DELAY_MS = 140;
static const uint8_t LED_FLASH_MAX_DUTY = 255;

static const int SOIL_ADC_GPIO = 1;
static const int LUX_ADC_GPIO = 14;
static const int DHT_GPIO = 21;

static const int PUMP_RELAY_GPIO = 47;
static const bool PUMP_RELAY_ACTIVE_LOW = true;

static const uint32_t SENSORS_PERIOD_MS = 2000;
static const uint32_t TELEMETRY_PERIOD_MS = 5000;

static const char *NTP_SERVERS[] = {"pool.ntp.org", "time.google.com", "time.cloudflare.com"};

static const uint32_t CAMERA_STREAM_FPS = 12;
static const uint8_t CAMERA_JPEG_QUALITY = 12;
static const uint16_t CAMERA_STREAM_CHUNK = 1460;

static const char *UI_HTML =
    "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
    "<title>Smart Tomato</title></head><body style='font-family:system-ui;margin:16px'>"
    "<h3>Smart Tomato</h3>"
    "<div><img id=v style='max-width:100%;border:1px solid #ddd' src='/stream'></div>"
    "<div style='margin-top:12px'>"
    "<button onclick=\"fetch('/capture').then(r=>r.blob()).then(b=>{let u=URL.createObjectURL(b);window.open(u,'_blank')})\">Capture</button>"
    "<button onclick=\"fetch('/health').then(r=>r.json()).then(j=>alert(JSON.stringify(j,null,2)))\">Health</button>"
    "<button onclick=\"fetch('/api/sensors').then(r=>r.json()).then(j=>alert(JSON.stringify(j,null,2)))\">Sensors</button>"
    "</div></body></html>";

struct RuntimeConfig
{
  uint16_t soil_dry_threshold_pct;
  uint16_t pump_on_ms;
  uint32_t pump_cooldown_ms;

  uint16_t soil_raw_dry;
  uint16_t soil_raw_wet;

  uint8_t led_duty;
  bool led_on_stream;

  uint8_t cam_quality;
  uint8_t cam_framesize;

  bool store_events;
  bool telemetry_enabled;
};

class ConfigStore
{
public:
  static void begin();
  static RuntimeConfig get();
  static void setPartialJson(const char *json);
  static String toJson();
};
