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

static const uint32_t INFER_HTTP_TIMEOUT_MS = 9000;

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

  bool infer_enabled;
  bool infer_skip_when_streaming;
  uint32_t infer_period_ms;

  char infer_host[64];
  uint16_t infer_port;
  char infer_path[64];
};

class ConfigStore
{
public:
  static void begin();
  static RuntimeConfig get();
  static void setPartialJson(const char *json);
  static String toJson();
};
