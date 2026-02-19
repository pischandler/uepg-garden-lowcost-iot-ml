#pragma once

#include <Arduino.h>

class Networking
{
public:
  static void begin();
  static void loop();
  static String ip();
  static String mac();
  static String deviceId();
  static int rssi();
  static bool online();
  static void publishEvent(const String &jsonLine);
  static void publishEvent(const char *jsonLine);
};
