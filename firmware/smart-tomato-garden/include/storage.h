#pragma once

#include <Arduino.h>

class Storage
{
public:
  static void begin();
  static void loop();
  static bool shouldBuffer();
  static void appendEvent(const char *line);
  static String drainEvents(size_t maxBytes);

  static void appendInferenceCsv(const char *line);
};
