#pragma once

#include <Arduino.h>

class InferenceClient
{
public:
  static void begin();
  static void loop();
  static void requestRun();
  static String lastJson();
  static String statusJson();
  static String diagnosticsJson();
};
