#pragma once

#include <Arduino.h>

class InferenceClient
{
public:
  static void begin();
  static void loop();
  static void requestRun();
  static String lastJson();
};
