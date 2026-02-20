#pragma once

#include <Arduino.h>

class Metrics
{
public:
  static void begin();
  static void incHttp();
  static void incCapture();
  static void incStreamClient();
  static void decStreamClient();
  static void incMqttPub();
  static void incMqttFail();
  static void incLog();

  static void incInferAttempt();
  static void incInferOk();
  static void incInferFail();

  static uint32_t http();
  static uint32_t capture();
  static uint32_t streamClients();
  static uint32_t mqttPub();
  static uint32_t mqttFail();
  static uint32_t logs();

  static uint32_t inferAttempt();
  static uint32_t inferOk();
  static uint32_t inferFail();
};
