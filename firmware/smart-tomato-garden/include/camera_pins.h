#pragma once
#include <Arduino.h>

static const int CAM_PIN_PWDN = -1;
static const int CAM_PIN_RESET = -1;

static const int CAM_PIN_XCLK = 15;
static const int CAM_PIN_SIOD = 4; // SIOD
static const int CAM_PIN_SIOC = 5; // SIOC

// IMPORTANT: esp_camera pin_d0..pin_d7 = Y2..Y9
static const int CAM_PIN_D0 = 11; // Y2
static const int CAM_PIN_D1 = 9;  // Y3
static const int CAM_PIN_D2 = 8;  // Y4
static const int CAM_PIN_D3 = 10; // Y5
static const int CAM_PIN_D4 = 12; // Y6
static const int CAM_PIN_D5 = 18; // Y7
static const int CAM_PIN_D6 = 17; // Y8
static const int CAM_PIN_D7 = 16; // Y9

static const int CAM_PIN_VSYNC = 6;
static const int CAM_PIN_HREF = 7;
static const int CAM_PIN_PCLK = 13;
