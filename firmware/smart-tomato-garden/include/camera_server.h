#pragma once

#include <Arduino.h>
#include <esp_camera.h>

class CameraServer
{
public:
  static void begin();
};

namespace CameraAccess
{
  bool lock(uint32_t timeoutMs);
  void unlock();
  camera_fb_t *fbGet(uint32_t timeoutMs);
  void fbReturn(camera_fb_t *fb);
}
