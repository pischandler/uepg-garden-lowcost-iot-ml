#pragma once

#include <Arduino.h>

static inline uint32_t fnv1a32(const char *s)
{
  uint32_t h = 2166136261u;
  while (*s)
  {
    h ^= (uint8_t)(*s++);
    h *= 16777619u;
  }
  return h;
}

constexpr uint32_t fnv1a32c(const char *s, uint32_t h = 2166136261u)
{
  return *s ? fnv1a32c(s + 1, (h ^ (uint8_t)*s) * 16777619u) : h;
}

#define HKEY(x) (fnv1a32c(x))
