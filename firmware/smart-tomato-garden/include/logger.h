#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

class Log
{
public:
  struct KV
  {
    const char *k;
    const char *v;
  };
  struct KVI
  {
    const char *k;
    int v;
  };
  struct KVU
  {
    const char *k;
    uint32_t v;
  };
  struct KVF
  {
    const char *k;
    float v;
  };
  struct KVB
  {
    const char *k;
    bool v;
  };

  static void begin();
  static void rawJson(const char *jsonLine);

  static void event(const char *name, std::initializer_list<Log::KVI> ints);
  static void event(const char *name, std::initializer_list<Log::KVU> uints);
  static void event(const char *name, std::initializer_list<Log::KV> strs);
  static void event(const char *name, std::initializer_list<Log::KVB> bools);
  static void event(const char *name, std::initializer_list<Log::KVF> floats);

  static void event(const char *name, std::initializer_list<Log::KVI> ints,
                    std::initializer_list<Log::KVU> uints,
                    std::initializer_list<Log::KV> strs,
                    std::initializer_list<Log::KVB> bools,
                    std::initializer_list<Log::KVF> floats);

  static void event(const char *name, std::initializer_list<Log::KVI> ints, std::initializer_list<Log::KV> strs);
};

#define LI(k, v) \
  Log::KVI { (k), (int)(v) }
#define LU(k, v) \
  Log::KVU { (k), (uint32_t)(v) }
#define LB(k, v) \
  Log::KVB { (k), (bool)(v) }
#define LF(k, v) \
  Log::KVF { (k), (float)(v) }
#define LS(k, v) \
  Log::KV { (k), (v) }
