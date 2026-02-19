#include "metrics.h"

static volatile uint32_t g_http = 0;
static volatile uint32_t g_capture = 0;
static volatile uint32_t g_streamClients = 0;
static volatile uint32_t g_mqttPub = 0;
static volatile uint32_t g_mqttFail = 0;
static volatile uint32_t g_logs = 0;

void Metrics::begin() {}

void Metrics::incHttp() { g_http++; }
void Metrics::incCapture() { g_capture++; }
void Metrics::incStreamClient() { g_streamClients++; }
void Metrics::decStreamClient()
{
  if (g_streamClients)
    g_streamClients--;
}
void Metrics::incMqttPub() { g_mqttPub++; }
void Metrics::incMqttFail() { g_mqttFail++; }
void Metrics::incLog() { g_logs++; }

uint32_t Metrics::http() { return g_http; }
uint32_t Metrics::capture() { return g_capture; }
uint32_t Metrics::streamClients() { return g_streamClients; }
uint32_t Metrics::mqttPub() { return g_mqttPub; }
uint32_t Metrics::mqttFail() { return g_mqttFail; }
uint32_t Metrics::logs() { return g_logs; }
