// Copie para secrets.h e preencha. NAO commite secrets.h.
#pragma once

#define WIFI_SSID     "SUA_REDE"
#define WIFI_PASS     "SUA_SENHA"

#define MQTT_ENABLED  0
#define MQTT_HOST     ""
#define MQTT_PORT     1883
#define MQTT_USER     ""
#define MQTT_PASSW    ""
#define MQTT_BASE_TOPIC ""

// Default do servidor de inferencia (ML). Pode ser alterado pela interface web ou API.
#define ML_API_HOST   "192.168.100.11"
#define ML_API_PORT   5000
#define ML_API_PATH   "/predict"
