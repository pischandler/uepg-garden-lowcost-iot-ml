# smart-tomato-garden (ESP32-S3 + OV2640)

Build recomendado: PlatformIO.

Recursos:
- Captura e streaming MJPEG: /capture, /stream
- Status câmera: /status
- Controle câmera: /control?var=quality&val=12 (dispatch por hash)
- Saúde/telemetria: /health, /metrics, /api/sensors, /api/irrigation, /api/config
- NTP, OTA, Watchdog
- LittleFS para buffer local de eventos (quando MQTT/Wi-Fi indisponível)
- MQTT assíncrono para telemetria (opcional)

Setup:
1) Copie `include/secrets.example.h` para `include/secrets.h` e preencha.
2) Ajuste pinos e parâmetros em `include/config.h` se necessário.
3) Build/Upload.

Endpoints:
- GET  /            UI simples
- GET  /capture      JPEG + headers de metadados
- GET  /stream       MJPEG
- GET  /status       JSON status câmera (compat)
- GET  /control      var/val (compat)
- GET  /health       JSON saúde do device
- GET  /metrics      JSON métricas
- GET  /api/sensors  JSON última leitura
- GET  /api/irrigation JSON estado irrigação
- POST /api/irrigation/start {"ms": 1500}
- POST /api/irrigation/stop  {}
- GET  /api/config
- POST /api/config   JSON parcial para atualizar config persistida
