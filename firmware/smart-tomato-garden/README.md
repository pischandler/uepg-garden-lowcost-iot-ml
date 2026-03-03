# smart-tomato-garden

Projeto **PlatformIO** do firmware ESP32-S3 + OV2640 para o Smart Tomato Garden.

## Documentação principal

A documentação completa do firmware (setup, endpoints, configuração do servidor de inferência, Wi-Fi) está em:

- **[../README.md](../README.md)** (firmware)

## Setup em uma linha

```bash
cp include/secrets.example.h include/secrets.h   # preencher Wi-Fi e opcionais
pio run
pio run -t upload
```

## Estrutura relevante

- `include/config.h` — pinos e constantes
- `include/secrets.example.h` → `include/secrets.h` — credenciais (não versionar `secrets.h`)
- `src/` — código C++ (config, camera_server, inference_client, sensors, irrigation, storage, etc.)
- `web/` — UI; empacotar com `tools/pack_webui.py`
- `tools/set_inference.py` — configurar servidor de inferência via CLI
