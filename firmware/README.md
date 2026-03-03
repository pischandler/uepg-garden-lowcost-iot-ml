# Firmware — Smart Tomato Garden (ESP32-S3 + OV2640)

Firmware para **ESP32-S3** com câmera **OV2640**, sensores e atuadores, usado na horta inteligente. Expõe captura de imagem e streaming para o servidor de inferência ML e permite configurar o endpoint da API (host, porta, path) pela interface web ou por script.

---

## Visão geral

- **Build:** PlatformIO (em `smart-tomato-garden/`).
- **Recursos:** Captura JPEG e streaming MJPEG, status e controle da câmera, saúde e telemetria (sensores, irrigação), configuração persistida em NVS (incluindo servidor de inferência), NTP, OTA, watchdog.
- **Opcional:** MQTT para telemetria; LittleFS para buffer de eventos quando Wi-Fi/MQTT indisponível.

---

## Estrutura

```
firmware/
├── smart-tomato-garden/   # Projeto PlatformIO
│   ├── include/           # config.h, secrets.example.h → secrets.h, headers dos módulos
│   ├── src/               # config, camera_server, inference_client, sensors, irrigation, etc.
│   ├── web/               # UI embarcada (template, i18n, js, css); empacotada com tools/pack_webui.py
│   ├── tools/             # set_inference.py, pack_webui.py, etc.
│   └── platformio.ini
└── README.md              # Este arquivo
```

---

## Setup rápido

1. **Entrar no projeto**
   ```bash
   cd firmware/smart-tomato-garden
   ```

2. **Configurar credenciais e defaults**
   - Copiar `include/secrets.example.h` para `include/secrets.h`.
   - Preencher Wi-Fi (SSID, senha), opcionalmente MQTT e o default do servidor de inferência (host, porta, path).

3. **Ajustar hardware (se necessário)**  
   Editar `include/config.h` (pinos, constantes).

4. **Build e upload**
   ```bash
   pio run
   pio run -t upload
   ```

**Importante:** Não versionar `secrets.h`; manter apenas `secrets.example.h` no repositório.

---

## Configuração do servidor de inferência (ML)

O ESP precisa saber para onde enviar as imagens (host, porta e caminho da API). Duas formas:

### 1. Interface web (recomendado para vários dispositivos)

Na interface do device (acessar pelo IP na rede), usar o card **"Configuração do servidor de inferência"**: informar IP ou hostname do servidor ML, porta (ex.: 5000) e caminho da API (ex.: `/predict` ou `/analisar`). Clicar em **Salvar**. A configuração é gravada na NVS e persiste após reinício.

### 2. Linha de comando

Com o ESP na rede e conhecendo o IP dele e do servidor ML:

```bash
python tools/set_inference.py --esp 192.168.100.12 --ml 192.168.100.11 --port 5000 --path /predict
```

---

## Endpoints HTTP

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | UI embarcada |
| GET | `/capture` | JPEG + headers de metadados |
| GET | `/stream` | Streaming MJPEG |
| GET | `/status` | JSON status da câmera (compat) |
| GET | `/control` | Controle câmera (var/val, compat) |
| GET | `/health` | JSON saúde do device |
| GET | `/metrics` | JSON métricas |
| GET | `/api/sensors` | Última leitura de sensores |
| GET | `/api/irrigation` | Estado da irrigação |
| POST | `/api/irrigation/start` | Body: `{"ms": 1500}` |
| POST | `/api/irrigation/stop` | Body: `{}` |
| GET | `/api/config` | Configuração atual |
| POST | `/api/config` | JSON parcial para atualizar config |
| GET | `/api/inference/config` | Config do servidor de inferência |
| POST | `/api/inference/config` | JSON `{ infer_host, infer_port, infer_path }` (persistido na NVS) |

---

## Câmera e stream

- **MJPEG** (`/stream`): o firmware usa timeout de captura que aumenta com a resolução (100 ms em baixa, 180 ms em média, 250 ms em alta) para reduzir falhas e travamentos. O header multipart é montado em buffer fixo (sem alocação por frame).
- Para **stream mais fluido**, use resoluções menores (ex.: CIF, SVGA) e qualidade JPEG moderada; resoluções altas (VGA, UXGA) são suportadas, mas o FPS efetivo será menor e podem ocorrer atrasos pontuais.
- **Captura única** (`/capture`): não sofre o mesmo gargalo do stream contínuo; use a resolução desejada para fotos ou inferência.

---

## Wi-Fi

A rede (SSID e senha) é definida em tempo de compilação em `secrets.h`. Configurar Wi-Fi pela interface web exigiria alterações no firmware (ex.: modo AP + portal de configuração ou gravação de credenciais na NVS).

---

## Documentação adicional

- **README principal do repositório:** [../README.md](../README.md)
- **Arquitetura:** [../docs/architecture.md](../docs/architecture.md)
- **API de inferência (servidor):** [../docs/api.md](../docs/api.md)
- **Processo e RFCs (se existirem):** `smart-tomato-garden/docs/`
