# Painel remoto (PC ou servidor externo)

Este documento descreve como implementar um painel de controle rodando no PC ou em um servidor externo, mantendo **todas** as funções do painel embarcado no ESP: bomba, sensores, logs, câmera ao vivo e inferência.

## Objetivo

- Servir a UI do painel a partir do PC ou de um servidor (ex.: React, Vue ou HTML estático em `http://localhost:3000` ou `https://servidor.com`).
- O painel continua controlando **um ou mais ESPs** via HTTP, desde que estejam ligados e acessíveis na rede.

## Pré-requisitos

- ESP ligado e na mesma rede do cliente que abre o painel (ou acessível por VPN/port forwarding).
- O firmware do ESP já envia `Access-Control-Allow-Origin: *` (ver `camera_server.cpp`), então requisições cross-origin do painel para o ESP são permitidas pelo navegador.
- O painel precisa conhecer o endereço base do ESP (ex.: `http://192.168.1.50` ou `http://esp-casa.local`). Pode ser configurável por dispositivo (ex.: campo "URL do ESP" ou lista de ESPs).

## Superfície de API do ESP (base URL = `<ESP_IP>` ou `http://esp.local`)

Todas as chamadas são relativas a um **base URL configurável** (ex.: `baseUrl = "http://192.168.1.50"`).

### Leitura (GET)

| Recurso            | Endpoint                    | Uso no painel                                                                                                 |
| ------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Dashboard agregado | `GET /api/dashboard`        | Payload único com health, sensors, irrigation, config, lastInfer, metrics. Preferir para reduzir round-trips. |
| Sensores           | `GET /api/sensors`          | Temperatura, umidade, solo, LDR (fallback se não usar dashboard).                                             |
| Irrigação          | `GET /api/irrigation`       | Estado da bomba, `remaining_ms`, cooldown.                                                                    |
| Config geral       | `GET /api/config`           | Parâmetros NVS (pump_on_ms, pump_cooldown_ms, etc.).                                                          |
| Config inferência  | `GET /api/inference/config` | infer_host, infer_port, infer_path.                                                                           |
| Última inferência  | `GET /api/inference/last`   | Último resultado (classe, score, topk, meta).                                                                 |
| Schema inferência  | `GET /api/inference/schema` | Listas de classes e reason IDs (i18n/mapeamento).                                                             |
| Log de inferência  | `GET /api/inference/log`    | Histórico de inferências (para lista/export).                                                                 |
| Métricas           | `GET /metrics`              | Prometheus (capture count, stream clients, etc.).                                                             |
| Saúde              | `GET /health`               | Sanidade do dispositivo.                                                                                      |

### Escrita / controle (POST)

| Ação                     | Endpoint                     | Body (JSON)                                                                                           |
| ------------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| Ligar bomba              | `POST /api/irrigation/start` | `{ "ms": 1500 }` (tempo em ms).                                                                       |
| Desligar bomba           | `POST /api/irrigation/stop`  | `{}` ou body vazio.                                                                                   |
| Rodar inferência         | `POST /api/inference/run`    | `{}`.                                                                                                 |
| Salvar config inferência | `POST /api/inference/config` | `{ "infer_host": "...", "infer_port": 5000, "infer_path": "/predict" }`.                              |
| Salvar config geral      | `POST /api/config`           | Objeto com chaves NVS (pump_on_ms, pump_cooldown_ms, led_on_stream, infer_skip_when_streaming, etc.). |

### Câmera

| Recurso       | Endpoint       | Uso no painel                                                                                                       |
| ------------- | -------------- | ------------------------------------------------------------------------------------------------------------------- |
| Stream MJPEG  | `GET /stream`  | `<img src="{baseUrl}/stream">` ou iframe. Atualização contínua.                                                     |
| Snapshot JPEG | `GET /capture` | Uma imagem por request. Para "ao vivo" com snapshots: polling (ex.: trocar `src` a cada 1–2 s) ou fetch + blob URL. |

- Para múltiplos ESPs, o painel deve usar um base URL por dispositivo (ex.: `baseUrlByDeviceId["esp-01"] = "http://192.168.1.50"`).

## O que o painel remoto precisa implementar

1. **Configuração do base URL do ESP**
   - Um campo (ou lista) para definir a URL base de cada ESP (ex.: `http://192.168.1.50`).
   - Opcional: descoberta por mDNS (ex.: `esp-casa.local`) ou lista estática de dispositivos.

2. **Chamadas HTTP**
   - Substituir todas as chamadas que hoje usam path relativo (ex.: `/api/dashboard`) por `baseUrl + "/api/dashboard"` (e assim por diante para cada endpoint acima).
   - Manter os mesmos métodos (GET/POST), headers (`Content-Type: application/json` onde aplicável) e bodies já usados no painel embarcado.

3. **Câmera ao vivo**
   - **MJPEG:** usar `src="{baseUrl}/stream"` em uma tag `<img>` (ou iframe) quando o painel estiver exibindo aquele ESP.
   - **Snapshot loop:** se preferir modo snapshot, fazer GET para `{baseUrl}/capture` em intervalo fixo e exibir a imagem (ex.: em `<img>` ou canvas).

4. **Fluxos idênticos ao painel atual**
   - Atualização periódica do estado: polling de `GET /api/dashboard` (ou conjunto de GETs) com o base URL do ESP selecionado.
   - Botão "Ligar irrigação" → `POST {baseUrl}/api/irrigation/start` com `{ "ms": <valor_configurado> }`.
   - Botão "Desligar irrigação" → `POST {baseUrl}/api/irrigation/stop`.
   - Botão "Rodar inferência" → `POST {baseUrl}/api/inference/run`; depois atualizar estado (polling ou refetch de `/api/inference/last` ou `/api/dashboard`).
   - Formulário de config do servidor de inferência → `POST {baseUrl}/api/inference/config` com infer_host, infer_port, infer_path.
   - Exibição de logs/histórico → `GET {baseUrl}/api/inference/log` (e/ou dados já presentes no dashboard).

## Múltiplos ESPs (opcional)

- Manter um base URL por dispositivo (ou por "nome" do dispositivo).
- Na UI: seletor de dispositivo (dropdown, lista, etc.); ao trocar de dispositivo, todas as chamadas e o stream passam a usar o base URL daquele ESP.
- O endpoint `/api/dashboard` já agrega dados de um único ESP; para N ESPs, o painel faz N chamadas (uma por base URL) e exibe por aba/cards/lista.

## Segurança e rede

- Em ambiente controlado (ex.: rede local), o uso de `Access-Control-Allow-Origin: *` é suficiente para o painel no PC/servidor chamar o ESP.
- Para acesso à Internet (painel em servidor público): o ESP deve ser alcançável (VPN, túnel ou port forwarding); avaliar riscos (ex.: não expor o ESP diretamente à Internet sem autenticação).

## Referências no repositório

- Endpoints e CORS: `firmware/smart-tomato-garden/src/camera_server.cpp`
- Uso atual da API no painel: `firmware/smart-tomato-garden/web/js/api.js`, `app.js` (jpost/jget, refreshPayload, ações de bomba e inferência)
- Irrigação no firmware: `firmware/smart-tomato-garden/src/irrigation.cpp`
- Inferência no firmware: `firmware/smart-tomato-garden/src/inference_client.cpp`
