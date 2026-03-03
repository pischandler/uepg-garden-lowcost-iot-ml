# Roadmap: produto reprodutível para hortas domésticas

Este documento descreve implementações recomendadas para transformar o Smart Tomato Garden em um **produto reprodutível**, focado em hortas domésticas com **controle completo**: múltiplos cultivos, parâmetros por tipo de planta, histórico persistido e experiência de uso escalável.

---

## 1. Visão de produto

- **Público:** hortas domésticas (balcão, quintal, estufa pequena).
- **Objetivo:** um mesmo software/hardware pode ser replicado por qualquer usuário: configurar dispositivo(s), escolher tipo de cultivo, definir regras de irrigação e acompanhar sensores + detecção de doenças.
- **Entregas mínimas para “produto”:**
  - Setup guiado (WiFi, servidor de inferência, primeiro cultivo).
  - Persistência de histórico (sensores, irrigação, inferências) em banco de dados.
  - Suporte a vários tipos de cultivo com parâmetros ajustáveis.
  - Painel único (embarcado ou remoto) que controle tudo que hoje já existe (bomba, câmera, inferência, logs).

---

## 2. Banco de dados: sim, recomendado

Centralizar dados em um backend (PC local ou servidor) permite:

- Histórico longo de sensores e irrigação (gráficos, relatórios).
- Histórico de inferências (tendência de doenças, eficácia do modelo).
- Multi-dispositivo e multi-horta (vários ESPs, vários “jardins”).
- Regras e agendamentos (irrigar quando solo < X%, ou às 7h).
- Alertas e notificações (doença detectada, sensor fora da faixa).

### 2.1 Quais dados controlar

#### 2.1.1 Dispositivos (ESPs)

| Campo                  | Tipo      | Descrição                                                  |
| ---------------------- | --------- | ---------------------------------------------------------- |
| id                     | PK        | Identificador único                                        |
| device_id              | string    | ID lógico (ex.: "stg-01")                                  |
| name                   | string    | Nome amigável ("Horta varanda")                            |
| base_url               | string    | URL base (ex.: http://192.168.1.50) ou último IP conhecido |
| firmware_version       | string    | Versão do firmware (para OTA e suporte)                    |
| garden_id              | FK        | Jardim ao qual pertence                                    |
| last_seen_at           | timestamp | Última comunicação (health/dashboard)                      |
| config_snapshot        | JSON      | Cópia última de config NVS (opcional, para diff/restore)   |
| created_at, updated_at | timestamp | Auditoria                                                  |

#### 2.1.2 Jardins / hortas

| Campo                  | Tipo      | Descrição                            |
| ---------------------- | --------- | ------------------------------------ |
| id                     | PK        | Identificador único                  |
| name                   | string    | Nome do jardim                       |
| user_id                | FK        | Dono (se multi-usuário)              |
| timezone               | string    | Fuso para agendamentos e gráficos    |
| location_notes         | string    | Opcional (interior/exterior, cidade) |
| created_at, updated_at | timestamp | Auditoria                            |

#### 2.1.3 Cultivos (plantas por jardim ou por dispositivo)

| Campo                  | Tipo      | Descrição                                                                      |
| ---------------------- | --------- | ------------------------------------------------------------------------------ |
| id                     | PK        | Identificador único                                                            |
| garden_id ou device_id | FK        | Escopo: jardim inteiro ou um ESP                                               |
| crop_type_id           | FK        | Tipo (tomate, alface, manjericão, etc.)                                        |
| name                   | string    | Nome amigável ("Tomateiros varanda")                                           |
| params                 | JSON      | Parâmetros (ver seção 3): temp_min/max, umidade_solo_meta, duração_bomba, etc. |
| active                 | boolean   | Se ainda está em uso                                                           |
| started_at, ended_at   | date      | Período do cultivo (opcional)                                                  |
| created_at, updated_at | timestamp | Auditoria                                                                      |

#### 2.1.4 Tipos de cultivo (catálogo)

| Campo                  | Tipo      | Descrição                                                             |
| ---------------------- | --------- | --------------------------------------------------------------------- |
| id                     | PK        | Identificador único                                                   |
| code                   | string    | Código estável ("tomato", "lettuce", "basil")                         |
| display_name           | string    | Nome para UI                                                          |
| default_params         | JSON      | Parâmetros padrão ideais para esse cultivo                            |
| ml_model_hint          | string    | Opcional: qual modelo/endpoint de inferência (ex.: "tomato_diseases") |
| created_at, updated_at | timestamp | Auditoria                                                             |

#### 2.1.5 Histórico de sensores (série temporal)

| Campo     | Tipo      | Descrição                           |
| --------- | --------- | ----------------------------------- |
| id        | PK        | Identificador único                 |
| device_id | FK        | Dispositivo que enviou              |
| ts        | timestamp | Momento da leitura                  |
| temp_c    | float     | Temperatura (°C)                    |
| hum_pct   | float     | Umidade ar (%)                      |
| soil_pct  | float     | Umidade solo (%) ou raw normalizado |
| soil_raw  | int       | Valor bruto ADC (opcional)          |
| lux_raw   | int       | Luminosidade bruta (opcional)       |
| dht_ok    | boolean   | Se DHT leu com sucesso              |

Amostragem: um registro a cada X minutos (ex.: 5–15 min) para não inflar o banco; pode vir de um job que consome `/api/dashboard` ou `/api/sensors` e persiste.

#### 2.1.6 Eventos de irrigação

| Campo                  | Tipo      | Descrição              |
| ---------------------- | --------- | ---------------------- | ---------- | ------ | ----- |
| id                     | PK        | Identificador único    |
| device_id              | FK        | Dispositivo            |
| started_at             | timestamp | Início                 |
| duration_ms            | int       | Duração em ms          |
| triggered_by           | enum      | "manual"               | "schedule" | "rule" | "api" |
| rule_id ou schedule_id | FK        | Opcional, se aplicável |
| created_at             | timestamp | Auditoria              |

#### 2.1.7 Resultados de inferência

| Campo           | Tipo      | Descrição                                               |
| --------------- | --------- | ------------------------------------------------------- |
| id              | PK        | Identificador único                                     |
| device_id       | FK        | Dispositivo                                             |
| ts              | timestamp | Momento da inferência                                   |
| predicted_class | string    | Classe retornada pelo modelo                            |
| confidence      | float     | Score (0–1)                                             |
| topk            | JSON      | Top-k classes/scores (opcional)                         |
| context         | JSON      | Opcional: temp_c, hum_pct, soil_pct, pump_on no momento |
| image_stored    | boolean   | Se a imagem foi guardada (política de privacidade)      |
| image_ref       | string    | Path ou blob ID, se image_stored=true                   |

Decisão de produto: por padrão **não** armazenar imagens; só metadados. Opção “guardar imagens para diagnóstico” pode ser ativável por usuário.

#### 2.1.8 Regras e agendamentos (opcional, v2)

| Campo                  | Tipo      | Descrição                                                                                                                  |
| ---------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------- | ----------- |
| id                     | PK        | Identificador único                                                                                                        |
| device_id / garden_id  | FK        | Escopo                                                                                                                     |
| type                   | enum      | "schedule"                                                                                                                 | "threshold" |
| config                 | JSON      | Ex.: schedule = { "cron": "0 7 \* \* \*", "duration_ms": 2000 }; threshold = { "soil_below_pct": 30, "duration_ms": 1500 } |
| active                 | boolean   | Se a regra está ativa                                                                                                      |
| created_at, updated_at | timestamp | Auditoria                                                                                                                  |

#### 2.1.9 Alertas e notificações (opcional)

| Campo                 | Tipo      | Descrição                                        |
| --------------------- | --------- | ------------------------------------------------ | --------------------- | -------------- | ---------------- |
| id                    | PK        | Identificador único                              |
| device_id / garden_id | FK        | Escopo                                           |
| kind                  | enum      | "disease_detected"                               | "sensor_out_of_range" | "pump_failure" | "device_offline" |
| payload               | JSON      | Detalhes (classe detectada, sensor, valor, etc.) |
| sent_at               | timestamp | Quando foi enviado (email/push)                  |
| created_at            | timestamp | Quando o evento ocorreu                          |

### 2.2 Escolha de tecnologia do banco

- **Single-user / local (PC na rede):** SQLite (arquivo único, backup simples, zero servidor).
- **Multi-user / nuvem ou vários acessos:** PostgreSQL (ou MySQL) com backend API (ex.: Flask/FastAPI) que o painel e os ESPs consomem.
- **Série temporal com muito volume:** considerar downsampling (agregar por hora/dia) ou tabela separada em DB de séries (ex.: InfluxDB, TimescaleDB), dependendo da escala.

### 2.3 Fluxo de dados

- **ESP → Backend:**
  - Backend faz polling periódico ao ESP (`/api/dashboard` ou `/api/sensors` + `/api/irrigation`) e grava em `sensor_readings` e, se houver evento novo, em `irrigation_events`.
  - Ou: ESP envia POST para um endpoint do backend (ex.: `/api/ingest/sensors`, `/api/ingest/inference`) quando há nova leitura ou após cada inferência.
- **Inferência:**
  - Backend chama o servidor ML (como hoje); resultado é salvo em `inference_results` (e opcionalmente imagem).
- **Painel:**
  - Lê do backend (dispositivos, jardins, cultivos, histórico, alertas) e envia comandos ao backend ou diretamente ao ESP (conforme arquitetura escolhida: proxy no backend ou chamada direta ao ESP com CORS).

---

## 3. Aumentar quantidade de cultivos e parâmetros por cultivo

### 3.1 Objetivo

- O usuário escolhe o **tipo de cultivo** (tomate, alface, manjericão, etc.).
- O sistema aplica **parâmetros ideais** para aquele cultivo (temperatura, umidade de solo, duração da bomba, cooldown).
- Opcionalmente: modelo de ML ou conjunto de classes adequado ao cultivo (ex.: doenças de tomate vs doenças de folhas de alface).

### 3.2 Parâmetros sugeridos por cultivo (ajustáveis)

| Parâmetro                | Descrição                                                | Ex. tomate | Ex. alface                |
| ------------------------ | -------------------------------------------------------- | ---------- | ------------------------- |
| temp_min_c, temp_max_c   | Faixa ideal de temperatura (°C)                          | 18–28      | 15–22                     |
| soil_moisture_target_pct | Umidade de solo “ideal” (%)                              | 40–60      | 50–70                     |
| soil_dry_threshold_pct   | Abaixo disso: acionar irrigação (se regra por threshold) | 35         | 45                        |
| soil_wet_threshold_pct   | Acima disso: não irrigar (evitar encharcamento)          | 70         | 75                        |
| irrigation_duration_ms   | Tempo que a bomba fica ligada por ciclo                  | 1500       | 1200                      |
| irrigation_cooldown_ms   | Intervalo mínimo entre ciclos                            | 120000     | 90000                     |
| light_preference         | "low" \| "medium" \| "high"                              | medium     | medium                    |
| ml_model_or_path         | Endpoint ou modelo (ex.: "/predict" para tomate)         | /predict   | /predict_lettuce (futuro) |

Esses valores podem ficar em `crop_types.default_params` e ser sobrescritos em `cultivos.params` por jardim/dispositivo.

### 3.3 Onde aplicar os parâmetros

- **No ESP:** hoje o firmware usa `pump_on_ms`, `pump_cooldown_ms` e lógica fixa. Para multi-cultivo:
  - **Opção A:** o painel (ou backend) envia `POST /api/config` com `pump_on_ms` e `pump_cooldown_ms` atualizados quando o usuário troca de cultivo ou edita parâmetros. O ESP não precisa conhecer “cultivo”, só os números.
  - **Opção B:** o ESP passa a aceitar um “perfil” por nome (ex.: `crop_profile: "tomato"`) e o backend/painel mantém a tabela perfil → parâmetros; ao escolher perfil, o painel envia os mesmos `POST /api/config` com os valores daquele perfil.
- **Regras “irrigar quando solo < X%”:** precisam rodar no backend (ou em um serviço no PC) que lê o último `soil_pct` do banco (ou do ESP), compara com `soil_dry_threshold_pct` do cultivo ativo e chama `POST {esp}/api/irrigation/start` com `duration_ms` do cultivo.

### 3.4 Modelo de ML e cultivos

- **Curto prazo:** um único modelo (ex.: doenças de folha de tomate); todos os cultivos usam o mesmo endpoint; parâmetros de irrigação/sensores é que mudam por cultivo.
- **Médio prazo:** vários modelos ou um modelo multi-classe (tomate + alface + …); em `crop_types` ou `cultivos` guardar `ml_model_hint` / `infer_path` (ex.: `/predict`, `/predict_lettuce`) e o painel/backend usa esse path ao chamar o servidor de inferência ou ao dizer ao ESP qual path usar em `POST /api/inference/config`.

---

## 4. Brainstorm completo de implementações

### 4.1 Setup e onboarding

- Assistente de primeiro uso: conectar ESP na rede (AP mode ou digitar WiFi), depois configurar URL do servidor de inferência e escolher primeiro cultivo.
- Lista de hardware e montagem (documentação + vídeo): ESP32-S3, câmera OV2640, DHT22, sensor de solo, relé, alimentação.
- Embalagem “kit”: checklist de peças, link para firmware pré-compilado ou OTA.

### 4.2 Backend e banco de dados

- API REST (Flask/FastAPI) com os recursos: devices, gardens, crop_types, cultivos, sensor_readings, irrigation_events, inference_results.
- Migrations (Alembic ou similar) para schema do banco.
- Ingestão: endpoints que o painel ou um daemon chama para enviar dados do ESP (sensors, inference); ou daemon que faz polling ao ESP e persiste.
- Backup/restore do banco (export SQLite/PostgreSQL, ou dump JSON para usuário avançado).

### 4.3 Painel (UX)

- Painel remoto (ver `docs/remote-panel.md`): base URL configurável por dispositivo; mesma funcionalidade do painel embarcado (bomba, câmera, inferência, config).
- Seletor de cultivo: dropdown “Tipo de cultivo” com presets (tomate, alface, …) e opção “Personalizado” para editar parâmetros.
- Gráficos: temperatura, umidade de solo e irrigação ao longo do tempo (dados do banco).
- Lista de dispositivos: status (online/offline), última vez visto, link para config e para stream.
- Notificações in-app ou por email: “Doença detectada: Late blight”, “Umidade do solo muito baixa”, “ESP desconectado há 1 hora”.

### 4.4 Regras e automação

- Regras de irrigação por threshold: “Se umidade solo < X%, ligar bomba por Y ms” (rodar no backend ou no ESP se no futuro o firmware tiver engine de regras).
- Agendamento: “Irrigar todo dia às 7h por Z ms” (cron no backend chamando `POST /api/irrigation/start` no ESP).
- Limites de segurança: tempo máximo de bomba ligada por dia; alerta se sensor de solo falhar.

### 4.5 Multi-cultivo e parâmetros

- Catálogo de `crop_types` com `default_params` (temperatura, solo, bomba, cooldown).
- Tela “Editar parâmetros” do cultivo ativo (salva em `cultivos.params` e envia ao ESP via `POST /api/config` quando for o dispositivo ativo).
- Opcional: sugestões por zona climática (ex.: “Tomate em clima subtropical”) como variante de presets.

### 4.6 ML e inferência

- Manter duas APIs (clássico + MobileNet) e comparativo (já planejado); no produto, escolher qual é a “principal” ou permitir A/B por dispositivo.
- Associar cultivo ao endpoint/path de inferência (ex.: tomate → `/predict`, alface → path futuro).
- Opção “não enviar imagens ao servidor” (inferência só local no futuro, ou desativar inferência): respeitar privacidade.

### 4.7 Firmware e dispositivos

- OTA (Over-The-Air): atualização de firmware pela rede (painel ou backend dispara, ESP baixa e reinicia).
- Identificação estável do dispositivo: `device_id` em NVS ou gerado uma vez (UUID) para associar ao banco.
- Heartbeat: ESP envia periodicamente “estou vivo” ao backend (ou backend faz health check); atualizar `last_seen_at` e gerar alerta “dispositivo offline”.

### 4.8 Segurança e privacidade

- Não armazenar imagens por padrão; só metadados da inferência; opção explícita “guardar imagens para diagnóstico”.
- Se houver usuários: autenticação (login/senha ou OAuth); API com API key ou JWT para painel e ingestão.
- HTTPS no painel remoto e no backend; ESP em rede local pode seguir HTTP (ou HTTPS com certificado autoassinado se necessário).

### 4.9 Integrações e extensibilidade

- Webhooks: ao detectar doença ou regra disparada, chamar URL configurável (ex.: IFTTT, Zapier).
- Export: CSV/PDF de histórico de sensores e irrigação (para usuário ou pesquisa).
- Integração opcional com Home Assistant / MQTT: publicar sensores e permitir acionar bomba por HA.

### 4.10 Documentação e produto

- Manual do usuário: configuração do kit, primeiro cultivo, leitura de gráficos e alertas.
- Documentação de API do backend (OpenAPI) para quem quiser integrar.
- `docs/remote-panel.md`: painel remoto (já sugerido).
- Este roadmap em `docs/product-roadmap.md` (ou `roadmap-produto.md`) com priorização (fase 1: banco + multi-cultivo básico; fase 2: regras e notificações; fase 3: OTA, multi-usuário, integrações).

---

## 5. Priorização sugerida (fases)

| Fase  | Escopo              | Entregas                                                                                                                                                                      |
| ----- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | Dados e cultivo     | Banco (dispositivos, jardins, sensores, irrigação, inferência); catálogo de cultivos com parâmetros; painel remoto com base URL; aplicar parâmetros ao ESP via `/api/config`. |
| **2** | Automação e alertas | Regras por threshold e agendamento; notificações (in-app ou email); gráficos de histórico.                                                                                    |
| **3** | Produto e escala    | OTA, multi-usuário, documentação de produto, opção de não armazenar imagens, integrações (webhook, export).                                                                   |

---

## 6. Referências no repositório

- API do ESP: `firmware/smart-tomato-garden/src/camera_server.cpp`
- Painel remoto: `docs/remote-panel.md`
- Config e irrigação no firmware: `firmware/smart-tomato-garden/src/config.cpp`, `irrigation.cpp`
- Inferência: `firmware/smart-tomato-garden/src/inference_client.cpp`
- ML e APIs: `ml/README.md`, `docs/api.md`
