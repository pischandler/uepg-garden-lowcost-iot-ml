# Schema de banco de dados — Smart Tomato Garden (produto)

Este documento descreve o schema recomendado para persistência de dados quando o projeto for evoluído para um produto reprodutível (hortas domésticas, controle completo, múltiplos cultivos). Referência: [product-roadmap.md](product-roadmap.md).

---

## Visão geral

- **Dispositivos (ESPs)** e **jardins** como entidades principais.
- **Cultivos** vinculados a jardim ou dispositivo, com **parâmetros por tipo de planta**.
- **Série temporal**: leituras de sensores e eventos de irrigação.
- **Inferências**: resultados de ML (sem imagem por padrão).
- **Regras/agendamentos** e **alertas** para fases futuras.

Sugestão de tecnologia: **SQLite** para uso local/single-user; **PostgreSQL** para multi-usuário ou serviço em nuvem.

---

## Entidades e atributos

### 1. `gardens` (jardins / hortas)

| Coluna         | Tipo      | Restrição | Descrição                 |
| -------------- | --------- | --------- | ------------------------- |
| id             | INTEGER   | PK        | Identificador único       |
| name           | TEXT      | NOT NULL  | Nome do jardim            |
| user_id        | INTEGER   | FK, NULL  | Dono (se multi-usuário)   |
| timezone       | TEXT      |           | Fuso para agendamentos    |
| location_notes | TEXT      |           | Interior/exterior, cidade |
| created_at     | TIMESTAMP |           | Auditoria                 |
| updated_at     | TIMESTAMP |           | Auditoria                 |

### 2. `devices` (dispositivos ESP)

| Coluna           | Tipo      | Restrição | Descrição                       |
| ---------------- | --------- | --------- | ------------------------------- |
| id               | INTEGER   | PK        | Identificador único             |
| device_id        | TEXT      | UNIQUE    | ID lógico (ex.: "stg-01")       |
| name             | TEXT      |           | Nome amigável ("Horta varanda") |
| base_url         | TEXT      |           | URL base (http://192.168.1.50)  |
| firmware_version | TEXT      |           | Versão do firmware              |
| garden_id        | INTEGER   | FK        | Jardim ao qual pertence         |
| last_seen_at     | TIMESTAMP |           | Última comunicação              |
| config_snapshot  | TEXT/JSON |           | Última config NVS (opcional)    |
| created_at       | TIMESTAMP |           | Auditoria                       |
| updated_at       | TIMESTAMP |           | Auditoria                       |

### 3. `crop_types` (catálogo de tipos de cultivo)

| Coluna         | Tipo      | Restrição | Descrição                                |
| -------------- | --------- | --------- | ---------------------------------------- |
| id             | INTEGER   | PK        | Identificador único                      |
| code           | TEXT      | UNIQUE    | Código estável ("tomato", "lettuce")     |
| display_name   | TEXT      | NOT NULL  | Nome para UI                             |
| default_params | TEXT/JSON |           | Parâmetros padrão (ver seção Parâmetros) |
| ml_model_hint  | TEXT      |           | Endpoint/modelo de inferência (opcional) |
| created_at     | TIMESTAMP |           | Auditoria                                |
| updated_at     | TIMESTAMP |           | Auditoria                                |

### 4. `cultivos` (cultivo ativo por jardim ou dispositivo)

| Coluna       | Tipo      | Restrição | Descrição                                  |
| ------------ | --------- | --------- | ------------------------------------------ |
| id           | INTEGER   | PK        | Identificador único                        |
| garden_id    | INTEGER   | FK, NULL  | Jardim (um de garden_id ou device_id)      |
| device_id    | INTEGER   | FK, NULL  | Dispositivo (um de garden_id ou device_id) |
| crop_type_id | INTEGER   | FK        | Tipo de cultivo                            |
| name         | TEXT      |           | Nome amigável                              |
| params       | TEXT/JSON |           | Parâmetros sobrescritos                    |
| active       | INTEGER   | DEFAULT 1 | 1 = ativo, 0 = encerrado                   |
| started_at   | DATE      |           | Início do cultivo                          |
| ended_at     | DATE      |           | Fim (opcional)                             |
| created_at   | TIMESTAMP |           | Auditoria                                  |
| updated_at   | TIMESTAMP |           | Auditoria                                  |

### 5. `sensor_readings` (histórico de sensores)

| Coluna    | Tipo      | Restrição | Descrição           |
| --------- | --------- | --------- | ------------------- |
| id        | INTEGER   | PK        | Identificador único |
| device_id | INTEGER   | FK        | Dispositivo         |
| ts        | TIMESTAMP | NOT NULL  | Momento da leitura  |
| temp_c    | REAL      |           | Temperatura (°C)    |
| hum_pct   | REAL      |           | Umidade ar (%)      |
| soil_pct  | REAL      |           | Umidade solo (%)    |
| soil_raw  | INTEGER   |           | Valor bruto ADC     |
| lux_raw   | INTEGER   |           | Luminosidade bruta  |
| dht_ok    | INTEGER   |           | 1 = DHT leu OK      |

Índice sugerido: `(device_id, ts)` para consultas por dispositivo e intervalo de tempo.

### 6. `irrigation_events` (eventos de irrigação)

| Coluna       | Tipo      | Restrição | Descrição                                 |
| ------------ | --------- | --------- | ----------------------------------------- |
| id           | INTEGER   | PK        | Identificador único                       |
| device_id    | INTEGER   | FK        | Dispositivo                               |
| started_at   | TIMESTAMP | NOT NULL  | Início                                    |
| duration_ms  | INTEGER   |           | Duração em ms                             |
| triggered_by | TEXT      |           | "manual" \| "schedule" \| "rule" \| "api" |
| rule_id      | INTEGER   | FK, NULL  | Regra que disparou (opcional)             |
| schedule_id  | INTEGER   | FK, NULL  | Agendamento (opcional)                    |
| created_at   | TIMESTAMP |           | Auditoria                                 |

### 7. `inference_results` (resultados de inferência)

| Coluna          | Tipo      | Restrição | Descrição                     |
| --------------- | --------- | --------- | ----------------------------- |
| id              | INTEGER   | PK        | Identificador único           |
| device_id       | INTEGER   | FK        | Dispositivo                   |
| ts              | TIMESTAMP | NOT NULL  | Momento da inferência         |
| predicted_class | TEXT      |           | Classe retornada              |
| confidence      | REAL      |           | Score (0–1)                   |
| topk            | TEXT/JSON |           | Top-k classes/scores          |
| context         | TEXT/JSON |           | temp_c, soil_pct, etc.        |
| image_stored    | INTEGER   | DEFAULT 0 | 1 = imagem guardada           |
| image_ref       | TEXT      |           | Path ou blob ID (se guardada) |
| created_at      | TIMESTAMP |           | Auditoria                     |

### 8. `rules` (regras e agendamentos — opcional, v2)

| Coluna     | Tipo      | Restrição | Descrição                                 |
| ---------- | --------- | --------- | ----------------------------------------- |
| id         | INTEGER   | PK        | Identificador único                       |
| device_id  | INTEGER   | FK, NULL  | Escopo dispositivo                        |
| garden_id  | INTEGER   | FK, NULL  | Escopo jardim                             |
| type       | TEXT      |           | "schedule" \| "threshold"                 |
| config     | TEXT/JSON |           | Ex.: cron + duration_ms ou soil_below_pct |
| active     | INTEGER   | DEFAULT 1 | 1 = ativa                                 |
| created_at | TIMESTAMP |           | Auditoria                                 |
| updated_at | TIMESTAMP |           | Auditoria                                 |

### 9. `alerts` (alertas / notificações — opcional)

| Coluna     | Tipo      | Restrição | Descrição                                                                         |
| ---------- | --------- | --------- | --------------------------------------------------------------------------------- |
| id         | INTEGER   | PK        | Identificador único                                                               |
| device_id  | INTEGER   | FK, NULL  | Dispositivo                                                                       |
| garden_id  | INTEGER   | FK, NULL  | Jardim                                                                            |
| kind       | TEXT      |           | "disease_detected" \| "sensor_out_of_range" \| "pump_failure" \| "device_offline" |
| payload    | TEXT/JSON |           | Detalhes do evento                                                                |
| sent_at    | TIMESTAMP |           | Quando foi enviado (email/push)                                                   |
| created_at | TIMESTAMP |           | Quando o evento ocorreu                                                           |

---

## Parâmetros de cultivo (`default_params` / `params`)

Objeto JSON sugerido (em `crop_types.default_params` e `cultivos.params`):

```json
{
  "temp_min_c": 18,
  "temp_max_c": 28,
  "soil_moisture_target_pct": 50,
  "soil_dry_threshold_pct": 35,
  "soil_wet_threshold_pct": 70,
  "irrigation_duration_ms": 1500,
  "irrigation_cooldown_ms": 120000,
  "light_preference": "medium",
  "ml_model_hint": "/predict"
}
```

````

- **temp_min_c / temp_max_c:** faixa ideal de temperatura (°C).
- **soil_dry_threshold_pct:** abaixo disso pode acionar irrigação (regra por threshold).
- **soil_wet_threshold_pct:** acima disso não irrigar.
- **irrigation_duration_ms / irrigation_cooldown_ms:** enviados ao ESP via `POST /api/config` (pump_on_ms, pump_cooldown_ms).
- **ml_model_hint:** path ou identificador do modelo de inferência (opcional).

---

## Exemplo SQL (SQLite)

```sql
CREATE TABLE gardens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  user_id INTEGER,
  timezone TEXT,
  location_notes TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE devices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL UNIQUE,
  name TEXT,
  base_url TEXT,
  firmware_version TEXT,
  garden_id INTEGER NOT NULL REFERENCES gardens(id),
  last_seen_at TEXT,
  config_snapshot TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE crop_types (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  default_params TEXT,
  ml_model_hint TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE cultivos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  garden_id INTEGER REFERENCES gardens(id),
  device_id INTEGER REFERENCES devices(id),
  crop_type_id INTEGER NOT NULL REFERENCES crop_types(id),
  name TEXT,
  params TEXT,
  active INTEGER DEFAULT 1,
  started_at TEXT,
  ended_at TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now')),
  CHECK ((garden_id IS NOT NULL) <> (device_id IS NOT NULL))
);

CREATE TABLE sensor_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER NOT NULL REFERENCES devices(id),
  ts TEXT NOT NULL,
  temp_c REAL,
  hum_pct REAL,
  soil_pct REAL,
  soil_raw INTEGER,
  lux_raw INTEGER,
  dht_ok INTEGER
);
CREATE INDEX idx_sensor_readings_device_ts ON sensor_readings(device_id, ts);

CREATE TABLE irrigation_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER NOT NULL REFERENCES devices(id),
  started_at TEXT NOT NULL,
  duration_ms INTEGER,
  triggered_by TEXT,
  rule_id INTEGER,
  schedule_id INTEGER,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE inference_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER NOT NULL REFERENCES devices(id),
  ts TEXT NOT NULL,
  predicted_class TEXT,
  confidence REAL,
  topk TEXT,
  context TEXT,
  image_stored INTEGER DEFAULT 0,
  image_ref TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER REFERENCES devices(id),
  garden_id INTEGER REFERENCES gardens(id),
  type TEXT,
  config TEXT,
  active INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id INTEGER REFERENCES devices(id),
  garden_id INTEGER REFERENCES gardens(id),
  kind TEXT,
  payload TEXT,
  sent_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
```

---

## Diagrama de relações (texto)

```
gardens (1) ----< (N) devices
    |                    |
    |                    +----< sensor_readings
    |                    +----< irrigation_events
    |                    +----< inference_results
    |                    +----< alerts
    |
    +----< cultivos >---- crop_types
    |
    +----< rules
    +----< alerts
```

---

## Referências

- Roadmap e justificativa dos dados: [product-roadmap.md](product-roadmap.md)
- API do ESP (origem dos dados): `firmware/smart-tomato-garden/src/camera_server.cpp`
- Ingestão sugerida: polling a `/api/dashboard` e `/api/sensors` por dispositivo; persistir em `sensor_readings` e `irrigation_events` conforme eventos
- Parâmetros aplicados ao ESP: `POST /api/config` com `pump_on_ms`, `pump_cooldown_ms` vindos de `cultivos.params` ou `crop_types.default_params`
````
