# API

Base URL: `http://<host>:<port>`

## GET /health
Retorna classes carregadas e sanity do serviço.

Resposta:
```json
{
  "ok": true,
  "classes": ["Tomato_healthy", "..."],
  "artifacts_dir": "artifacts/model_registry/v0001"
}
GET /metrics
Endpoint Prometheus.

POST /analisar
Upload de imagem.

Content-Type: multipart/form-data

Campo: image

Query params:

normalize=1 (opcional) ativa normalização fotométrica

Headers:

X-Device-Id (opcional)

Resposta:

{
  "classe_predita": "Tomato_healthy",
  "score": 0.93,
  "topk": [{"classe":"...","score":0.93}],
  "timings_ms": {"decode_ms":1.2,"features_ms":12.0,"predict_ms":0.5,"total_ms":13.7},
  "meta": {"device_id":"...", "photometric_normalize":true}
}
POST /analisar_url
Servidor busca a imagem direto do ESP32 (modelo recomendado).

Body JSON:

{
  "url": "http://192.168.0.10/capture",
  "device_id": "stg-001",
  "normalize": true
}
Resposta no mesmo formato do /analisar.

Observação:
Se o firmware incluir headers como:

X-Lux-Raw

X-Soil-Raw

X-Temp-C

X-Hum-Pct

eles serão repassados no meta para rastreabilidade e análises.
