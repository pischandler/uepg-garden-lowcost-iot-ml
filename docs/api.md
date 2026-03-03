# API de inferência

Base URL: `http://<host>:<port>`

A API usa o mesmo pré-processamento do treino (segmentação HSV + redimensionamento); não é aplicada normalização fotométrica.

---

## GET /health

Retorna classes carregadas e sanidade do serviço.

**Resposta exemplo:**

```json
{
  "ok": true,
  "classes": ["Tomato_healthy", "Tomato_Late_blight", "..."],
  "artifacts_dir": "artifacts/model_registry/v0001",
  "model_img_size": 128
}
```

---

## GET /metrics

Métricas no formato Prometheus (contadores, histogramas de latência, etc.).

---

## POST /analisar

Upload de imagem para análise.

- **Content-Type:** `multipart/form-data`
- **Campo:** `image` (arquivo)
- **Headers opcionais:** `X-Device-Id` para rastreabilidade

**Resposta exemplo:**

```json
{
  "classe_predita": "Tomato_healthy",
  "score": 0.93,
  "topk": [{"classe": "Tomato_healthy", "score": 0.93}, "..."],
  "timings_ms": {"decode_ms": 1.2, "features_ms": 12.0, "predict_ms": 0.5, "total_ms": 13.7},
  "meta": {"device_id": "...", "model_img_size": 128}
}
```

---

## POST /analisar_url

O servidor busca a imagem no ESP (ou em qualquer URL) e analisa. Modo recomendado quando o cliente é o próprio servidor chamando o ESP.

- **Content-Type:** `application/json`
- **Body exemplo:**

```json
{
  "url": "http://192.168.0.10/capture",
  "device_id": "stg-001"
}
```

Resposta no mesmo formato do `/analisar`.

Se o firmware enviar headers como `X-Lux-Raw`, `X-Soil-Raw`, `X-Temp-C`, `X-Hum-Pct` na resposta de `/capture`, eles podem ser repassados no `meta` para rastreabilidade.
