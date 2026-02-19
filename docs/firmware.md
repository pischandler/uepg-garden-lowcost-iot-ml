# Firmware

## Responsabilidades
- Capturar imagem via OV2640
- Ler sensores (solo, luz, temp/umidade)
- Controlar atuadores (bomba/ventilação etc.)
- Expor endpoints HTTP

## Endpoint mínimo recomendado
### GET /capture
- Retorna `image/jpeg` como body
- Inclui telemetria em headers HTTP para rastreabilidade

Headers sugeridos:
- `X-Device-Id: stg-001`
- `X-Lux-Raw: <int>`
- `X-Soil-Raw: <int>`
- `X-Soil-Pct: <float>`
- `X-Temp-C: <float>`
- `X-Hum-Pct: <float>`

## Integração recomendada
O servidor ML deve chamar:
- `POST /analisar_url` com `url=http://esp32/capture`

Vantagens:
- Firmware não precisa implementar multipart/form-data
- Falhas/retries ficam no servidor
- Facilita logging e controle

## Considerações práticas
- Garantir estabilidade do Wi-Fi
- Padronizar iluminação (ou LED fixo) quando possível
- Se máscara (segmentação) falhar frequentemente, ativar normalização no servidor
