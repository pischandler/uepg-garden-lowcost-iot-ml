# Architecture

## Visão geral
O projeto é dividido em 3 camadas:

1) **Firmware (ESP32-S3 + OV2640 + sensores/atuadores)**
- Captura imagens e mede variáveis ambientais.
- Exponde endpoints HTTP (ex.: `/capture`) para entrega da imagem e telemetria.

2) **ML (Servidor)**
- Pipeline offline: augmentação → extração 102 features → treino (RF/SVM/XGB) → avaliação (métricas + robustez a iluminação).
- Pipeline online: API recebe imagem (upload) ou busca a imagem via URL (puxando do ESP32) e retorna diagnóstico.

3) **Consumo/Integração**
- Ferramentas/automação para coleta e avaliação (scripts).
- Futuro: dashboard, banco, orquestração e alertas.

## Fluxos principais

### Fluxo A: Treino offline
1. `tools/dataset/` organiza dataset bruto
2. `garden-ml-augment` gera `dataset_aug/` + manifest/config
3. `garden-ml-train` faz split por **group_id**:
   - Treino: orig + aug
   - Teste: apenas orig
4. CV por grupo para evitar vazamento
5. Exporta artefatos em `ml/artifacts/model_registry/vXXXX/`

### Fluxo B: Inferência online (recomendado)
1. Servidor chama `POST /analisar_url`
2. API faz `GET http://esp32/capture`
3. Extrai features usando a mesma implementação do treino
4. Executa modelo (predict_proba) e retorna:
   - classe, score, topk
   - timings (decode/features/predict/total)
   - metadados (device_id, normalize etc.)

### Fluxo C: Inferência upload
1. Cliente faz `POST /analisar` (multipart)
2. API processa bytes e retorna resposta do mesmo formato

## Padrões de engenharia aplicados
- Fonte única da verdade: extração de features em `garden_ml/features/*`
- Schema explícito e verificável: `feature_schema.json`
- Reprodutibilidade: seed e split por grupos
- Avaliação ampla: macro-F1, weighted-F1, balanced acc, MCC, matriz confusão, latência
- Robustez: experimento de sensibilidade à iluminação + normalização fotométrica
- Observabilidade: Prometheus `/metrics`

## Evoluções futuras
- Persistência: SQLite/PostgreSQL para histórico e rastreabilidade
- Autenticação: API key por device
- DL edge: baseline TinyML/TFLite ou quantização
