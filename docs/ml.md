# ML

## Pipeline clássico (features + ML)
1) Segmentação HSV
2) Extração 102 features:
- Haralick (13)
- Zernike (25)
- Hist HSV (48)
- LBP mean/std (2)
- Hu (7)
- Mean HSV (3)
- Std HSV (3)
- Morfologia (1)

3) Treino com RF/SVM/XGBoost
4) Seleção automática do melhor modelo por macro-F1
5) Export:
- modelo_tomate.pkl
- label_encoder.pkl
- training_metadata.json
- feature_schema.json

## Rigor científico aplicado (revisores)
- Sem vazamento: split por grupo e teste só com originais
- CV por grupo (evita variantes do mesmo original entre folds)
- Métricas completas:
  - accuracy, macro/weighted F1, balanced acc, MCC
  - por classe: precision/recall/F1/support
  - matriz confusão
- Latência:
  - decode/features/predict/total
- Robustez a iluminação:
  - variações controladas (gamma, brilho, contraste)
  - comparação com normalização (Gray-World + CLAHE)
- Baseline DL opcional:
  - MobileNetV3-Small (transfer learning)
  - compara precisão e trade-offs

## Observabilidade
- Prometheus `/metrics`
- logs estruturados (loguru)
- MLflow opcional para rastrear experimentos
