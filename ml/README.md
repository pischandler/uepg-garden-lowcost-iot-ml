# garden-ml

Pipeline ML do Smart Tomato Garden:
- Augmentação com Albumentations (manifest + config exportada)
- Extração de 102 features (única fonte usada por treino e inferência)
- Treino com RandomForest, SVM e XGBoost com CV por grupo (sem vazamento)
- Avaliação com métricas completas e análise de robustez a iluminação
- Baseline DL opcional (MobileNetV3-Small) para comparação
- API Flask com métricas Prometheus e latência detalhada

## Instalação
Dentro de `ml/`:

```bash
pip install -e .
# Opcional baseline DL
pip install -e ".[dl]"
Augmentar dataset
garden-ml-augment --input_dir dataset --output_dir dataset_aug --img_size 128 --aug_per_image 5 --seed 42
Treinar (features + modelos clássicos)
garden-ml-train --dataset_dir dataset_aug --output_dir artifacts/model_registry/v0001 --img_size 128 --seed 42
Avaliar (inclui robustez a iluminação + normalização)
garden-ml-eval --dataset_dir dataset_aug --artifacts_dir artifacts/model_registry/v0001 --img_size 128 --seed 42
Baseline DL (opcional)
garden-ml-deep --dataset_dir dataset_aug --output_dir artifacts/model_registry/v0001_dl --img_size 224 --seed 42 --epochs 10
Servir API
garden-ml-serve --artifacts_dir artifacts/model_registry/v0001 --host 0.0.0.0 --port 5000
Endpoints:

GET /health

GET /metrics

POST /analisar (multipart form-data: image=arquivo)

POST /analisar_url (json: {"url":"http://esp32/capture","device_id":"..."} )


---

# `ml/src/garden_ml/__init__.py`

```python
__all__ = ["__version__"]
__version__ = "0.1.0"
