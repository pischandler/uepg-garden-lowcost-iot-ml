# garden-ml — Pipeline ML do Smart Tomato Garden

Pacote Python para **augmentação**, **treino**, **avaliação** e **API de inferência** na detecção de doenças em folhas de tomate. Uma única extração de **188 features** é usada em treino e inferência; **não** é aplicada normalização fotométrica.

---

## Visão geral

- **Augmentação:** Albumentations; manifest e config exportados.
- **Features:** 188 dimensões (Haralick, Zernike, histogramas HSV, LBP, Hu, mean/std HSV, forma, LAB, chroma). Schema em `garden_ml/features/schema.py` e `feature_schema.json`.
- **Modelos:** Random Forest, SVM, XGBoost; seleção por macro F1 em CV por grupo (StratifiedGroupKFold). Scaler global (StandardScaler) aplicado antes dos modelos quando o artefato for pipeline (scaler+model).
- **Avaliação:** macro F1, balanced accuracy, ECE, matriz de confusão, sensibilidade à iluminação. Com dependência opcional `[viz]`: gera `calibration_curve.png` e `confusion_matrix.png`.
- **API:** Flask com `/health`, `/metrics` (Prometheus), upload de imagem e análise por URL.

**Pré-processamento (contrato fixo):** Apenas **segmentação HSV** e **redimensionamento** (letterbox para `img_size`, ex.: 128). O modelo não deve ser usado com normalização de imagem; inferência deve usar o mesmo `img_size` do treino.

---

## Instalação

Na pasta `ml/` (ou a partir da raiz com `pip install -e ./ml`):

```bash
pip install -e .
```

Opcionais:

```bash
pip install -e ".[viz]"      # Gráficos na avaliação (calibração, matriz de confusão)
pip install -e ".[dl]"      # Baseline deep learning (MobileNetV3-Small)
pip install -e ".[reports]" # Pandas para relatórios (já usado por train/eval)
pip install -e ".[tracking]"# MLflow
pip install -e ".[dev]"     # pytest, ruff, mypy, etc.
```

---

## Pipeline em sequência

1. **Augmentar** o dataset bruto (saída: pasta augmentada + manifest + config).
2. **Treinar** com split por grupo (treino: orig+aug; teste: só orig); CV por grupo; artefatos em `model_registry/vXXXX/`.
3. **Avaliar** no conjunto de teste e gerar métricas + sensibilidade à iluminação (e, com `[viz]`, PNGs).
4. **Servir** a API apontando para o diretório de artefatos.

---

## Comandos

Todos assumem que o pacote está instalado (`pip install -e .`). Na raiz do repo pode-se usar `make ml-*` (ver [SKILLS.md](../SKILLS.md)).

### Augmentação

```bash
garden-ml-augment --input_dir <dataset_bruto> --output_dir <dataset_aug> \
  --img_size 128 --aug_per_image 5 --seed 42
```

Opção `--segment_before_aug` aplica segmentação antes da augmentação.

### Treino

```bash
garden-ml-train --dataset_dir <dataset_aug> --output_dir <artifacts_dir> \
  --img_size 128 --test_size 0.30 --seed 42 --cv_folds 5 --manifest augmentation_manifest.csv
```

Artefatos gerados: `modelo_tomate.pkl`, `label_encoder.pkl`, `training_metadata.json`, `feature_schema.json`, manifests de treino/teste, comparação de modelos.

### Avaliação

```bash
garden-ml-eval --dataset_dir <dataset_aug> --artifacts_dir <artifacts_dir> \
  --img_size 128 --manifest augmentation_manifest.csv
```

Saída em `artifacts_dir/evaluation/`: `eval_trained.json`, `eval_base.json`, `per_class_base.csv`, `illumination_sensitivity.json`. Com `[viz]`: `calibration_curve.png`, `confusion_matrix.png`.

### Servir API

```bash
garden-ml-serve --artifacts_dir <artifacts_dir> --host 0.0.0.0 --port 5000
```

### Baseline DL (opcional, requer `.[dl]`)

```bash
garden-ml-deep --dataset_dir <dataset_aug> --output_dir <artifacts_dl_dir> \
  --img_size 224 --seed 42 --epochs 10
```

---

## Endpoints da API

| Método | Rota           | Descrição |
|--------|----------------|-----------|
| GET    | `/health`      | Classes carregadas e sanidade do serviço |
| GET    | `/metrics`     | Métricas Prometheus |
| POST   | `/analisar`    | Upload de imagem (multipart: `image`) |
| POST   | `/analisar_url`| Body JSON: `{"url":"http://esp/capture","device_id":"..."}` — servidor busca a imagem |

Resposta típica: `classe_predita`, `score`, `topk`, `timings_ms`, `meta`. Detalhes: [docs/api.md](../docs/api.md).

---

## Contrato de dados e pré-processamento

- **Split por grupo:** Treino e teste não compartilham o mesmo grupo (ex.: mesma imagem original); evita vazamento por augmentação.
- **Duplicatas:** Originais que caem no teste por hash são removidos do teste (registrado em `leakage_check.json`).
- **Pré-processamento:** Sem normalização fotométrica; apenas segmentação (HSV) e redimensionamento. Inferência deve usar o mesmo `img_size` e nenhuma normalização.
- **Artefatos:** O `training_metadata.json` registra `img_size`, `photometric_normalize: false` e, quando aplicável, uso de scaler/pipeline.

---

## Testes e qualidade

```bash
# Na raiz do repo (com venv ativo)
make ml-test

# Ou dentro de ml/
pytest
```

Formatação e lint: `make format`, `make lint` (ruff).

---

## Estrutura do pacote

```
ml/
├── src/garden_ml/
│   ├── config/       # constants, settings, logging
│   ├── data/         # manifest, splits, io
│   ├── features/     # extract, schema, components
│   ├── image/        # segmentation, resize, photometric (não usado na pipeline atual)
│   ├── training/     # train, evaluate, augment, deep_baseline
│   ├── inference/    # predictor, api_flask, serializers, storage
│   └── viz/          # plots (curva de calibração, matriz de confusão; requer [viz])
├── tests/
├── scripts/          # Pontos de entrada que chamam os comandos garden-ml-*
└── pyproject.toml
```

---

## Referências

- [README principal](../README.md) — visão do projeto e início rápido
- [docs/architecture.md](../docs/architecture.md) — fluxos e camadas
- [docs/api.md](../docs/api.md) — API de inferência
- [SKILLS.md](../SKILLS.md) — comandos e caminhos de referência
