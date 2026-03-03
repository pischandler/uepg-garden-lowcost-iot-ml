# Referência rápida — Comandos e caminhos (Smart Tomato Garden)

Use este arquivo como cheatsheet de comandos, variáveis e caminhos do projeto.

## Make (raiz do repo)

| Alvo | Descrição |
|------|-----------|
| `make venv` | Cria/usa .venv e instala ml com [dev,reports,tracking] |
| `make ml-test` | Roda pytest em ml/ |
| `make ml-augment-default` | Augment com defaults (LOCAL_DATASET_RAW → LOCAL_DATASET_AUG) |
| `make ml-train-default` | Treino com defaults (LOCAL_DATASET_AUG → LOCAL_ARTIFACTS_DIR) |
| `make ml-eval-default` | Avaliação com defaults |
| `make ml-serve` | Sobe API Flask (artifacts em LOCAL_ARTIFACTS_DIR) |
| `make ml-pipeline` | ml-test + augment + train + eval (defaults) |
| `make ml-run-all` | ml-pipeline + servidor em foreground |
| `make format` | ruff format no ml/ |
| `make lint` | ruff check no ml/ |
| `make docker-build` | Build das imagens Docker |
| `make docker-pipeline` | Sanity + testes + augment + train + eval no container |
| `make docker-up` | Sobe API em background (Docker) |
| `make docker-down` | Derruba stack Docker |

Variáveis úteis (Make): `LOCAL_DATASET_RAW`, `LOCAL_DATASET_AUG`, `LOCAL_ARTIFACTS_DIR`, `IMG_SIZE`, `AUG_PER_IMAGE`, `SEED`, `TEST_SIZE`, `CV_FOLDS`, `MANIFEST`. Valores padrão no `Makefile`.

## ML — Comandos diretos (com venv ativo)

```bash
cd ml
pip install -e ".[dev,reports,tracking]"   # + [viz] para gráficos na avaliação
garden-ml-augment --input_dir <raw> --output_dir <aug> --img_size 128 --aug_per_image 5 --seed 42
garden-ml-train   --dataset_dir <aug> --output_dir <artifacts_dir> --img_size 128 --seed 42 --cv_folds 5
garden-ml-eval    --dataset_dir <aug> --artifacts_dir <artifacts_dir> --img_size 128
garden-ml-serve   --artifacts_dir <artifacts_dir> --host 0.0.0.0 --port 5000
```

Scripts de entrada em `ml/scripts/`: `augment_dataset.py`, `train_model.py`, `evaluate_model.py`, `serve_api.py` (encaminham para os comandos acima).

## Caminhos importantes

| Conceito | Caminho típico |
|----------|----------------|
| Artefatos de modelo | `ml/artifacts/model_registry/v0004/` (ou vXXXX) |
| Dataset augmentado | `ml/dataset/.../aug` ou variável `LOCAL_DATASET_AUG` |
| Dataset bruto | `ml/dataset/.../raw` ou `LOCAL_DATASET_RAW` |
| Avaliação (JSON/CSV/PNG) | `ml/artifacts/model_registry/vXXXX/evaluation/` |
| Config firmware (exemplo) | `firmware/smart-tomato-garden/include/secrets.example.h` |
| Config inferência no ESP | Web: card "Configuração do servidor de inferência"; CLI: `firmware/.../tools/set_inference.py` |

## Firmware (ESP32-S3)

- **Build/upload**: `pio run`, `pio run -t upload` (dentro de `firmware/smart-tomato-garden/`).
- **Configurar inferência via CLI**: `python firmware/smart-tomato-garden/tools/set_inference.py --esp <IP_ESP> --ml <IP_ML> [--port 5000] [--path /predict]`.

## Testes e qualidade (ML)

- Testes: `make ml-test` ou `pytest` no venv (raiz do repo com venv ativo, ou de dentro de `ml/`).
- Formatação: `make format` (ruff format).
- Lint: `make lint` (ruff check).
