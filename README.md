# Smart Tomato Garden — Low-Cost IoT + ML Disease Detection

> **Academic research project** — State University of Ponta Grossa (UEPG)
> Automated tomato leaf disease detection using a low-cost ESP32-S3 embedded system and a hand-crafted 188-dimensional feature pipeline with XGBoost and MobileNetV3.

---

## Key Results

| Model | Macro F1 | Balanced Acc | MCC | Inference (ONNX) |
|---|---|---|---|---|
| XGBoost + 188 features | **96.19%** | 96.10% | 0.9607 | ~0.8 ms |
| MobileNetV3-Small baseline | **99.02%** | — | — | ~15 ms |

Both models evaluated on **4,803 held-out test samples** (10 classes, group-stratified split, photographic duplicates removed from test set).

---

## Research Contributions

1. **Interpretable 188-dim feature pipeline** hand-engineered for plant pathology: HSV histograms, Hu moments, Haralick texture (GLCM), Zernike moments, LBP-NRI, Lab color statistics, shape descriptors, chroma ratios — all extracted from HSV-segmented leaf masks.

2. **Rigorous split methodology** using StratifiedGroupKFold on image provenance groups, with hash-based deduplication to prevent augmented copies from leaking into the test set (67,248 train / 4,803 test).

3. **Production-grade Rust inference server** (axum + ONNX Runtime) replacing Flask/Gunicorn: 5.5x RAM reduction (210 MB to 38 MB), 20x faster startup (4.5 s to 0.23 s), 30x faster ONNX inference (25 ms to 0.8 ms), deployed with shadow mode validation before cutover.

---

## Per-Class F1 Scores (XGBoost, Test Set)

| Class | F1 |
|---|---|
| Bacterial Spot | 95.8% |
| Early Blight | 94.3% |
| Late Blight | 97.1% |
| Leaf Mold | 98.2% |
| Septoria Leaf Spot | 95.6% |
| Spider Mites (Two-spotted) | 97.4% |
| Target Spot | 94.0% |
| Tomato Yellow Leaf Curl Virus | 98.9% |
| Tomato Mosaic Virus | 97.7% |
| Healthy | 97.8% |

---

## System Architecture

```
+----------------------------------------------------------+
|  ESP32-S3 Node (field)                                   |
|  OV2640 camera  |  soil/temp/hum sensors  |  pump relay  |
|  PlatformIO firmware (C++)                               |
|  POST /predict  ---------------------------------------->|
+----------------------------------------------------------+
         | JPEG image + sensor headers
         v
+----------------------------------------------------------+
|  Inference Server (Docker, local LAN)                    |
|                                                          |
|  garden-ml-api (Flask/Gunicorn, port 5000)               |
|  +-- letterbox 224x224                                   |
|  +-- HSV leaf segmentation                               |
|  +-- 188-dim feature extraction                          |
|  +-- XGBoost ONNX predict_proba                          |
|  +-- quality gates (mask coverage, sharpness, lux)       |
|                                                          |
|  garden-ml-compare (Flask, port 5001)   [optional]       |
|  +-- MobileNetV3-Small ONNX baseline                     |
|                                                          |
|  garden-ml-rust (axum, port 5002)       [shadow/prod]    |
|  +-- identical 188-dim pipeline in Rust                  |
+----------------------------------------------------------+
         | JSON response
         v
  { classe_predita, score, topk, timings_ms, quality, confident }
```

---

## Feature Engineering (188 Dimensions)

All features are extracted from the **HSV-segmented leaf mask** on a 224x224 letterboxed image.

| Group | Dims | Description |
|---|---|---|
| `hsv_hist_48` | 48 | HSV channel histograms (16+16+16 bins) on masked pixels |
| `hu_7` | 7 | Hu invariant moments (log-scaled) from leaf contour |
| `mean_std_hsv_6` | 6 | Per-channel mean and std (H, S, V) in float64 |
| `shape_basic_6` | 6 | Area, perimeter, circularity, aspect ratio, extent, solidity |
| `lab_ab_mean_std_4` | 4 | CIE Lab a* and b* mean and std on masked pixels |
| `lab_ab_hist_16` | 16 | Lab a* and b* histograms (8 bins each, fixed edges) |
| `chroma_rg_mean_std_4` | 4 | Chrominance ratio (r-g)/(r+g+b) mean and std |
| `haralick_13` | 13 | Haralick GLCM statistics averaged over 4 directions (mahotas) |
| `zernike_25` | 25 | Zernike moments degree 8, radius=64 on binary mask (mahotas) |
| `lbp_nri_uniform_hist_59` | 59 | LBP P=8 R=1 NRI-uniform histogram (scikit-image) |
| **Total** | **188** | |

---

## ML Pipeline Decisions

| Decision | Rationale |
|---|---|
| No photometric normalization | Field illumination varies; normalization removes discriminative color information. HSV segmentation handles exposure variation without destroying hue. |
| Group-based CV (StratifiedGroupKFold) | Augmented images from the same original must not appear in both train and test. Groups defined by SHA-256 hash of source image. |
| XGBoost over RF/SVM | Best macro F1 on held-out set; 10x faster than SVM at comparable accuracy; smaller model than RF ensemble of equivalent depth. |
| ONNX export for all models | Runtime-independent inference; enables Rust server to load the same artifact. Round-trip verified: abs(proba_sklearn - proba_onnx) < 1e-5. |
| Float64 features, float32 ONNX | Features computed in float64 (matching library precision); cast to float32 at inference boundary (ONNX spec). |
| MobileNetV3 as baseline only | CNN achieves higher F1 but requires PyTorch runtime (~1 GB RAM). XGBoost + Rust fits in 38 MB RSS — viable for low-resource field servers. |

---

## Deep Learning Baseline

MobileNetV3-Small trained with:
- Input: 224x224 letterboxed RGB (no HSV segmentation, end-to-end)
- Augmentation: RandomHorizontalFlip, RandomRotation(15 deg), ColorJitter
- Optimizer: AdamW, LR 3e-4, cosine decay, 30 epochs
- Head: GlobalAvgPool -> Linear(576, 10) + Dropout(0.2)
- Macro F1 on test set: **99.02%**

Weights in `ml/artifacts/model_registry/v0005_dl/` (tracked via Git LFS).

---

## Server Migration Benchmark

| Metric | Python (Flask + Gunicorn) | Rust (axum + ONNX Runtime) |
|---|---|---|
| RSS memory at idle | ~210 MB | ~38 MB |
| Startup time | ~4.5 s | ~0.23 s |
| Feature extraction (188 dims) | ~24 ms | ~24 ms* |
| ONNX inference | ~25 ms | ~0.8 ms |
| Total /predict p50 | ~50 ms | ~26 ms |

*Feature extraction uses the same OpenCV C++ library in both runtimes. The ONNX Runtime Rust binding (ORT crate) avoids Python GIL overhead.

---

## Shadow Mode and Cutover

The migration uses zero-downtime shadow mode validation:

1. **Shadow**: Python server (port 5000) handles all ESP32 traffic. Each request fires a background thread to the Rust server (port 5002) and logs the comparison to `salvas/shadow_log.csv`.
2. **Validate**: `make shadow-report` prints agreement rate, latency delta, and cutover recommendation (triggers when n>=100, agreement>=98%, error rate<=1%).
3. **Cutover**: `make cutover-rust DEVICE_IP=<esp32-ip>` updates the ESP32 NVS setting via HTTP — no firmware reflash needed.
4. **Rollback**: `make rollback-to-python DEVICE_IP=<esp32-ip>` reverts in seconds.
5. **Retire**: After 4 weeks of stable operation, remove `garden-ml-api` from compose.

---

## Hardware

| Component | Part | Notes |
|---|---|---|
| MCU | ESP32-S3 | Dual-core Xtensa LX7, 512 KB SRAM |
| Camera | OV2640 | JPEG hardware encoder, up to 2MP |
| Soil sensor | Capacitive v1.2 | Analog, ADC1 |
| Temperature/humidity | DHT22 | GPIO, OneWire |
| Water pump | 5V mini pump | MOSFET relay on GPIO |

---

## Repository Structure

```
uepg-garden-lowcost-iot-ml/
+-- firmware/smart-tomato-garden/   # C++, PlatformIO, ESP32-S3
+-- ml/
|   +-- src/garden_ml/
|   |   +-- features/               # extract.py, components.py, segmentation
|   |   +-- training/               # train.py, deep_baseline.py, data_module.py
|   |   +-- inference/              # api_flask.py, predictor.py, onnx_export.py, shadow.py
|   |   +-- config/settings.py
|   +-- scripts/                    # export_feature_vectors.py, validate_parity.py, shadow_report.py
|   +-- artifacts/model_registry/
|       +-- v0005/                  # XGBoost: model.pkl, model.onnx, metrics, metadata
|       +-- v0005_dl/               # MobileNetV3: model.pt, model.onnx, metrics
+-- server/                         # Rust axum inference server
|   +-- src/
|   |   +-- features/               # 188-dim pipeline in Rust
|   |   +-- inference/              # ONNX session, response structs
|   |   +-- routes/                 # /health, /predict, /debug
|   +-- Dockerfile                  # Multi-stage: ubuntu:24.04 builder + slim runtime
+-- tools/                          # Dataset tools, set_inference.py
+-- docs/                           # architecture.md, api.md, firmware.md, ml.md
+-- docker-compose.yml
+-- Makefile
+-- CLAUDE.md
```

---

## Quickstart

### Prerequisites

- Docker + Docker Compose v2
- Python 3.11+ with `uv` (for training only)
- Git LFS (`git lfs install`)

### Clone and fetch artifacts

```bash
git clone <repo-url>
cd uepg-garden-lowcost-iot-ml
git lfs pull   # downloads .pkl, .onnx, .pt, .npy artifacts
```

### Run inference server (Python)

```bash
cp .env.example .env
docker compose up -d garden-ml-api

curl -X POST http://localhost:5000/predict \
     --data-binary @sample.jpg \
     -H 'Content-Type: image/jpeg'
```

### Run Rust inference server

```bash
docker compose --profile rust up -d garden-ml-rust

curl -X POST http://localhost:5002/predict \
     --data-binary @sample.jpg \
     -H 'Content-Type: image/jpeg'
```

### Retrain from scratch

```bash
make ml-augment        # prepare dataset (PlantVillage tomato subset)
make docker-train      # train XGBoost pipeline
make docker-eval       # evaluate on test set
make docker-export-onnx
```

### Run tests

```bash
make ml-test           # Python unit tests
make rust-test         # Rust feature parity tests
make validate-parity   # end-to-end Python vs Rust agreement (requires both servers running)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Predict from JPEG body |
| `POST` | `/predict_compare` | Side-by-side XGBoost vs MobileNetV3 with agreement block |
| `POST` | `/debug/features` | Raw 188-dim feature vector (when GML_SAVE_DEBUG=1) |

**Example response (`/predict`):**

```json
{
  "classe_predita": "Tomato___Early_blight",
  "score": 0.873,
  "topk": [
    {"classe": "Tomato___Early_blight", "score": 0.873},
    {"classe": "Tomato___Target_Spot",  "score": 0.089},
    {"classe": "Tomato___healthy",      "score": 0.021}
  ],
  "timings_ms": {"feature_ms": 23.4, "infer_ms": 0.8, "total_ms": 26.1},
  "quality": {"mask_coverage": 0.61, "mean_v": 142, "laplacian_var": 312},
  "confident": true,
  "reasons": []
}
```

---

## Reproducibility

| Artifact | Path | Tracked |
|---|---|---|
| XGBoost model | `ml/artifacts/model_registry/v0005/model.pkl` | Git LFS |
| XGBoost ONNX | `ml/artifacts/model_registry/v0005/model.onnx` | Git LFS |
| MobileNetV3 weights | `ml/artifacts/model_registry/v0005_dl/model.pt` | Git LFS |
| MobileNetV3 ONNX | `ml/artifacts/model_registry/v0005_dl/model.onnx` | Git LFS |
| Training metadata | `ml/artifacts/model_registry/v0005/training_metadata.json` | Git |
| Evaluation metrics | `ml/artifacts/model_registry/v0005/evaluation_metrics.json` | Git |
| Feature ground truth | `ml/artifacts/validation/feature_vectors.npy` | Git LFS |
| Migration benchmark | `ml/artifacts/migration_benchmark.png` | Git |

---

## Citation

```bibtex
@misc{smarttomatogarden2025,
  title  = {Smart Tomato Garden: Low-Cost IoT with Interpretable ML for Tomato Leaf Disease Detection},
  author = {UEPG Research Group},
  year   = {2025},
  url    = {https://github.com/<org>/uepg-garden-lowcost-iot-ml},
  note   = {ESP32-S3 embedded system, 188-dimensional hand-crafted feature pipeline,
            XGBoost macro F1 96.19 pct, MobileNetV3 baseline 99.02 pct,
            production Rust inference server with shadow-mode migration}
}
```

---

## Documentation Index

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Full system diagram, data flow, deployment topology |
| [docs/api.md](docs/api.md) | Complete API reference with request/response schemas |
| [docs/ml.md](docs/ml.md) | Feature pipeline, training decisions, evaluation protocol |
| [docs/firmware.md](docs/firmware.md) | ESP32 firmware, NVS config, web UI, PlatformIO |
| [ml/README.md](ml/README.md) | ML package internals, training contract, inference config |
| [CLAUDE.md](CLAUDE.md) | AI assistant context and codebase guide |

---

## License

MIT — see [LICENSE](LICENSE).
