# 🌱 Low-Cost Smart Garden — IoT + AI (ESP32-S3 + Flask)

Low-cost smart indoor garden integrating an **ESP32-S3** (sensors, actuators, and **OV2640** camera) with a **Flask inference API** for **tomato leaf disease detection**.

The AI pipeline performs:
- **Leaf segmentation (HSV)** + **letterbox resize (128×128)**
- Extraction of **102 handcrafted features** (Haralick, LBP, Hu, Zernike, HSV histograms, morphology)
- Model comparison (**Random Forest**, **SVM**, **XGBoost**) with **5-fold stratified cross-validation**
- Deployment of the best model in a **Flask server** returning **prediction + confidence + top-3 + latency**
- Structured **logs** for traceability

End-to-end flow: **capture → transmit → inference → JSON → logs**


## Repository Structure

uepg-horta-lowcost-iot-ml/
├── smart-tomato-garden/
│   └── (ESP32-S3 firmware + OV2640 streaming + sensor/actuator automation)
└── ai/
    ├── model_training/
    │   ├── 01_dataset_augmentation/
    │   └── 02_leaf_segmentation_feature_training/
    └── inference/
        └── 03_flask_inference_server/


## Hardware Module (ESP32-S3)

The firmware runs on ESP32-S3 and is responsible for:
- Reading sensors (soil moisture, light, temperature/humidity)
- Controlling actuators (irrigation pump and fan via relay logic)
- Providing camera streaming and/or image capture for transmission to the inference service
- Maintaining stable operation with timing and hysteresis safeguards

### Main GPIO Mapping (as used in the project)

Component | ESP32-S3 GPIO | Purpose
---|---:|---
Soil moisture sensor (analog) | 1 | Soil moisture reading
LDR light sensor (analog) | 14 | Light intensity reading
DHT22 | 21 | Air temperature/humidity
Relay (pump) | 47 | Pump control

> Camera OV2640 pin mapping is defined in the firmware configuration for the selected ESP32-S3 camera board variant.


## AI Module (Flask + ML)

The AI module is located in `ai/` and contains:
- Dataset augmentation tooling
- Leaf segmentation + feature extraction + model training/evaluation
- Flask inference server aligned with the training pipeline

### What the API returns
For each request, the service outputs:
- Predicted class
- Confidence score
- Top-3 classes with scores
- Inference time (seconds)
- Optional saved artifacts (original + intermediate visual representations) and a CSV log for audit


## Quick Start (AI Inference)

### 1) Create a virtual environment and install dependencies

```bash
cd ai/inference/03_flask_inference_server
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

2) Place trained artifacts

Ensure the following files are available where the inference server expects them:

modelo_tomate.pkl

label_encoder.pkl

These are produced by:
ai/model_training/02_leaf_segmentation_feature_training

3) Run the server

python app.py

4) Test with curl

curl -X POST http://127.0.0.1:5000/analisar \
  -F "image=@/path/to/leaf.jpg"

  Example response:

{
  "classe_predita": "Tomato_Late_blight",
  "score": 0.88,
  "top3": [
    {"classe": "Tomato_Late_blight", "score": 0.88},
    {"classe": "Tomato_Early_blight", "score": 0.07},
    {"classe": "Tomato_Bacterial_spot", "score": 0.03}
  ],
  "tempo_inferencia_s": 0.21
}

Model Training Overview

Training scripts (inside ai/model_training/) follow this methodology:

Segment leaves using HSV thresholds and morphological operations

Standardize images to 128×128 with letterbox resizing

Extract 102-feature vectors per image

Train and compare:

Random Forest

SVM (RBF + scaling)

XGBoost

Hyperparameter tuning via GridSearchCV

5-fold stratified CV for robust selection

Export:

modelo_tomate.pkl

label_encoder.pkl

evaluation reports and comparison files

Operational Notes

Training and inference must use the same preprocessing and 102-feature schema.

If the model is retrained, always replace both:

modelo_tomate.pkl

label_encoder.pkl

For best real-world stability, keep capture conditions as consistent as possible (lighting, focus, distance), or expand augmentation/collection to match field conditions.

License

Recommended: MIT License
Add a LICENSE file at the repository root.

Credits

Developed at Universidade Estadual de Ponta Grossa (UEPG) — Department of Informatics.
Advisor: Prof. Luciano J. Senger
Co-advisor: Prof. Gabrielly de Queiroz Pereira