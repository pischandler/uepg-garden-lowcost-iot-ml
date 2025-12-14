AI Module — Tomato Leaf Disease Detection (IoT + Computer Vision + ML)
=====================================================================

This folder contains the complete Artificial Intelligence (AI) pipeline used in the project, aligned with the TCC methodology:
- Image standardization using letterbox resize (128×128)
- Leaf segmentation in HSV color space (green and green-yellow ranges) with morphological filtering
- Data augmentation to improve robustness to lighting and capture variability
- Feature extraction with a fixed schema of 102 handcrafted descriptors
- Model comparison (Random Forest, SVM, XGBoost) using 5-fold stratified cross-validation
- Deployment of the best-performing model in a Flask inference server
- Traceability through saved artifacts, reports, and request logs


Folder Layout
-------------

ai/
├── model_training/
│   ├── 01_dataset_augmentation/
│   └── 02_leaf_segmentation_feature_training/
└── inference/
    └── 03_flask_inference_server/


1) model_training/01_dataset_augmentation
----------------------------------------

Purpose:
- Prepare the image dataset through controlled data augmentation.
- Output an augmented dataset with the same class-folder structure.
- Produce a manifest file for reproducibility and auditing.

Key operations:
- Letterbox resize to 128×128 (preserves aspect ratio with padding)
- Augmentation operations:
  - rotation
  - brightness variation
  - width/height shifts
  - zoom
  - horizontal flip
  - mild hue shift and gamma jitter
- Manifest generation (CSV) linking each output image to its source image and seed.

Expected input structure:
dataset/
└── <class_name>/
    ├── img1.jpg
    ├── img2.jpg
    └── ...

Expected output structure:
dataset_aug/
└── <class_name>/
    ├── <source>_aug001.jpg
    ├── <source>_aug002.jpg
    └── ...


2) model_training/02_leaf_segmentation_feature_training
-------------------------------------------------------

Purpose:
- Build the training dataset by extracting the exact 102-feature vector per image as described in the TCC.
- Train and evaluate Random Forest, SVM, and XGBoost.
- Select the best model based on test performance and cross-validation results.
- Export the final artifacts for inference.

Pipeline steps:
- Input: augmented dataset folder (class subfolders)
- Preprocessing:
  - letterbox resize to 128×128
  - HSV leaf segmentation with two hue ranges + open/close morphology + small component removal
- Feature extraction (102 total):
  - Haralick texture descriptors (13)
  - LBP histogram (8)
  - Hu moments (7)
  - Zernike moments (25)
  - HSV histograms (48: 16 bins for H, S, and V)
  - Morphology ratio feature (1)
- Train/test split: stratified 70/30
- Hyperparameter search: GridSearchCV with StratifiedKFold (5 folds)
- Models evaluated:
  - Random Forest
  - SVM (RBF kernel with scaling)
  - XGBoost

Outputs (recommended artifacts folder):
- features_selecionadas.csv
- reports/<ModelName>/classification_report.txt
- reports/<ModelName>/confusion_matrix.csv
- reports/<ModelName>/cv_results.csv
- model_comparison.csv and model_comparison.json
- training_metadata.json
- modelo_tomate.pkl (best model)
- label_encoder.pkl


3) inference/03_flask_inference_server
--------------------------------------

Purpose:
- Provide a production-ready inference API for real-time diagnosis.
- Ensure the inference pipeline matches training exactly (preprocessing + segmentation + feature schema).

API behavior:
- Receives an image via HTTP (multipart/form-data)
- Standardizes the image (letterbox 128×128)
- Segments the leaf (HSV-based)
- Extracts the same 102 features
- Predicts disease class with probability outputs
- Returns:
  - predicted class
  - confidence score
  - top-3 classes
  - inference latency

Traceability outputs:
- Request log in CSV (timestamp, prediction, score, inference time)
- Saved copies of:
  - original image
  - intermediate visual representations used for documentation:
    - Haralick proxy visualization
    - Zernike mask visualization
    - LBP visualization
    - Hu contour overlay
    - HSV-H channel visualization
    - Morphology visualization


Operational Notes
-----------------

- The training and inference pipelines must remain identical:
  image size, segmentation parameters, and the 102-feature schema.
- The label encoder exported during training must be used in inference.
- If a model is retrained, replace both:
  - modelo_tomate.pkl
  - label_encoder.pkl

Recommended usage flow:
1) Run dataset augmentation (01)
2) Train/evaluate/export artifacts (02)
3) Launch Flask inference server (03)
4) Send images from ESP32-S3 or external clients (Postman/curl)


Project Scope
-------------

This AI module is designed to support low-cost indoor cultivation automation with computer vision-based disease diagnosis, enabling future extensions such as:
- domain adaptation to real-world lighting conditions,
- additional crops and disease sets,
- deployment optimizations for edge inference.
