from __future__ import annotations

from pathlib import Path

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMG_SIZE = 128
FEATURES_DIM = 102

DEFAULT_AUG_MANIFEST = "augmentation_manifest.csv"
DEFAULT_AUG_CONFIG = "augmentation_config.json"

MODEL_FILE = "modelo_tomate.pkl"
ENCODER_FILE = "label_encoder.pkl"
TRAIN_META_FILE = "training_metadata.json"
FEATURE_SCHEMA_FILE = "feature_schema.json"

DEFAULT_REPORTS_DIR = "reports"
DEFAULT_MODEL_REGISTRY_DIR = Path("artifacts") / "model_registry"
