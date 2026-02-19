import numpy as np

from garden_ml.features.extract import ExtractOptions, extract_102_from_rgb


def test_extract_returns_102_float64():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    feat = extract_102_from_rgb(rgb, ExtractOptions(img_size=128, photometric_normalize=False))
    assert feat.shape == (102,)
    assert feat.dtype == np.float64


def test_extract_normalize_keeps_dim():
    rgb = np.full((128, 128, 3), 127, dtype=np.uint8)
    feat = extract_102_from_rgb(rgb, ExtractOptions(img_size=128, photometric_normalize=True))
    assert feat.shape == (102,)
