import numpy as np

from garden_ml.features.extract import ExtractOptions, extract_features_from_rgb


def test_extract_returns_188_float64():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    feat = extract_features_from_rgb(rgb, ExtractOptions(img_size=128))
    assert feat.shape == (188,)
    assert feat.dtype == np.float64
