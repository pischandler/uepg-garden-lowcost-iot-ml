import numpy as np

from garden_ml.inference.predictor import LoadedArtifacts, predict_topk


class DummyEnc:
    def __init__(self):
        self.classes_ = np.array(["A", "B"], dtype=object)

    def inverse_transform(self, idx):
        return self.classes_[np.array(idx, dtype=int)]


class DummyModel:
    def __init__(self):
        self.n_features_in_ = 188

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile(np.array([[0.2, 0.8]], dtype=np.float64), (n, 1))


def test_predict_topk_shapes():
    arts = LoadedArtifacts(model=DummyModel(), encoder=DummyEnc(), classes=["A", "B"], photometric_normalize_default=False)
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    cls, score, topk, timings = predict_topk(arts, rgb, img_size=128, k=2, photometric_normalize=False)
    assert cls in {"A", "B"}
    assert isinstance(score, float)
    assert len(topk) == 2
    assert "features_ms" in timings
    assert "predict_ms" in timings
    assert "total_ms" in timings
