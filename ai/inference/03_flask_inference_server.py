import csv
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import mahotas
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

IMG_SIZE = (128, 128)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024

SAVE_DEBUG = True
OUTPUT_ROOT = Path("salvas")
CSV_PATH = OUTPUT_ROOT / "csv" / "testes_documentados.csv"

MODEL_PATH = Path("modelo_tomate.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

if SAVE_DEBUG:
    for p in [
        OUTPUT_ROOT / "original",
        OUTPUT_ROOT / "haralick",
        OUTPUT_ROOT / "zernike",
        OUTPUT_ROOT / "lbp",
        OUTPUT_ROOT / "hu",
        OUTPUT_ROOT / "hsv",
        OUTPUT_ROOT / "morfologia",
        OUTPUT_ROOT / "csv",
    ]:
        p.mkdir(parents=True, exist_ok=True)

if SAVE_DEBUG and not CSV_PATH.exists():
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["arquivo", "classe", "score", "tempo_s", "timestamp"])

modelo = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)


def letterbox_rgb(rgb: np.ndarray, size: tuple[int, int], bg: int = 0) -> np.ndarray:
    h, w = rgb.shape[:2]
    th, tw = size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), bg, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def segment_leaf_hsv_from_bgr(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = letterbox_rgb(rgb, IMG_SIZE, bg=0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower1 = np.array([15, 25, 25], dtype=np.uint8)
    upper1 = np.array([40, 255, 255], dtype=np.uint8)
    lower2 = np.array([40, 25, 25], dtype=np.uint8)
    upper2 = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    area_total = mask.shape[0] * mask.shape[1]
    min_area = max(64, int(area_total * 0.01))

    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255

    keep = cv2.GaussianBlur(keep, (5, 5), 0)
    keep = (keep > 0).astype(np.uint8) * 255

    seg_rgb = cv2.bitwise_and(rgb, rgb, mask=keep)
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
    return seg_bgr, keep


def haralick_13(gray: np.ndarray) -> np.ndarray:
    return mahotas.features.haralick(gray).mean(axis=0).astype(np.float64)


def zernike_25(mask: np.ndarray) -> np.ndarray:
    radius = min(mask.shape[0], mask.shape[1]) // 2
    z = mahotas.features.zernike_moments((mask > 0).astype(np.uint8), radius, degree=8).astype(np.float64)
    if z.size != 25:
        raise ValueError(f"zernike size {z.size} != 25")
    return z


def hsv_hist_48(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (mask > 0).astype(np.uint8)

    h_hist = cv2.calcHist([hsv], [0], m, [16], [0, 180]).flatten().astype(np.float64)
    s_hist = cv2.calcHist([hsv], [1], m, [16], [0, 256]).flatten().astype(np.float64)
    v_hist = cv2.calcHist([hsv], [2], m, [16], [0, 256]).flatten().astype(np.float64)

    h_hist /= (h_hist.sum() + 1e-12)
    s_hist /= (s_hist.sum() + 1e-12)
    v_hist /= (v_hist.sum() + 1e-12)

    return np.hstack([h_hist, s_hist, v_hist]).astype(np.float64)


def lbp_mean_std_2(gray: np.ndarray) -> np.ndarray:
    lbp = mahotas.features.lbp(gray, radius=1, points=8).astype(np.float64)
    return np.array([float(lbp.mean()), float(lbp.std())], dtype=np.float64)


def hu_7_from_mask(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments((mask > 0).astype(np.uint8))
    hu = cv2.HuMoments(m).flatten().astype(np.float64)
    out = np.zeros_like(hu)
    for i, v in enumerate(hu):
        if v == 0:
            out[i] = 0.0
        else:
            out[i] = -np.sign(v) * np.log10(abs(v))
    return out


def mean_std_hsv_6(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = mask > 0
    if m.any():
        vals = hsv[m]
        mean = vals.mean(axis=0)[:3].astype(np.float64)
        std = vals.std(axis=0)[:3].astype(np.float64)
    else:
        mean, std = cv2.meanStdDev(hsv)
        mean = mean.flatten()[:3].astype(np.float64)
        std = std.flatten()[:3].astype(np.float64)
    return np.hstack([mean, std]).astype(np.float64)


def morphology_1_from_gray(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = gray.copy()
    fg = mask > 0
    if fg.any():
        fg_mean = float(np.mean(g[fg]))
        g = g.astype(np.float64)
        g[~fg] = fg_mean
        g = np.clip(g, 0, 255).astype(np.uint8)

    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)

    if fg.any():
        val = float(np.mean(opened[fg]) / 255.0)
    else:
        val = float(opened.mean() / 255.0)

    return np.array([val], dtype=np.float64)


def extract_features_102(seg_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)

    har = haralick_13(gray)
    zer = zernike_25(mask)
    hsvh = hsv_hist_48(seg_bgr, mask)
    lbp = lbp_mean_std_2(gray)
    hu = hu_7_from_mask(mask)
    hsv_ms = mean_std_hsv_6(seg_bgr, mask)
    morph = morphology_1_from_gray(gray, mask)

    feat = np.hstack([har, zer, hsvh, lbp, hu, hsv_ms, morph]).astype(np.float64)
    if feat.size != 102:
        raise ValueError(f"expected 102 features, got {feat.size}")
    return feat.reshape(1, -1)


def save_visuals(seg_bgr: np.ndarray, mask: np.ndarray, base: str) -> None:
    if not SAVE_DEBUG:
        return

    gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
    sobel = cv2.normalize(
        cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_ROOT / "haralick" / f"{base}.jpg"), sobel)
    cv2.imwrite(str(OUTPUT_ROOT / "zernike" / f"{base}.jpg"), mask)

    lbp = mahotas.features.lbp(gray, radius=1, points=8).astype(np.float64)
    lbp_img = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_ROOT / "lbp" / f"{base}.jpg"), lbp_img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_v = seg_bgr.copy()
    cv2.drawContours(hu_v, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(OUTPUT_ROOT / "hu" / f"{base}.jpg"), hu_v)

    hsv = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2HSV)
    h_chan = cv2.normalize(hsv[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_ROOT / "hsv" / f"{base}.jpg"), h_chan)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=1)
    cv2.imwrite(str(OUTPUT_ROOT / "morfologia" / f"{base}.jpg"), morph)


def allowed_filename(filename: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in ALLOWED_EXT


def predict_with_topk(feats: np.ndarray, k: int = 3) -> tuple[str, float, list[dict]]:
    if not hasattr(modelo, "predict_proba"):
        raise RuntimeError("loaded model does not support predict_proba")

    probs = modelo.predict_proba(feats)[0]
    idx = int(np.argmax(probs))
    cls = le.inverse_transform([idx])[0]
    score = float(probs[idx])

    top_idx = np.argsort(probs)[::-1][:k]
    topk = [{"classe": le.inverse_transform([int(i)])[0], "score": float(probs[int(i)])} for i in top_idx]
    return cls, score, topk


@app.get("/health")
def health():
    return jsonify(ok=True, classes=list(le.classes_))


@app.post("/analisar")
def analisar():
    if "image" not in request.files:
        return jsonify({"erro": 'Envie o arquivo no campo "image" (multipart/form-data).'}), 400

    file = request.files["image"]
    if not allowed_filename(file.filename or ""):
        ext = os.path.splitext(file.filename or "")[1].lower()
        return jsonify({"erro": f"Extensão não suportada: {ext}"}), 400

    try:
        start = time.time()

        data = file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Imagem inválida")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = secure_filename(f"folha_{ts}")

        if SAVE_DEBUG:
            cv2.imwrite(str(OUTPUT_ROOT / "original" / f"{base}.jpg"), bgr)

        seg_bgr, mask = segment_leaf_hsv_from_bgr(bgr)
        feats = extract_features_102(seg_bgr, mask)

        classe, score, top3 = predict_with_topk(feats, k=3)
        elapsed = round(time.time() - start, 3)

        if SAVE_DEBUG:
            save_visuals(seg_bgr, mask, base)
            with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([f"{base}.jpg", classe, score, elapsed, ts])

        return jsonify(
            {
                "classe_predita": classe,
                "score": score,
                "top3": top3,
                "tempo_inferencia_s": elapsed,
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "erro": "Falha durante a análise.",
                    "mensagem": str(e),
                    "trace": traceback.format_exc(),
                }
            ),
            500,
        )


if __name__ == "__main__":
    logging_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    import logging

    logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))
    app.run(host="0.0.0.0", port=5000, debug=False)
