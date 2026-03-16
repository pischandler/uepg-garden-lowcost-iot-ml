use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde_json::{json, Value};
use std::{sync::Arc, time::Instant};

use crate::{features, AppState};

/// POST /predict — raw JPEG/PNG bytes in body.
/// Mirrors Python api_flask.py /predict (raw bytes, same quality checks, same response schema).
pub async fn handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    let t0 = Instant::now();

    // ── size guard ──────────────────────────────────────────────────────────
    if body.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"erro": "Envie bytes JPEG no body (Content-Type: image/jpeg)."})),
        );
    }
    if body.len() > state.config.max_content_length {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({"erro": "Imagem excede tamanho máximo permitido."})),
        );
    }

    // ── feature extraction (CPU-heavy → offload from async executor) ────────
    let t_feat = Instant::now();
    let img_size = state.model.img_size;
    let body_clone = body.clone();
    let extract = match tokio::task::spawn_blocking(move || features::extract(&body_clone, img_size))
        .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "feature_extraction_failed");
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"erro": "Falha na extração de features.", "mensagem": e.to_string()})),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "spawn_blocking_failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"erro": "Erro interno no servidor."})),
            );
        }
    };
    let features_ms = t_feat.elapsed().as_secs_f64() * 1000.0;

    // ── ONNX inference ──────────────────────────────────────────────────────
    let t_pred = Instant::now();
    let probs = match state.model.predict_proba(&extract.features) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "onnx_inference_failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"erro": "Falha durante a inferência.", "mensagem": e.to_string()})),
            );
        }
    };
    let predict_ms = t_pred.elapsed().as_secs_f64() * 1000.0;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── top-k ───────────────────────────────────────────────────────────────
    let k = state.config.topk;
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let best_idx = indexed[0].0;
    let best_score = indexed[0].1 as f64;
    let best_class = state.model.classes.get(best_idx).cloned().unwrap_or_default();

    let topk: Vec<Value> = indexed
        .iter()
        .take(k)
        .filter_map(|&(idx, score)| {
            state.model.classes.get(idx).map(|cls| {
                json!({"class": cls, "score": score as f64})
            })
        })
        .collect();

    // ── quality checks ──────────────────────────────────────────────────────
    let cfg = &state.config;
    let mut reasons: Vec<&str> = Vec::new();

    if cfg.min_mask_coverage > 0.0 && extract.mask_coverage < cfg.min_mask_coverage {
        reasons.push("low_mask_coverage");
    }
    if best_score < cfg.min_confidence {
        reasons.push("low_confidence");
    }
    if cfg.min_mean_v > 0.0 && extract.mean_v < cfg.min_mean_v {
        reasons.push("low_light");
    }

    let confident = reasons.is_empty();
    let classe_predita: Value = if confident {
        json!(best_class)
    } else {
        Value::Null
    };

    // ── optional sensor headers (forwarded as-is) ───────────────────────────
    let device_id = header_str(&headers, "x-device-id");
    let lux_raw = header_str(&headers, "x-lux-raw");
    let soil_raw = header_str(&headers, "x-soil-raw");
    let temp_c = header_str(&headers, "x-temp-c");
    let hum_pct = header_str(&headers, "x-hum-pct");

    let response = json!({
        "classe_predita": classe_predita,
        "score": best_score,
        "topk": topk,
        "confident": confident,
        "reasons": reasons,
        "timings_ms": {
            "features_ms": features_ms,
            "predict_ms": predict_ms,
            "total_ms": total_ms,
        },
        "quality": {
            "mask_coverage": extract.mask_coverage,
            "mean_v": extract.mean_v,
        },
        "meta": {
            "endpoint": "/predict",
            "device_id": device_id,
            "lux_raw": lux_raw,
            "soil_raw": soil_raw,
            "temp_c": temp_c,
            "hum_pct": hum_pct,
        },
        "min_confidence": cfg.min_confidence,
        "min_mask_coverage": cfg.min_mask_coverage,
        "model_img_size": state.model.img_size,
    });

    tracing::info!(
        class = %best_class,
        score = best_score,
        confident = confident,
        features_ms = features_ms,
        predict_ms = predict_ms,
        total_ms = total_ms,
        "predict_ok"
    );

    (StatusCode::OK, Json(response))
}

fn header_str(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string()
}
