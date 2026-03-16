use axum::{body::Bytes, extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::{features, AppState};

/// POST /debug/features — raw JPEG/PNG bytes in body.
/// Returns the 188-dim feature vector as JSON for parity validation against Python.
/// Only enabled when GML_DEBUG=1.
pub async fn handler(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    if body.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"erro": "Body vazio."})),
        );
    }

    let img_size = state.model.img_size;
    let body_clone = body.clone();
    let extract = match tokio::task::spawn_blocking(move || features::extract(&body_clone, img_size))
        .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"erro": "Falha na extração de features.", "mensagem": e.to_string()})),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"erro": "Erro interno.", "mensagem": e.to_string()})),
            );
        }
    };

    // Return features as f64 array (cast back for readability; f32 precision is kept)
    let features_f64: Vec<f64> = extract.features.iter().map(|&x| x as f64).collect();

    (
        StatusCode::OK,
        Json(json!({
            "features": features_f64,
            "n_features": features_f64.len(),
            "mask_coverage": extract.mask_coverage,
            "mean_v": extract.mean_v,
        })),
    )
}
