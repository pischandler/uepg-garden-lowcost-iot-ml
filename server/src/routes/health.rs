use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::AppState;

pub async fn handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "classes": state.model.classes,
        "artifacts_dir": state.config.artifacts_dir.to_string_lossy(),
        "model_img_size": state.model.img_size,
        "min_input_side_px": state.config.min_input_side_px,
        "min_confidence": state.config.min_confidence,
        "min_mask_coverage": state.config.min_mask_coverage,
        "backends": { "classical": true, "mobilenet": false }
    }))
}
