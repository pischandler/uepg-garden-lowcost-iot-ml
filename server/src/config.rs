use std::path::PathBuf;

/// Server configuration — mirrors GML_* env vars from Python settings.py
#[derive(Debug, Clone)]
pub struct Config {
    pub artifacts_dir: PathBuf,
    pub host: String,
    pub port: u16,
    pub topk: usize,
    pub min_confidence: f64,
    pub min_mask_coverage: f64,
    pub min_mean_v: f64,
    pub min_laplacian_var: f64,
    pub min_input_side_px: u32,
    pub max_content_length: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            artifacts_dir: PathBuf::from(
                std::env::var("GML_ARTIFACTS_DIR")
                    .unwrap_or_else(|_| "artifacts/model_registry/v0005".into()),
            ),
            host: std::env::var("GML_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: std::env::var("GML_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5002),
            topk: std::env::var("GML_TOPK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
            min_confidence: std::env::var("GML_MIN_CONFIDENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.60),
            min_mask_coverage: std::env::var("GML_MIN_MASK_COVERAGE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.08),
            min_mean_v: std::env::var("GML_MIN_MEAN_V")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            min_laplacian_var: std::env::var("GML_MIN_LAPLACIAN_VAR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            min_input_side_px: std::env::var("GML_MIN_INPUT_SIDE_PX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            max_content_length: std::env::var("GML_MAX_CONTENT_LENGTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8 * 1024 * 1024),
        }
    }
}
