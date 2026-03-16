mod config;
mod features;
mod inference;
mod routes;

use anyhow::Result;
use axum::{routing::{get, post}, Router};
use std::sync::Arc;
use tower_http::timeout::TimeoutLayer;
use std::time::Duration;

use config::Config;
use inference::OnnxModel;

pub struct AppState {
    pub config: Config,
    pub model: OnnxModel,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "garden_ml_server=info,tower_http=info".into()),
        )
        .init();

    let config = Config::from_env();

    tracing::info!(
        artifacts_dir = %config.artifacts_dir.display(),
        host = %config.host,
        port = config.port,
        "startup"
    );

    let model = OnnxModel::load(&config.artifacts_dir)?;
    let debug_mode = std::env::var("GML_DEBUG").as_deref() == Ok("1");
    let state = Arc::new(AppState { config: config.clone(), model });

    let mut app = Router::new()
        .route("/health", get(routes::health::handler))
        .route("/predict", post(routes::predict::handler));

    if debug_mode {
        tracing::warn!("debug_mode_enabled endpoint=/debug/features");
        app = app.route("/debug/features", post(routes::debug::handler));
    }

    let app = app
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, "listening");

    axum::serve(listener, app).await?;
    Ok(())
}
