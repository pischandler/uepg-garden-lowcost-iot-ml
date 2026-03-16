use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

pub struct OnnxModel {
    session: Mutex<Session>,
    pub classes: Vec<String>,
    pub img_size: u32,
}

impl OnnxModel {
    pub fn load(artifacts_dir: &Path) -> Result<Self> {
        let onnx_path = artifacts_dir.join("model.onnx");
        anyhow::ensure!(
            onnx_path.exists(),
            "model.onnx not found at {}",
            onnx_path.display()
        );

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create ort session builder: {e:?}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e:?}"))?
            .commit_from_file(&onnx_path)
            .map_err(|e| anyhow::anyhow!("failed to load model.onnx: {e:?}"))?;

        let meta_path = artifacts_dir.join("training_metadata.json");
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&meta_path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", meta_path.display()))?,
        )
        .map_err(|e| anyhow::anyhow!("failed to parse training_metadata.json: {e}"))?;

        let classes: Vec<String> = meta["classes"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing 'classes' in training_metadata.json"))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let img_size = meta["img_size"].as_u64().unwrap_or(128) as u32;

        tracing::info!(
            path = %onnx_path.display(),
            n_classes = classes.len(),
            img_size,
            "model_loaded"
        );

        Ok(Self { session: Mutex::new(session), classes, img_size })
    }

    /// Run inference on a [1, 188] float32 feature vector.
    /// Returns probabilities aligned with self.classes.
    pub fn predict_proba(&self, features: &[f32]) -> Result<Vec<f32>> {
        let n = features.len();

        // Use tuple form (shape, data) to avoid ndarray version conflicts with ort internals
        let tensor = Tensor::from_array(([1usize, n], features.to_vec()))
            .map_err(|e| anyhow::anyhow!("failed to create ort tensor: {e:?}"))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["float_input" => tensor])
            .map_err(|e| anyhow::anyhow!("ort session run failed: {e:?}"))?;

        // skl2onnx exports: output[0] = labels, output[1] = probabilities
        // try_extract_tensor returns (Shape, &[T]) — .1 is the data slice
        let probs_raw = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("failed to extract probabilities: {e:?}"))?;

        Ok(probs_raw.1.to_vec())
    }
}
