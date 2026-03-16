mod group_a;
mod group_b;
mod preprocess;

use anyhow::Result;
use opencv::{imgcodecs, imgproc, prelude::*};

pub const N_FEATURES: usize = 188;

/// Output of a successful feature extraction.
pub struct ExtractResult {
    /// 188-dim feature vector (f32 for ONNX Runtime).
    pub features: Vec<f32>,
    /// Fraction of pixels classified as leaf foreground (0.0–1.0).
    pub mask_coverage: f64,
    /// Mean brightness (V channel in HSV) over foreground pixels.
    pub mean_v: f64,
}

/// Decode raw image bytes, segment leaf, and extract the 188-dim feature vector.
///
/// Feature layout (must match Python extract.py hstack order):
///   [0..13]   haralick_13          (Group B stub — zeros until Phase 2C)
///   [13..38]  zernike_25           (Group B stub)
///   [38..86]  hsv_hist_48          (Group A ✓)
///   [86..145] lbp_nri_uniform_59   (Group B stub)
///   [145..152] hu_7                (Group A ✓)
///   [152..158] mean_std_hsv_6      (Group A ✓)
///   [158..164] shape_basic_6       (Group A ✓)
///   [164..168] lab_ab_mean_std_4   (Group A ✓)
///   [168..184] lab_ab_hist_16      (Group A ✓)
///   [184..188] chroma_rg_mean_std_4 (Group A ✓)
pub fn extract(bytes: &[u8], img_size: u32) -> Result<ExtractResult> {
    let rgb = decode_to_rgb(bytes)?;
    let size = img_size as i32;
    let (segmented, mask) = preprocess::segment_leaf_hsv(&rgb, size)?;

    // Gray image with background filled to mean fg value (for haralick + lbp)
    let mut gray_raw = opencv::core::Mat::default();
    imgproc::cvt_color(&segmented, &mut gray_raw, imgproc::COLOR_RGB2GRAY, 0)?;
    let gray = preprocess::fill_background_gray(&gray_raw, &mask)?;

    let nz = opencv::core::count_non_zero(&mask)? as f64;
    let total = (mask.rows() * mask.cols()) as f64;
    let mask_coverage = nz / total.max(1.0);
    let mean_v = preprocess::mean_v_hsv(&segmented, &mask)?;

    let mut feat: Vec<f64> = Vec::with_capacity(N_FEATURES);

    feat.extend(group_b::haralick_13(&gray)?);                         // [0..13]
    feat.extend(group_b::zernike_25(&mask)?);                          // [13..38]
    feat.extend(group_a::hsv_hist_48(&segmented, &mask)?);             // [38..86]
    feat.extend(group_b::lbp_nri_uniform_hist_59(&gray, &mask)?);      // [86..145]
    feat.extend(group_a::hu_7(&mask)?);                                // [145..152]
    feat.extend(group_a::mean_std_hsv_6(&segmented, &mask)?);          // [152..158]
    feat.extend(group_a::shape_basic_6(&mask)?);                       // [158..164]
    feat.extend(group_a::lab_ab_mean_std_4(&segmented, &mask)?);       // [164..168]
    feat.extend(group_a::lab_ab_hist_16(&segmented, &mask)?);          // [168..184]
    feat.extend(group_a::chroma_rg_mean_std_4(&segmented, &mask)?);    // [184..188]

    debug_assert_eq!(feat.len(), N_FEATURES, "feature count mismatch: {}", feat.len());

    Ok(ExtractResult {
        features: feat.iter().map(|&x| x as f32).collect(),
        mask_coverage,
        mean_v,
    })
}

/// Decode raw JPEG/PNG/etc bytes to an RGB Mat (CV_8UC3).
fn decode_to_rgb(bytes: &[u8]) -> Result<opencv::core::Mat> {
    let buf = opencv::core::Mat::from_slice(bytes)?;
    let bgr = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;
    anyhow::ensure!(!bgr.empty(), "failed to decode image (empty result)");
    let mut rgb = opencv::core::Mat::default();
    imgproc::cvt_color(&bgr, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
    Ok(rgb)
}
