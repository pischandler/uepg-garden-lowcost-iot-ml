use anyhow::Result;
use opencv::{
    core::{self as cv, Mat, Vector, no_array},
    imgproc::{self, COLOR_RGB2HSV, COLOR_RGB2Lab},
    prelude::*,
};
use std::f64::consts::PI;

// ─── internal helpers ────────────────────────────────────────────────────────

/// mean ± population std of a single-channel Mat, masked.
/// When mask is all-zero, uses all pixels (matches Python use_mask=None).
fn masked_mean_std(chan: &Mat, mask: &Mat) -> Result<(f64, f64)> {
    let mut mean = Mat::default();
    let mut std = Mat::default();
    if cv::count_non_zero(mask)? > 0 {
        cv::mean_std_dev(chan, &mut mean, &mut std, mask)?;
    } else {
        cv::mean_std_dev(chan, &mut mean, &mut std, &no_array())?;
    }
    Ok((*mean.at::<f64>(0)?, *std.at::<f64>(0)?))
}

/// Extract, optionally mask, and normalize a 1-channel histogram.
/// range: [lo, hi). Returns Vec of length `bins` that sums to 1.
fn calc_hist_1ch(chan: &Mat, mask: &Mat, bins: i32, lo: f32, hi: f32) -> Result<Vec<f64>> {
    let images = Vector::<Mat>::from_iter([chan.try_clone()?]);
    let channels = Vector::<i32>::from_slice(&[0i32]);
    let hist_size = Vector::<i32>::from_slice(&[bins]);
    let ranges = Vector::<f32>::from_slice(&[lo, hi]);
    let mut hist = Mat::default();
    if cv::count_non_zero(mask)? > 0 {
        imgproc::calc_hist(&images, &channels, mask, &mut hist, &hist_size, &ranges, false)?;
    } else {
        imgproc::calc_hist(
            &images,
            &channels,
            &no_array(),
            &mut hist,
            &hist_size,
            &ranges,
            false,
        )?;
    }
    // hist is [bins, 1] f32 — normalize to sum=1
    let n = hist.total() as i32;
    let mut vals: Vec<f64> = (0..n).map(|i| *hist.at::<f32>(i).unwrap() as f64).collect();
    let sum: f64 = vals.iter().sum();
    for v in &mut vals {
        *v /= sum + 1e-12;
    }
    Ok(vals)
}

// ─── public feature functions ────────────────────────────────────────────────

/// 48 features: H(16) + S(16) + V(16) normalized histograms.
/// Mirrors Python hsv_hist_48.
pub fn hsv_hist_48(segmented: &Mat, mask: &Mat) -> Result<Vec<f64>> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(segmented, &mut hsv, COLOR_RGB2HSV, 0)?;

    let mut h_ch = Mat::default();
    let mut s_ch = Mat::default();
    let mut v_ch = Mat::default();
    cv::extract_channel(&hsv, &mut h_ch, 0)?;
    cv::extract_channel(&hsv, &mut s_ch, 1)?;
    cv::extract_channel(&hsv, &mut v_ch, 2)?;

    let mut out = Vec::with_capacity(48);
    out.extend(calc_hist_1ch(&h_ch, mask, 16, 0.0, 180.0)?);
    out.extend(calc_hist_1ch(&s_ch, mask, 16, 0.0, 256.0)?);
    out.extend(calc_hist_1ch(&v_ch, mask, 16, 0.0, 256.0)?);
    Ok(out)
}

/// 7 features: log-scale Hu moments from binary mask.
/// Mirrors Python hu_7.
pub fn hu_7(mask: &Mat) -> Result<Vec<f64>> {
    // binary_image=true: treats nonzero as 1, matching Python (mask>0).astype(uint8)
    let m = imgproc::moments(mask, true)?;
    let mut hu = [0.0f64; 7];
    imgproc::hu_moments(m, &mut hu)?;
    let out = hu
        .iter()
        .map(|&v| {
            if v == 0.0 {
                0.0
            } else {
                -(v.signum()) * v.abs().log10()
            }
        })
        .collect();
    Ok(out)
}

/// 6 features: [mean_H, mean_S, mean_V, std_H, std_S, std_V].
/// Mirrors Python mean_std_hsv_6.
pub fn mean_std_hsv_6(segmented: &Mat, mask: &Mat) -> Result<Vec<f64>> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(segmented, &mut hsv, COLOR_RGB2HSV, 0)?;

    let mut mean = Mat::default();
    let mut std_dev = Mat::default();
    if cv::count_non_zero(mask)? > 0 {
        cv::mean_std_dev(&hsv, &mut mean, &mut std_dev, mask)?;
    } else {
        cv::mean_std_dev(&hsv, &mut mean, &mut std_dev, &no_array())?;
    }
    // mean/std_dev are [3,1] f64; use linear index
    let mut out = Vec::with_capacity(6);
    for i in 0..3 {
        out.push(*mean.at::<f64>(i)?);
    }
    for i in 0..3 {
        out.push(*std_dev.at::<f64>(i)?);
    }
    Ok(out)
}

/// 6 features: area_ratio, perimeter_ratio, aspect, extent, solidity, circularity.
/// Mirrors Python shape_basic_6.
pub fn shape_basic_6(mask: &Mat) -> Result<Vec<f64>> {
    let h = mask.rows() as f64;
    let w = mask.cols() as f64;
    let area_total = h * w;

    if cv::count_non_zero(mask)? == 0 {
        return Ok(vec![0.0; 6]);
    }

    let mut contours = Vector::<Vector<cv::Point>>::new();
    imgproc::find_contours(
        mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        cv::Point::new(0, 0),
    )?;
    if contours.is_empty() {
        return Ok(vec![0.0; 6]);
    }

    // Largest contour by contour area
    let mut max_area = 0.0f64;
    let mut best = 0usize;
    for (i, c) in contours.iter().enumerate() {
        let a = imgproc::contour_area(&c, false)?;
        if a > max_area {
            max_area = a;
            best = i;
        }
    }
    let c = contours.get(best)?;

    let area = imgproc::contour_area(&c, false)?;
    let per = imgproc::arc_length(&c, true)?;
    let rect = imgproc::bounding_rect(&c)?;
    let bw = rect.width as f64;
    let bh_r = rect.height as f64;
    let bbox_area = bw * bh_r;

    let mut hull_pts = Vector::<cv::Point>::new();
    imgproc::convex_hull(&c, &mut hull_pts, false, true)?;
    let hull_area = imgproc::contour_area(&hull_pts, false)?;

    let area_ratio = area / (area_total + 1e-12);
    let per_ratio = per / (2.0 * (h + w) + 1e-12);
    let aspect = bw / (bh_r + 1e-12);
    let extent = area / (bbox_area + 1e-12);
    let solidity = area / (hull_area + 1e-12);
    let circularity = (4.0 * PI * area) / (per * per + 1e-12);

    Ok(vec![area_ratio, per_ratio, aspect, extent, solidity, circularity])
}

/// 4 features: [mean_a, std_a, mean_b, std_b] of L*a*b* a and b channels (uint8 range).
/// Mirrors Python lab_ab_mean_std_4.
pub fn lab_ab_mean_std_4(segmented: &Mat, mask: &Mat) -> Result<Vec<f64>> {
    let mut lab = Mat::default();
    imgproc::cvt_color(segmented, &mut lab, COLOR_RGB2Lab, 0)?;

    let mut a_ch = Mat::default();
    let mut b_ch = Mat::default();
    cv::extract_channel(&lab, &mut a_ch, 1)?;
    cv::extract_channel(&lab, &mut b_ch, 2)?;

    let (am, ast) = masked_mean_std(&a_ch, mask)?;
    let (bm, bst) = masked_mean_std(&b_ch, mask)?;
    Ok(vec![am, ast, bm, bst])
}

/// 16 features: normalized histogram a(8 bins) + b(8 bins), range [0,256).
/// Mirrors Python lab_ab_hist_16.
pub fn lab_ab_hist_16(segmented: &Mat, mask: &Mat) -> Result<Vec<f64>> {
    let mut lab = Mat::default();
    imgproc::cvt_color(segmented, &mut lab, COLOR_RGB2Lab, 0)?;

    let mut a_ch = Mat::default();
    let mut b_ch = Mat::default();
    cv::extract_channel(&lab, &mut a_ch, 1)?;
    cv::extract_channel(&lab, &mut b_ch, 2)?;

    let mut out = Vec::with_capacity(16);
    out.extend(calc_hist_1ch(&a_ch, mask, 8, 0.0, 256.0)?);
    out.extend(calc_hist_1ch(&b_ch, mask, 8, 0.0, 256.0)?);
    Ok(out)
}

/// 4 features: [mean_rc, std_rc, mean_gc, std_gc] of normalised chromaticity.
/// rc = R/(R+G+B+ε), gc = G/(R+G+B+ε). Mirrors Python chroma_rg_mean_std_4.
pub fn chroma_rg_mean_std_4(segmented: &Mat, mask: &Mat) -> Result<Vec<f64>> {
    let rows = segmented.rows();
    let cols = segmented.cols();
    let use_all = cv::count_non_zero(mask)? == 0;

    let mut rc_vals: Vec<f64> = Vec::with_capacity((rows * cols) as usize);
    let mut gc_vals: Vec<f64> = Vec::with_capacity((rows * cols) as usize);

    for row in 0..rows {
        for col in 0..cols {
            if !use_all && *mask.at_2d::<u8>(row, col)? == 0 {
                continue;
            }
            let px = segmented.at_2d::<cv::Vec3b>(row, col)?;
            let r = px[0] as f64;
            let g = px[1] as f64;
            let b = px[2] as f64;
            let s = r + g + b + 1e-12;
            rc_vals.push(r / s);
            gc_vals.push(g / s);
        }
    }

    let rcm = slice_mean(&rc_vals);
    let rcs = slice_std(&rc_vals, rcm);
    let gcm = slice_mean(&gc_vals);
    let gcs = slice_std(&gc_vals, gcm);

    Ok(vec![rcm, rcs, gcm, gcs])
}

fn slice_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn slice_std(v: &[f64], mean: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt()
}
