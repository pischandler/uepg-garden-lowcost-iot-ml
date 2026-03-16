use anyhow::Result;
use opencv::{core as cv, prelude::*};
use std::f64::consts::PI;

// ─── LBP NRI Uniform (59 features) ─────────────────────────────────────────

/// 59 features: normalized NRI uniform LBP histogram.
/// Mirrors skimage.feature.local_binary_pattern(gray, P=8, R=1, method='nri_uniform').
///
/// Neighbor ordering (P=8, R=1) matches skimage:
///   theta = 2π*p/P, dc = cos(theta), dr = -sin(theta) → image coords.
/// Comparison: neighbor >= center (same as skimage `>= 0` check).
/// Border pixels: nearest-neighbor clamping.
pub fn lbp_nri_uniform_hist_59(gray: &opencv::core::Mat, mask: &opencv::core::Mat) -> Result<Vec<f64>> {
    let table = build_lbp_table();
    let rows = gray.rows();
    let cols = gray.cols();
    let use_all = cv::count_non_zero(mask)? == 0;

    // (dr, dc) for p = 0..7 at theta = 2π*p/8
    const OFFSETS: [(i32, i32); 8] = [
        (0, 1),   // p=0: right
        (-1, 1),  // p=1: top-right
        (-1, 0),  // p=2: top
        (-1, -1), // p=3: top-left
        (0, -1),  // p=4: left
        (1, -1),  // p=5: bottom-left
        (1, 0),   // p=6: bottom
        (1, 1),   // p=7: bottom-right
    ];

    let mut hist = [0u64; 59];

    for r in 0..rows {
        for c in 0..cols {
            if !use_all && *mask.at_2d::<u8>(r, c)? == 0 {
                continue;
            }
            let center = *gray.at_2d::<u8>(r, c)? as i32;
            let mut code = 0u8;
            for (p, &(dr, dc)) in OFFSETS.iter().enumerate() {
                let nr = (r + dr).clamp(0, rows - 1);
                let nc = (c + dc).clamp(0, cols - 1);
                if *gray.at_2d::<u8>(nr, nc)? as i32 >= center {
                    code |= 1u8 << p;
                }
            }
            hist[table[code as usize] as usize] += 1;
        }
    }

    let total = hist.iter().sum::<u64>() as f64;
    Ok(hist.iter().map(|&h| h as f64 / (total + 1e-12)).collect())
}

/// LBP lookup table: pattern code (0–255) → NRI uniform bin (0–57, or 58 = non-uniform).
/// Uniform patterns have ≤ 2 circular bit transitions.
fn build_lbp_table() -> [u8; 256] {
    let mut table = [58u8; 256];
    let mut bin = 0u8;
    for i in 0u32..256 {
        let bits = i as u8;
        let trans = (0..8u32)
            .filter(|&p| ((bits >> p) & 1) != ((bits >> ((p + 1) % 8)) & 1))
            .count();
        if trans <= 2 {
            table[i as usize] = bin;
            bin += 1;
        }
    }
    table
}

// ─── Haralick (13 features) ─────────────────────────────────────────────────

/// 13 features: Haralick GLCM texture features, mean over 4 directions.
/// Mirrors mahotas.features.haralick(gray) with default params
/// (distance=1, preserve_haralick_bug=False → Sum Variance uses Sum Average).
pub fn haralick_13(gray: &opencv::core::Mat) -> Result<Vec<f64>> {
    let rows = gray.rows();
    let cols = gray.cols();

    // GLCM size = max gray value + 1 (mahotas uses image max, not dtype max)
    let mut max_val = 0u8;
    for r in 0..rows {
        for c in 0..cols {
            let v = *gray.at_2d::<u8>(r, c)?;
            if v > max_val {
                max_val = v;
            }
        }
    }
    let ng = max_val as usize + 1;

    // 4 directions (dy, dx): 0°, 45°, 90°, 135°
    const DIRS: [(i32, i32); 4] = [(0, 1), (-1, 1), (-1, 0), (-1, -1)];

    let mut all_feats: Vec<Vec<f64>> = Vec::with_capacity(4);

    for &(dy, dx) in &DIRS {
        // Build symmetric GLCM: count each pair in both (i,j) and (j,i)
        let mut glcm = vec![0u64; ng * ng];
        for r in 0..rows {
            for c in 0..cols {
                let nr = r + dy;
                let nc = c + dx;
                if nr >= 0 && nr < rows && nc >= 0 && nc < cols {
                    let i = *gray.at_2d::<u8>(r, c)? as usize;
                    let j = *gray.at_2d::<u8>(nr, nc)? as usize;
                    glcm[i * ng + j] += 1;
                    glcm[j * ng + i] += 1;
                }
            }
        }

        // Normalize
        let total = glcm.iter().sum::<u64>() as f64;
        let p: Vec<f64> = if total == 0.0 {
            vec![0.0; ng * ng]
        } else {
            glcm.iter().map(|&v| v as f64 / total).collect()
        };

        all_feats.push(haralick_features(&p, ng));
    }

    // Mean across 4 directions
    let mut result = vec![0.0f64; 13];
    for k in 0..13 {
        result[k] = all_feats.iter().map(|f| f[k]).sum::<f64>() / 4.0;
    }
    Ok(result)
}

/// Compute 13 Haralick features from a normalized GLCM p of size ng×ng.
fn haralick_features(p: &[f64], ng: usize) -> Vec<f64> {
    // Row and column marginals
    let mut px = vec![0.0f64; ng];
    let mut py = vec![0.0f64; ng];
    for i in 0..ng {
        for j in 0..ng {
            px[i] += p[i * ng + j];
            py[j] += p[i * ng + j];
        }
    }

    let mu_x: f64 = (0..ng).map(|i| i as f64 * px[i]).sum();
    let mu_y: f64 = (0..ng).map(|j| j as f64 * py[j]).sum();
    let sigma_x: f64 = (0..ng)
        .map(|i| (i as f64 - mu_x).powi(2) * px[i])
        .sum::<f64>()
        .sqrt();
    let sigma_y: f64 = (0..ng)
        .map(|j| (j as f64 - mu_y).powi(2) * py[j])
        .sum::<f64>()
        .sqrt();

    // px+y[k] = Σ_{i+j=k} P[i,j],  k ∈ [0, 2*(ng-1)]
    let mut pxy_sum = vec![0.0f64; 2 * ng - 1];
    // px-y[k] = Σ_{|i-j|=k} P[i,j],  k ∈ [0, ng-1]
    let mut pxy_diff = vec![0.0f64; ng];
    for i in 0..ng {
        for j in 0..ng {
            pxy_sum[i + j] += p[i * ng + j];
            pxy_diff[i.abs_diff(j)] += p[i * ng + j];
        }
    }

    // f1..f5: computed with explicit loops to avoid Vec<f64> move issues in closures
    let mut f1 = 0.0f64;
    let mut f2 = 0.0f64;
    let mut f3_num = 0.0f64;
    let mut f4 = 0.0f64;
    let mut f5 = 0.0f64;
    for i in 0..ng {
        for j in 0..ng {
            let pij = p[i * ng + j];
            let diff = i as i64 - j as i64;
            f1 += pij * pij;
            f2 += diff.pow(2) as f64 * pij;
            f3_num += i as f64 * j as f64 * pij;
            f4 += (i as f64 - mu_x).powi(2) * pij;
            f5 += pij / (1.0 + diff.pow(2) as f64);
        }
    }
    let f3 = if sigma_x < 1e-15 || sigma_y < 1e-15 {
        1.0
    } else {
        (f3_num - mu_x * mu_y) / (sigma_x * sigma_y)
    };

    // f6: Sum Average
    let f6: f64 = pxy_sum
        .iter()
        .enumerate()
        .map(|(k, &v)| k as f64 * v)
        .sum();

    // f7: Sum Variance — uses f6 (Sum Average), not f8 [Haralick 1973, mahotas default]
    let f7: f64 = pxy_sum
        .iter()
        .enumerate()
        .map(|(k, &v)| (k as f64 - f6).powi(2) * v)
        .sum();

    // f8: Sum Entropy
    let f8: f64 = -pxy_sum
        .iter()
        .map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 })
        .sum::<f64>();

    // f9: Entropy
    let f9: f64 = -p
        .iter()
        .map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 })
        .sum::<f64>();

    // f10: Difference Variance
    let diff_mean: f64 = pxy_diff
        .iter()
        .enumerate()
        .map(|(k, &v)| k as f64 * v)
        .sum();
    let f10: f64 = pxy_diff
        .iter()
        .enumerate()
        .map(|(k, &v)| (k as f64 - diff_mean).powi(2) * v)
        .sum();

    // f11: Difference Entropy
    let f11: f64 = -pxy_diff
        .iter()
        .map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 })
        .sum::<f64>();

    // f12, f13: Information Measures of Correlation
    let hx: f64 = -px.iter().map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 }).sum::<f64>();
    let hy: f64 = -py.iter().map(|&v| if v > 0.0 { v * v.ln() } else { 0.0 }).sum::<f64>();

    let mut hxy1 = 0.0f64;
    let mut hxy2 = 0.0f64;
    for i in 0..ng {
        for j in 0..ng {
            let pij = p[i * ng + j];
            let q = px[i] * py[j];
            if pij > 0.0 && q > 0.0 {
                hxy1 += pij * q.ln();
            }
            if q > 0.0 {
                hxy2 += q * q.ln();
            }
        }
    }
    let hxy1 = -hxy1;
    let hxy2 = -hxy2;

    let max_h = hx.max(hy);
    let f12 = if max_h < 1e-15 { 0.0 } else { (f9 - hxy1) / max_h };
    let f13 = (1.0_f64 - (-2.0 * (hxy2 - f9)).exp()).max(0.0).sqrt();

    vec![f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
}

// ─── Zernike moments (25 features) ──────────────────────────────────────────

/// 25 features: Zernike moment magnitudes up to degree 8.
/// Mirrors mahotas.features.zernike_moments((mask>0).astype(uint8), radius, degree=8).
///
/// Convention matches mahotas _zernike.pyx:
///   center = (rows//2, cols//2)
///   x = col - col_center,  y = -(row - row_center)   [math coords]
///   rho = sqrt(x²+y²) / radius
///   theta = atan2(y, x)
///   A_nm = (n+1)/π * Σ_{ρ≤1} R_nm(ρ) * exp(-j*m*θ) * f(x,y)
///
/// Single-pass optimisation: outer loop = pixels, inner loop = 25 moments.
/// Avoids 25× redundant rho/theta computation of the naive per-moment approach.
pub fn zernike_25(mask: &opencv::core::Mat) -> Result<Vec<f64>> {
    let rows = mask.rows();
    let cols = mask.cols();

    let row_center = (rows / 2) as f64;
    let col_center = (cols / 2) as f64;
    let radius = ((rows.min(cols) / 2) as f64).max(1.0);

    let pairs = zernike_pairs(8); // 25 (n,m) pairs
    let n_moments = pairs.len();
    let mut re = vec![0.0f64; n_moments];
    let mut im = vec![0.0f64; n_moments];

    for r in 0..rows {
        for c in 0..cols {
            // Binary mask: nonzero → pixel value 1.0 (matches Python (mask>0).astype(uint8))
            if *mask.at_2d::<u8>(r, c)? == 0 {
                continue;
            }

            // Math coordinates (matches mahotas)
            let x = c as f64 - col_center;
            let y = -(r as f64 - row_center);
            let rho = (x * x + y * y).sqrt() / radius;
            if rho > 1.0 {
                continue;
            }

            let theta = y.atan2(x);

            // Accumulate all 25 moments for this pixel in one pass
            for (k, &(n, m)) in pairs.iter().enumerate() {
                let rpoly = radial_poly(n, m, rho);
                let mt = -(m as f64) * theta;
                re[k] += rpoly * mt.cos();
                im[k] += rpoly * mt.sin();
            }
        }
    }

    let result = pairs
        .iter()
        .enumerate()
        .map(|(k, &(n, _m))| {
            let scale = (n as f64 + 1.0) / PI;
            ((re[k] * scale).powi(2) + (im[k] * scale).powi(2)).sqrt()
        })
        .collect();

    Ok(result)
}

/// (n, m) pairs for Zernike moments of degree ≤ max_degree.
/// Order: increasing n, then m = n%2, n%2+2, ..., n (same as mahotas enumerate_moments).
fn zernike_pairs(max_degree: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for n in 0..=max_degree {
        let mut m = n % 2;
        while m <= n {
            pairs.push((n, m));
            m += 2;
        }
    }
    pairs
}

/// Zernike radial polynomial R_nm(ρ).
/// R_nm(ρ) = Σ_{s=0}^{(n-m)/2} (-1)^s (n-s)! / (s! * ((n+m)/2-s)! * ((n-m)/2-s)!) * ρ^(n-2s)
fn radial_poly(n: usize, m: usize, rho: f64) -> f64 {
    let half_nm = (n - m) / 2;
    let half_np = (n + m) / 2;
    let mut result = 0.0f64;
    for s in 0..=half_nm {
        let sign = if s % 2 == 0 { 1.0f64 } else { -1.0 };
        let coeff = sign
            * fac(n - s)
            / (fac(s) * fac(half_np - s) * fac(half_nm - s));
        result += coeff * rho.powi((n - 2 * s) as i32);
    }
    result
}

fn fac(n: usize) -> f64 {
    (1..=n).fold(1.0f64, |acc, x| acc * x as f64)
}
