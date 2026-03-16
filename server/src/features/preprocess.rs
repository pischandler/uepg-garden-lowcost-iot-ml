use anyhow::Result;
use opencv::{
    core::{
        self as cv, Mat, Point, Rect, Scalar, Size, no_array, BORDER_CONSTANT, BORDER_DEFAULT,
        CV_32S, CV_8UC1, CV_8UC3,
    },
    imgproc::{
        self, COLOR_RGB2HSV, INTER_AREA, MORPH_CLOSE, MORPH_ELLIPSE, MORPH_OPEN,
        THRESH_BINARY,
    },
    prelude::*,
};

/// Letterbox resize to (size × size) with zero-padding.
/// Mirrors Python letterbox_rgb: cv2.resize(INTER_AREA) + center padding.
pub fn letterbox_rgb(rgb: &Mat, size: i32) -> Result<Mat> {
    let h = rgb.rows() as f64;
    let w = rgb.cols() as f64;
    let scale = f64::min(size as f64 / w, size as f64 / h);
    let nw = (w * scale).round() as i32;
    let nh = (h * scale).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(rgb, &mut resized, Size::new(nw, nh), 0.0, 0.0, INTER_AREA)?;

    let mut canvas = Mat::new_rows_cols_with_default(size, size, CV_8UC3, Scalar::all(0.0))?;
    let top = (size - nh) / 2;
    let left = (size - nw) / 2;
    let roi = Rect::new(left, top, nw, nh);
    let mut canvas_roi = canvas.roi_mut(roi)?;
    resized.copy_to(&mut canvas_roi)?;

    Ok(canvas)
}

/// HSV-based leaf segmentation.
/// Returns (segmented_rgb, mask) both size×size, mirroring Python segment_leaf_hsv.
pub fn segment_leaf_hsv(rgb: &Mat, size: i32) -> Result<(Mat, Mat)> {
    let rgb = letterbox_rgb(rgb, size)?;

    let mut hsv = Mat::default();
    imgproc::cvt_color(&rgb, &mut hsv, COLOR_RGB2HSV, 0)?;

    // Two overlapping green hue ranges (matches Python lower1/upper1, lower2/upper2)
    let mut mask1 = Mat::default();
    let mut mask2 = Mat::default();
    cv::in_range(
        &hsv,
        &Mat::from_slice(&[15u8, 25u8, 25u8])?,
        &Mat::from_slice(&[40u8, 255u8, 255u8])?,
        &mut mask1,
    )?;
    cv::in_range(
        &hsv,
        &Mat::from_slice(&[40u8, 25u8, 25u8])?,
        &Mat::from_slice(&[95u8, 255u8, 255u8])?,
        &mut mask2,
    )?;

    let mut mask = Mat::default();
    cv::bitwise_or(&mask1, &mask2, &mut mask, &no_array())?;

    // Morphological open then close with 5×5 ellipse kernel (1 iteration each)
    let kernel =
        imgproc::get_structuring_element(MORPH_ELLIPSE, Size::new(5, 5), Point::new(-1, -1))?;
    let bval = imgproc::morphology_default_border_value()?;
    imgproc::morphology_ex(
        &mask.clone(),
        &mut mask,
        MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        bval,
    )?;
    imgproc::morphology_ex(
        &mask.clone(),
        &mut mask,
        MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        bval,
    )?;

    // Keep components with area >= min_area; discard small noise
    let area_total = rgb.rows() * rgb.cols();
    let min_area = 64_i32.max(area_total / 100);

    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let n = imgproc::connected_components_with_stats(
        &mask,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        CV_32S,
    )?;

    let mut keep =
        Mat::new_rows_cols_with_default(rgb.rows(), rgb.cols(), CV_8UC1, Scalar::all(0.0))?;

    for i in 1..n {
        // CC_STAT_AREA == column 4
        let area = *stats.at_2d::<i32>(i, 4)?;
        if area >= min_area {
            // Build mask for pixels belonging to this component
            let i_full = Mat::new_rows_cols_with_default(
                labels.rows(),
                labels.cols(),
                CV_32S,
                Scalar::all(i as f64),
            )?;
            let mut comp = Mat::default();
            cv::compare(&labels, &i_full, &mut comp, cv::CMP_EQ)?;
            cv::bitwise_or(&keep.clone(), &comp, &mut keep, &no_array())?;
        }
    }

    // If nothing passed the area threshold, use entire frame
    if cv::count_non_zero(&keep)? == 0 {
        let full = Mat::new_rows_cols_with_default(
            rgb.rows(),
            rgb.cols(),
            CV_8UC1,
            Scalar::all(255.0),
        )?;
        return Ok((rgb, full));
    }

    // Smooth edges then re-binarize
    imgproc::gaussian_blur(
        &keep.clone(),
        &mut keep,
        Size::new(5, 5),
        0.0,
        0.0,
        BORDER_DEFAULT,
    )?;
    imgproc::threshold(&keep.clone(), &mut keep, 0.0, 255.0, THRESH_BINARY)?;

    let mut segmented = Mat::default();
    cv::bitwise_and(&rgb, &rgb, &mut segmented, &keep)?;

    Ok((segmented, keep))
}

/// Replace background pixels in a gray image with the mean foreground gray value.
/// Mirrors Python _fill_background_gray.
pub fn fill_background_gray(gray: &Mat, mask: &Mat) -> Result<Mat> {
    let mut g = gray.clone();
    if cv::count_non_zero(mask)? > 0 {
        let mean_val = cv::mean(gray, mask)?[0].round() as u8;
        let mut inv = Mat::default();
        cv::bitwise_not(mask, &mut inv, &no_array())?;
        g.set_to(&Scalar::all(mean_val as f64), &inv)?;
    }
    Ok(g)
}

/// Mean V channel of HSV over foreground pixels (or all pixels if mask is empty).
/// Mirrors Python _mean_v_hsv.
pub fn mean_v_hsv(segmented_rgb: &Mat, mask: &Mat) -> Result<f64> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(segmented_rgb, &mut hsv, COLOR_RGB2HSV, 0)?;
    let mut v = Mat::default();
    cv::extract_channel(&hsv, &mut v, 2)?;
    let mean = if cv::count_non_zero(mask)? > 0 {
        cv::mean(&v, mask)?[0]
    } else {
        cv::mean(&v, &no_array())?[0]
    };
    Ok(mean)
}

/// Laplacian variance over foreground pixels (or all pixels if mask is empty).
/// Mirrors Python _laplacian_var.
#[allow(dead_code)]
pub fn laplacian_var(gray: &Mat, mask: &Mat) -> Result<f64> {
    let mut lap = Mat::default();
    imgproc::laplacian(gray, &mut lap, cv::CV_64F, 1, 1.0, 0.0, BORDER_DEFAULT)?;
    let mut mean_mat = Mat::default();
    let mut std_mat = Mat::default();
    if cv::count_non_zero(mask)? > 0 {
        cv::mean_std_dev(&lap, &mut mean_mat, &mut std_mat, mask)?;
    } else {
        cv::mean_std_dev(&lap, &mut mean_mat, &mut std_mat, &no_array())?;
    }
    let std = *std_mat.at::<f64>(0)?;
    Ok(std * std)
}
