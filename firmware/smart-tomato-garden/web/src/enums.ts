/**
 * Enums and constants for STG web UI — stream profile, inference status, resolution thresholds.
 */

export const StreamProfile = {
  Auto: "auto",
  Stable: "stable",
  Fast: "fast",
} as const;

export type StreamProfileType = (typeof StreamProfile)[keyof typeof StreamProfile];

export const InferenceStatus = {
  Empty: "empty",
  Ok: "ok",
  LowConfidence: "low_confidence",
  Fail: "fail",
} as const;

export type InferenceStatusType = (typeof InferenceStatus)[keyof typeof InferenceStatus];

/** Framesize index >= this uses high-res stream (e.g. VGA+). */
export const HIGH_RES_STREAM_FS = 8;

/** Framesize index >= this uses mid-res (e.g. CIF+). */
export const MID_RES_STREAM_FS = 6;
