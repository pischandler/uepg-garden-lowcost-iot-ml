/**
 * API paths and response contracts — align with firmware camera_server and inference_client.
 */

import type { DashboardPayload, InferenceSchema } from "./types";

/** Base path for aggregated dashboard (preferred over multiple GETs). */
export const API_DASHBOARD_PATH = "/api/dashboard";

/** Schema endpoint for dynamic class/reason labels. */
export const API_INFERENCE_SCHEMA_PATH = "/api/inference/schema";

/** Single endpoints used when dashboard is unavailable (fallback). */
export const API_PATHS = {
  health: "/health",
  sensors: "/api/sensors",
  irrigation: "/api/irrigation",
  status: "/status",
  metrics: "/metrics",
  config: "/api/config",
  inferenceLast: "/api/inference/last",
  inferenceRun: "/api/inference/run",
  inferenceConfig: "/api/inference/config",
  capture: "/capture",
  stream: "/stream",
} as const;

/**
 * Response of GET /api/dashboard.
 * Use this type when calling getDashboard() or refreshPayload().
 */
export type DashboardResponse = DashboardPayload;

/**
 * Response of GET /api/inference/schema.
 * Use when initializing mappers (STGMap.init).
 */
export type InferenceSchemaResponse = InferenceSchema;
