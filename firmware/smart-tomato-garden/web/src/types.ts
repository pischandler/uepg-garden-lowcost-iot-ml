/**
 * Shared types for STG web UI — single source of truth for API and UI state.
 * Align with firmware endpoints: /health, /api/sensors, /api/irrigation, /status,
 * /metrics, /api/config, /api/inference/last, /api/dashboard, /api/inference/schema.
 */

export interface Health {
  ip?: string;
  rssi?: number;
  online?: boolean;
  device_id?: string;
  heap?: number;
  psram?: number;
  uptime_ms?: number;
  stream_clients?: number;
  api_version?: string;
}

export interface Sensors {
  ts_ms?: number;
  age_ms?: number;
  soil_raw?: number;
  lux_raw?: number;
  soil_pct?: number;
  temp_c?: number;
  hum_pct?: number;
  dht_ok?: boolean;
  api_version?: string;
}

export interface Irrigation {
  pump_on?: boolean;
  pump_until_ms?: number;
  remaining_ms?: number;
  last_run_ms?: number;
  auto_enabled?: boolean;
  cooldown_remaining_ms?: number;
  soil_dry_threshold_pct?: number;
  pump_on_ms?: number;
  api_version?: string;
}

export interface CameraStatus {
  quality?: number;
  framesize?: number;
  led_intensity?: number;
  api_version?: string;
}

export interface Metrics {
  http?: number;
  capture?: number;
  stream_clients?: number;
  mqtt_pub?: number;
  mqtt_fail?: number;
  logs?: number;
  infer_attempt?: number;
  infer_ok?: number;
  infer_fail?: number;
  api_version?: string;
}

export interface RuntimeConfig {
  soil_dry_threshold_pct?: number;
  pump_on_ms?: number;
  pump_cooldown_ms?: number;
  soil_raw_dry?: number;
  soil_raw_wet?: number;
  led_duty?: number;
  led_on_stream?: boolean;
  cam_quality?: number;
  cam_framesize?: number;
  store_events?: boolean;
  telemetry_enabled?: boolean;
  infer_enabled?: boolean;
  infer_skip_when_streaming?: boolean;
  infer_period_ms?: number;
  infer_host?: string;
  infer_port?: number;
  infer_path?: string;
  infer_min_lux_raw?: number;
  infer_use_led?: boolean;
  infer_max_retries?: number;
  infer_retry_delay_ms?: number;
}

export interface LastInfer {
  ok?: boolean;
  http_status?: number;
  latency_ms?: number;
  ts_ms?: number;
  predicted?: string;
  confidence?: number;
  confident?: boolean;
  reasons?: string | string[];
  model_version?: string;
  raw?: string;
}

export interface DashboardPayload {
  health?: Health;
  sensors?: Sensors;
  irrigation?: Irrigation;
  camera?: CameraStatus;
  metrics?: Metrics;
  config?: RuntimeConfig;
  lastInfer?: LastInfer;
}

export interface InferenceSchema {
  classes?: string[];
  reasons?: string[];
}

export interface InferenceViewModel {
  status: string;
  labelFriendly: string;
  labelRaw: string;
  confidencePct: number | null;
  confidenceBadge: string;
  latencyMs: number | null;
  httpStatus: number | null;
  reasons: string[];
  topk: Array<{ labelFriendly: string; labelRaw: string; scorePct: number | null }>;
  context: {
    tempC: number | null;
    humPct: number | null;
    luxRaw: number | null;
    soilPct: number | null;
    pumpOn: boolean | null;
  };
  rawPayload: unknown;
  tsMs: number | null;
}
