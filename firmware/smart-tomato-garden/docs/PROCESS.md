# Smart Tomato Garden Process

## Definition Of Done
- API contract unchanged or versioned with explicit migration notes.
- `python tools/pack_webui.py` passes (includes i18n key validation).
- Firmware build passes for `esp32-s3-devkitc-1`.
- Size budget check passes.
- Mobile screenshot captured (camera + diagnosis + controls).
- Regression smoke executed:
  - stream starts
  - inference can be requested
  - irrigation start/stop endpoints respond

## Weekly Reliability KPIs
- Inference success rate: `infer_ok / infer_attempt`.
- Stream stability: reconnect count per hour and freeze incidents.
- Mean inference latency (`latency_ms` from `/api/inference/last` and CSV).
- Pump command success ratio (`/api/irrigation/start|stop` ok responses).

## RFC Flow (1 page max)
- Problem statement and scope.
- Constraints and non-goals.
- Options considered and tradeoffs.
- Chosen approach and rollback plan.
- Validation plan (metrics/tests).

Use: `docs/rfcs/RFC_TEMPLATE.md`.
