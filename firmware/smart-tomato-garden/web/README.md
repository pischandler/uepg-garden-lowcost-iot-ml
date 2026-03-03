# Web UI — Smart Tomato Garden

Single-page UI served by the ESP32 (gzipped HTML in PROGMEM). Runtime: vanilla JS only. Types and contracts live in TypeScript for scalability.

## TypeScript (types, enums, contracts)

- **`src/types.ts`**: Interfaces for API payloads and UI state (`Health`, `Sensors`, `DashboardPayload`, `LastInfer`, `InferenceViewModel`, etc.). Single source of truth; align with firmware endpoints.
- **`src/enums.ts`**: Stream profile and inference status constants; resolution thresholds (`HIGH_RES_STREAM_FS`, `MID_RES_STREAM_FS`).
- **`src/api-contracts.ts`**: API paths and response types (`API_DASHBOARD_PATH`, `DashboardResponse`, `InferenceSchemaResponse`).

Run from `web/`:

```bash
npm install
npm run build    # emits dist/*.d.ts (declarations only; no runtime change)
npm run typecheck  # type-check without emitting
```

The pack step does **not** depend on Node; it only concatenates the existing `js/*.js` files. Use the generated `dist/*.d.ts` when converting modules to TypeScript or for IDE/type-checking. New API fields or endpoints should be reflected in `src/types.ts` and `src/api-contracts.ts`.

## Build (pack)

The UI is packed by `tools/pack_webui.py` (run automatically on PlatformIO build). It:

1. Reads `index.template.html` and `styles.css`
2. Injects i18n dicts and scripts in **fixed order** (see below)
3. Outputs `index.html` and `index.html.gz`; the firmware serves the gzip

Do **not** edit `index.html` by hand — edit the template and the source files under `js/` and `i18n/`.

## Script order and dependencies

Scripts are concatenated in this order. Dependencies are on `window` globals from earlier scripts.

| Order | File        | Exposes        | Depends on                    |
|-------|-------------|----------------|-------------------------------|
| 1     | api.js      | STGApi         | —                             |
| 2     | i18n.js     | STGI18n        | —                             |
| 3     | formatters.js | STGFmt       | STGI18n                       |
| 4     | mappers.js  | STGMap         | STGI18n                       |
| 5     | inference.js| STGInference   | STGMap, STGI18n               |
| 6     | render.js   | STGRender      | STGI18n, STGFmt, STGMap       |
| 7     | store.js    | STGStore       | —                             |
| 8     | app.js      | STGActions     | All above                     |

- **api.js**: `jget`, `jpost`, `getDashboard`, `refreshPayload` (prefers `/api/dashboard`, fallback to 7 GETs).
- **store.js**: Central state + `updateState(partial)` and `subscribe(fn)`. Subscribers run after each merge; app.js registers one that calls setSystem, setInference, setDashboard (when overview visible), setRaw.
- **app.js**: Boot, polling (with backoff on failure), stream/snapshot loop, bindUi; exposes named actions on `STGActions`.

When adding a new module, add it to `js_order` in `tools/pack_webui.py` and keep the dependency order (e.g. anything using STGRender must come after render.js).

## Data flow

1. **Polling**: `refreshPayload()` → one GET `/api/dashboard` (or 7 GETs on fallback) → `Store.updateState({ ...payload, hist })` → subscribers run → UI updated.
2. **Snapshot stream**: Next frame is scheduled only after the current image `load` or `error` (no overlapping requests).
3. **Dashboard**: Sparklines and overview are redrawn only when the overview panel is visible; changing the trend range triggers `Store.updateState({})` to redraw without refetch.

## i18n

Keys in `i18n/pt.json`, `en.json`, `cn.json`. Use `data-i18n="key"` in HTML or `t("key")` in JS. New keys must exist in all three files or the pack step fails.

## Adding a new panel/card

1. Add markup in `index.template.html` (e.g. a new `.card` with `data-panel-card`).
2. In `render.js`, add a `setXxx(state)` (or extend an existing setter) that updates the DOM from `state`.
3. In `app.js`, register a store subscriber that calls your setter when relevant state keys change, or reuse the existing subscriber and have it call your setter with `Store.getState()`.
