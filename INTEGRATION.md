# Frontend ↔ Backend Integration

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # optional
python run_server.py          # or: start_server.bat / ./start_server.sh
```

Then open **http://localhost:8000** — the FastAPI app serves the UI and the API
on the same origin, so the frontend's relative `/api/...` calls just work with
no CORS configuration and no hard-coded host.

| URL | Serves |
|---|---|
| `/` | `frontend/index.html` |
| `/pages/psychology/index.html` | Psychology page |
| `/pages/custom-industry/index.html` | Custom Industry page |
| `/api/generate-captions`, `/api/caption` | Caption pipeline |
| `/api/digital-twin/analyze` | Digital Twin tab |
| `/api/status`, `/health` | Model + API state |
| `/docs` | Swagger UI |

If you prefer to serve the UI separately (Live Server, Vite), the pages detect
ports 5500/5501/3000/5173/8080 and `file://` and fall back to
`http://localhost:8000`; CORS is already enabled for that case. Change
`window.VV_FALLBACK_API_ORIGIN` in `frontend/index.html` to point elsewhere.

## What was broken

1. **`backend/config/settings.py` contained a copy of `model_loader.py`** and
   imported itself, so `from backend.config.settings import settings` failed and
   the app could not start at all. Rewritten as a real settings module.
2. **`generateCaptions()` used `audience`, `tone` and `language` — undefined
   globals.** The `ReferenceError` fired before `fetch`, so *every* run fell into
   the `catch` and showed demo captions. The UI looked connected but never called
   the backend. Now read via `getGenerationContext()`.
3. **Response shapes did not match.** The backend returned one flat object; the
   UI wanted an array of `{type, caption}`. Even a successful request hit
   "No captions returned". The backend now returns `captions[]` *and* the
   original flat fields.
4. **The JSON `image_url` path had no server side.** When an image host blocks
   browser CORS the UI posts JSON; the route only accepted multipart and
   returned 422. Both are handled now.
5. **No CORS and no static serving**, so the two halves could only meet if you
   manually proxied them.
6. **`/api/digital-twin/analyze` did not exist** (the page referenced a
   `digital_twin_model.py` that was never written) — added as an honest
   placeholder.

## Request / response contract

`POST /api/generate-captions` — multipart (`image`, `industry`, `audience`,
`tone`, `language`) **or** JSON `{image_url, industry, audience, tone, language}`.

```jsonc
{
  "success": true,
  "model_loaded": true,
  "captions": [ { "type": "Professional", "caption": "..." }, ... ],
  "caption": "...",            // first variant, original flat contract
  "hashtags": ["#VisionVerse", ...],
  "cta": "Shop the look",
  "marketing_strategy": "...",
  "recommendations": ["..."],
  "image_analysis": { "description": "...", "features": {...}, "source": "..." }
}
```

Errors return `{"detail": "..."}` with 400/413/502/500; the UI surfaces the
detail text and still falls back to demo captions so the page stays usable.

## About the model weights

`weights/visionverse_caption/` holds `config.json`, the tokenizer and the
processor config, but **no weight file** (`model.safetensors` /
`pytorch_model.bin`). The loader detects this and falls back to
`settings.BLIP_MODEL` from the Hub, reporting which source it used in
`/api/status`. Drop your fine-tuned `model.safetensors` into that folder and it
is preferred automatically — no code change. Set `ALLOW_HUB_FALLBACK=false` if
you would rather fail loudly than run the base model.

Verified end to end: real BLIP captions returned for both the multipart and
`image_url` paths, with `model_loaded: true`.

## Still mock

`pages/psychology/index.html` keeps `MOCK_MODE:true` — it documents an EEG
contract (`/eeg/upload`, `/eeg/analyze`, `/eeg/report/generate`, a websocket)
and no EEG model exists in this project. Its `API_BASE` now resolves to the same
origin, so implement those routes and flip the flag. The Digital Twin endpoint
is live but returns a placeholder for the same reason.
