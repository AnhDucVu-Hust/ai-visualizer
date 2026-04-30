# AI Visualizer Studio

Turn an audio narration into a fully-rendered video, with the expensive parts
(images) generated manually by you. The app wraps three steps into one UI:

1. **Audio → prompts** — transcribe the audio (faster-whisper), merge the
   transcript into scenes of `min_duration..max_duration` seconds, and ask an
   LLM (Gemini / Groq / OpenAI-compatible / OpenRouter) to write a cinematic
   image prompt for every scene. Outputs `results/scenes.json` +
   `results/prompts.txt` + `results/out.json`.
2. **You generate the images manually.** Save them in a folder, and prefix
   every filename with its scene number — e.g. `01_opening.png`,
   `02_newsroom.jpg`, `03_sunset.webp`, …
3. **Images + audio → video.** Combine the image folder with `scenes.json`
   (each image is held for its scene's duration with a Ken-Burns zoom/pan),
   mux in the original narration and, optionally, multiple background music
   tracks with arbitrary start/end/volume.

## Architecture

- `backend/` — FastAPI server that wraps the existing Python pipelines
  (`stt/`, `scene/`, `video_combine/`). Jobs run on a background thread and
  report live progress counters (scene 3/10, clip 7/42, …) by polling.
- `frontend/` — React + Vite + TypeScript desktop-style app with two tabs
  (**Prompts** and **Video**) and a music editor that auto-defaults each new
  track to start where the previous track ends and play to the end of the video.
- `config.yaml` / `.env` — still the single source of truth for your LLM
  provider and API keys. The UI just triggers the pipeline.
- `.app_state.json` — small JSON at the repo root that remembers the last
  audio file, images folder, output paths, etc. so the **Video** tab can
  auto-fill from the **Prompts** tab across app restarts.

## One-time setup

```bash
# Python backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# React frontend
cd frontend
npm install
cd ..
```

Make sure `config.yaml` and `.env` are configured (LLM provider, keys, etc.) —
the same way the existing `generate_scenes.py` CLI uses them.

## Run it

Two terminals:

```bash
# terminal 1 — backend (listens on 127.0.0.1:8000)
source venv/bin/activate
python run_app.py
```

```bash
# terminal 2 — frontend (listens on 127.0.0.1:5173, proxies /api → :8000)
cd frontend
npm run dev
```

Open http://localhost:5173 in a browser.

### Production build (optional)

If you run `npm run build` inside `frontend/`, the backend will serve the
static bundle from `frontend/dist/` at the same port as the API, so you only
need to run `python run_app.py` and visit http://127.0.0.1:8000/.

## Deploy backend free on Render

This repo includes:

- `Dockerfile` (Python + ffmpeg)
- `render.yaml` (web service blueprint)

### Steps

1. Push this repo to GitHub.
2. In Render, create a new Blueprint and point to your repo.
3. Render will detect `render.yaml` and create `ai-visualizer-backend`.
4. Set required environment variables in Render dashboard (API keys, etc.).
5. Deploy and verify:

```bash
curl https://<your-render-domain>/api/health
```

You should get:

```json
{"status":"ok"}
```

### Frontend/Desktop client config

Point your local frontend/app to the Render API base URL (instead of localhost).
For example, if you centralize API base in frontend config, set it to:

`https://<your-render-domain>`

## Important: arbitrary local paths do NOT work with cloud backend

When backend is on cloud, a local path like `/Users/alice/Desktop/scenes` only
exists on the user's machine, not on the cloud server. So the server cannot read
that path directly.

For cloud mode, the correct flow is:

1. Client picks local files/folders.
2. Client uploads data to backend (or cloud storage).
3. Backend processes uploaded files on server disk.

### Cloud-friendly video render endpoint (no local paths)

For cloud deployment, use:

- `POST /api/video/ffmpeg/jobs/upload`

Multipart form fields:

- `scene_bundle` (required): zip containing `scenes.json`, `out.json`, and images
- `include_narration` (bool, default `true`)
- `narration_file` (required when `include_narration=true`)
- optional render params: `resolution`, `fps`, `preset`, `threads`, `crf`, ...

After job is done, download rendered video via:

- `GET /api/jobs/{job_id}/artifact`

## 10 customers running in parallel: upload-then-delete is OK?

Yes, this is a normal pattern, as long as each job is isolated.

Use this policy:

- Per-job working directory (e.g. `/tmp/jobs/<job_id>/...`).
- Never share input/output files between jobs.
- Delete only that job directory after completion/failure.
- Add TTL cleanup for abandoned jobs (for example: delete after 1-6 hours).

This is safe for parallel users and keeps disk usage bounded.
On free tiers, still expect limits (CPU/RAM/disk/timeouts), so keep files small
for MVP and move to paid plan when concurrency grows.

## Workflow

1. **Prompts tab**: upload the audio file, set min/max scene duration, pick an
   output folder, click **Generate prompts**. The progress bar shows
   "Generating prompt 4/27…" while the LLM works.
2. Open `results/prompts.txt` and generate an image for each prompt (any tool
   you like). Save them into a folder, prefixed with the scene number.
3. **Video tab**: enter the images folder. The audio path is pre-filled from
   step 1. Add any music tracks (their start/end auto-default). Click
   **Build video**.

## CLI is still fully supported

The original CLIs are untouched:

```bash
python generate_scenes.py --config config.yaml
python -m video_combine.combine --images path/to/images --audio audio/script.wav --output results/video.mp4
```
