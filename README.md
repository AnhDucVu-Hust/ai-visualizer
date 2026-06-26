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



## CLI is still fully supported

The original CLIs are untouched:

```bash
python generate_scenes.py --config config.yaml
python -m video_combine.combine --images path/to/images --audio audio/script.wav --output results/video.mp4
```
