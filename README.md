# AI Visualizer Studio

Turn an audio narration into a fully-rendered video, with the expensive parts
(images) generated manually by you. The app wraps three steps into one UI:

1. **Audio → prompts** — transcribe the audio (faster-whisper), merge the
   transcript into scenes of `min_duration..max_duration` seconds, and ask an
   LLM (Gemini / Groq / OpenAI-compatible / OpenRouter) to write a cinematic
   image prompt for every scene. Outputs `results/scenes.json` +
   `results/prompts.txt` + `results/out.json`.
2. **You generate the images manually.** Save them in a folder, and prefix
   every filename with its scene number — e.g. `001.png`,
   `002.png`, `003.png`, …
   This setting allows you to use Google Flow for free media generation. Currently, API calls for image generation are not supported (cost-saving measure).
4. **Images + audio → video.** Combine the image folder with `scenes.json`
   (each image is held for its scene's duration with a Ken-Burns zoom/pan),
   mux in the original narration and, optionally, multiple background music
   tracks with arbitrary start/end/volume.


## CLI is still fully supported

The original CLIs are untouched:

```bash
python generate_scenes.py --config config.yaml
python -m video_combine.combine --images path/to/images --audio audio/script.wav --output results/video.mp4
```
