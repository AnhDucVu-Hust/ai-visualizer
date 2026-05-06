#!/bin/bash
set -e

echo "⚙️ Building binary..."
python -m PyInstaller run_app.py \
  --onefile \
  --paths . \
  --hidden-import=uvicorn \
  --hidden-import=moviepy \
  --collect-all moviepy \
  --collect-all ffmpeg \
  --collect-all imageio \
  --collect-all imageio_ffmpeg \
  --copy-metadata moviepy \
  --copy-metadata imageio \
  --copy-metadata imageio-ffmpeg \
  --hidden-import=fastapi \
  --collect-all fastapi \
  --collect-submodules backend \
  --collect-submodules scene \
  --collect-submodules stt \
  --collect-submodules video_combine

#echo "🐳 Building Docker..."
#docker build -t ai-visualizer-backend .

#echo "📦 Exporting..."
#docker save ai-visualizer-backend > ai-visualizer-backend.tar

echo "✅ Done!"