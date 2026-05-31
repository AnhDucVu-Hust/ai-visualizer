#!/usr/bin/env bash
# Chạy image build từ Dockerfile (không dùng docker compose).
# Usage (từ thư mục gốc repo):
#   cp .env.example .env && nano .env
#   bash deploy/docker/run.sh
#
# Yêu cầu: nginx trên VPS + deploy/nginx/ai-visualizer.conf.example
# (proxy_pass 127.0.0.1:8000, alias trùng TEMP_HOST_DIR bên dưới).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

IMAGE_NAME="${IMAGE_NAME:-ai-visualizer}"
CONTAINER_NAME="${CONTAINER_NAME:-ai-visualizer}"
# Thư mục trên HOST — nginx alias phải trỏ đúng path này (X-Accel-Redirect)
TEMP_HOST_DIR="${TEMP_HOST_DIR:-${HOME}/ai-visualizer/temp}"

mkdir -p "$TEMP_HOST_DIR"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy from .env.example and set API keys." >&2
  exit 1
fi

docker build -t "$IMAGE_NAME" .

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file .env \
  -e NGINX_ACCEL_ENABLED=1 \
  -p 127.0.0.1:8000:8000 \
  -v "${TEMP_HOST_DIR}:/app/temp" \
  "$IMAGE_NAME"

echo "Container $CONTAINER_NAME listening on 127.0.0.1:8000"
echo "temp volume: $TEMP_HOST_DIR -> /app/temp"
echo "Configure host nginx: alias ${TEMP_HOST_DIR}/ in location /internal-temp/"
echo "Health: curl -s http://127.0.0.1:8000/api/health"
