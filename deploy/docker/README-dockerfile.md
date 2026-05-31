# Chỉ dùng Dockerfile (không docker compose)

Image `Dockerfile` chỉ chạy **FastAPI + ffmpeg**. Nginx đặt **trên VPS** (không trong container).

## Luồng

```
Internet → nginx trên host :80
              proxy_pass → 127.0.0.1:8000  (container publish port)
              X-Accel-Redirect → đọc file từ thư mục temp trên host (volume mount)
```

| File nginx | Dùng khi |
|------------|----------|
| `deploy/nginx/ai-visualizer.conf.example` | **Có** — Dockerfile + nginx host |
| `deploy/nginx/nginx.docker.conf` | **Không** — chỉ cho docker compose 2 container |

## 1. Build image

```bash
cd /opt/ai-visualizer
cp .env.example .env   # điền API keys
docker build -t ai-visualizer .
```

## 2. Chạy container

```bash
mkdir -p /opt/ai-visualizer/temp

docker run -d \
  --name ai-visualizer \
  --restart unless-stopped \
  --env-file .env \
  -e NGINX_ACCEL_ENABLED=1 \
  -p 127.0.0.1:8000:8000 \
  -v /opt/ai-visualizer/temp:/app/temp \
  ai-visualizer
```

Hoặc script:

```bash
bash deploy/docker/run.sh
```

- `-p 127.0.0.1:8000:8000` — chỉ máy VPS gọi được (nginx proxy vào đây).
- `-v .../temp:/app/temp` — video MP4 nằm trên host để nginx `alias` đọc khi tải lớn.

Kiểm tra:

```bash
curl -s http://127.0.0.1:8000/api/health
```

## 3. Nginx trên host

```bash
sudo cp deploy/nginx/ai-visualizer.conf.example /etc/nginx/sites-available/ai-visualizer
sudo nano /etc/nginx/sites-available/ai-visualizer
```

Sửa:

```nginx
upstream ai_visualizer_app {
    server 127.0.0.1:8000;   # container đã publish như trên
}

location /internal-temp/ {
    internal;
    alias /opt/ai-visualizer/temp/;   # TRÙNG TEMP_HOST_DIR khi docker run
}
```

```bash
sudo ln -sf /etc/nginx/sites-available/ai-visualizer /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

User truy cập `http://domain/api/...` qua port **80**, không cần mở **8000** ra internet.

## Không dùng nginx (dev / thử nhanh)

```bash
docker run -d --name ai-visualizer --env-file .env -p 8000:8000 ai-visualizer
# Không set NGINX_ACCEL_ENABLED (hoặc =0)
curl http://VPS_IP:8000/api/health
```

Tải video lớn sẽ chậm hơn (stream qua Python). Production nên có nginx + volume `temp`.

## Lệnh quản lý

```bash
docker logs -f ai-visualizer
docker restart ai-visualizer
docker stop ai-visualizer && docker rm ai-visualizer
```

## `docker-compose.yml`

Tùy chọn (2 container nginx+app). Nếu chỉ Dockerfile thì **bỏ qua** compose.
