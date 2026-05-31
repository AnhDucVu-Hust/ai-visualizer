# Deploy bằng Docker trên VPS

## Chỉ Dockerfile (không compose) — khuyến nghị nếu bạn build 1 image

→ **[README-dockerfile.md](README-dockerfile.md)**  
`docker build` + `docker run` + nginx trên host (`ai-visualizer.conf.example`, `127.0.0.1:8000`).

Script: `bash deploy/docker/run.sh`

---

## Docker Compose (tùy chọn, 2 container)

Phần dưới dùng `docker-compose.yml` + `nginx.docker.conf` (`proxy_pass` → `app:8000`).

## `proxy_pass` là gì?

**Nginx** là cổng ra internet (port **80/443**). **FastAPI (uvicorn)** chạy bên trong, không cần mở port 8000 ra ngoài.

```
Internet  →  VPS:80 (container nginx)
                 proxy_pass  →  app:8000 (container FastAPI, mạng Docker nội bộ)
```

| Khái niệm | Ý nghĩa |
|-----------|---------|
| `upstream ai_visualizer_app { server app:8000; }` | Đặt tên backend; `app` = tên service trong `docker-compose.yml` |
| `proxy_pass http://ai_visualizer_app;` | Nginx **chuyển tiếp** request HTTP sang container đó |
| `127.0.0.1:8000` | Chỉ dùng khi nginx cài **trên host**, app chạy **trên host** (`ai-visualizer.conf.example`) |
| `app:8000` | Dùng khi **cả nginx + app** trong Docker Compose (`nginx.docker.conf`) |

User gọi `https://domain/api/health` → nginx nhận → gửi copy request tới uvicorn → nhận response → trả lại browser.

**Tải video lớn:** request `/api/jobs/.../artifact` vẫn qua `proxy_pass` để FastAPI **kiểm tra job**; sau đó app trả `X-Accel-Redirect` và nginx **đọc file từ volume** `app_temp` (không stream 500MB qua Python).

## Chạy trên VPS

### 1. Chuẩn bị

```bash
# Trên VPS: cài Docker + Compose plugin
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER   # logout/login lại
```

Clone repo, tạo `.env` từ `.env.example` (API keys, …).

### 2. Build và chạy

```bash
cd /opt/ai-visualizer
docker compose up -d --build
```

- **Port 80** → container `nginx`
- Container `app` **không** publish 8000 ra host (chỉ `expose` trong mạng Compose)

### 3. Kiểm tra

```bash
curl -s http://YOUR_VPS_IP/api/health
# {"status":"ok"}
```

### 4. HTTPS (tùy chọn)

Cách đơn giản: reverse proxy thêm (Caddy / nginx trên host) hoặc mount cert vào container nginx. Với production nên dùng domain + Let's Encrypt.

## Cấu trúc Compose

```yaml
app:     build Dockerfile, volume app_temp → /app/temp, NGINX_ACCEL_ENABLED=1
nginx:   port 80, cùng volume app_temp (read-only), proxy_pass → app:8000
```

Volume `app_temp` **bắt buộc** để nginx đọc được file MP4 khi X-Accel-Redirect.

## Frontend

Image hiện tại chỉ **backend** (+ ffmpeg). Nếu có `frontend/dist`:

- Build local: `cd frontend && npm run build`
- Mount vào app: thêm volume `./frontend/dist:/app/frontend/dist:ro` (backend tự serve nếu thư mục tồn tại), hoặc
- Host frontend riêng (Vercel) trỏ API tới domain VPS.

## Lệnh hữu ích

```bash
docker compose logs -f app
docker compose logs -f nginx
docker compose restart app
docker compose down
```

## So với nginx cài trên host

| | Docker Compose | Nginx trên host |
|--|----------------|-----------------|
| Config | `deploy/nginx/nginx.docker.conf` | `deploy/nginx/ai-visualizer.conf.example` |
| upstream | `app:8000` | `127.0.0.1:8000` |
| temp path | volume `/app/temp` | `alias /opt/.../temp/` |

Cả hai đều cần `NGINX_ACCEL_ENABLED=1` trong `.env` của app.
