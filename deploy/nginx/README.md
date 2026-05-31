# Nginx + X-Accel-Redirect (VPS, file video lớn)

**Hướng dẫn cài đặt đầy đủ (Docker + nginx):** **[INSTALL.md](INSTALL.md)**

**Deploy bằng Docker Compose:** xem **[deploy/docker/README.md](../docker/README.md)**  
(dùng `nginx.docker.conf` với `proxy_pass` → `app:8000`, không dùng file example dưới đây).

---

## `proxy_pass` (nginx trên host)

Nginx lắng nghe port 80 công khai. Mỗi request `/api/...` được **chuyển tiếp** tới uvicorn:

```nginx
upstream ai_visualizer_app {
    server 127.0.0.1:8000;   # FastAPI trên cùng VPS
}
location / {
    proxy_pass http://ai_visualizer_app;   # = gửi request sang :8000
}
```

User **không** truy cập trực tiếp `:8000` — chỉ nginx ra internet.

---

Khi video **500MB+**, để nginx gửi file trực tiếp từ ổ đĩa (`sendfile`) thay vì stream qua uvicorn:

1. Client vẫn gọi `GET /api/jobs/{job_id}/artifact` (FastAPI kiểm tra job + `client_key`).
2. FastAPI trả header `X-Accel-Redirect: /internal-temp/video/….mp4`.
3. Nginx đọc file từ `temp/` và trả cho client.

## Luồng

```
Browser → nginx:443 → uvicorn (auth) → X-Accel-Redirect
                ↘ nginx internal location → sendfile → Browser
```

## Bước cài trên VPS

### 1. Cài nginx

```bash
sudo apt update && sudo apt install -y nginx
```

### 2. Chạy backend (uvicorn chỉ listen localhost)

```bash
cd /opt/ai-visualizer   # đường dẫn repo của bạn
source venv/bin/activate
python run_app.py --host 127.0.0.1 --port 8000
```

Production nên dùng **systemd** hoặc **supervisor** để giữ process.

### 3. Cấu hình nginx

```bash
sudo cp deploy/nginx/ai-visualizer.conf.example /etc/nginx/sites-available/ai-visualizer
sudo nano /etc/nginx/sites-available/ai-visualizer
```

Sửa hai chỗ quan trọng:

| Trong file | Giá trị |
|------------|---------|
| `alias` trong `location /internal-temp/` | **Đường dẫn tuyệt đối** tới `…/repo/temp/` (có dấu `/` cuối) |
| `server_name` | domain của bạn |
| `proxy_pass` upstream | port backend (mặc định 8000) |

```nginx
location /internal-temp/ {
    internal;
    alias /opt/ai-visualizer/temp/;   # ← phải trùng thư mục temp/ của app
}
```

```bash
sudo ln -sf /etc/nginx/sites-available/ai-visualizer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Bật accel trong `.env`

```bash
NGINX_ACCEL_ENABLED=1
# Mặc định khớp location trên; đổi nếu bạn đổi path nginx:
# NGINX_ACCEL_INTERNAL_PREFIX=/internal-temp/
```

Khởi động lại backend sau khi sửa `.env`.

### 5. Kiểm tra

```bash
# Health qua nginx
curl -s https://your-domain/api/health

# Sau khi job video xong, tải thử (thay job_id, client_key)
curl -L -o test.mp4 "https://your-domain/api/jobs/JOB_ID/artifact?client_key=KEY" -w '%{http_code} %{size_download}\n'
```

Nếu **bật** `NGINX_ACCEL_ENABLED` mà **chưa** cấu hình `location /internal-temp/` → tải sẽ lỗi (502/404). Local dev: **không** set biến này.

## HTTPS (khuyến nghị)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.example
```

## Lưu ý

- Chỉ file nằm trong `temp/` mới dùng accel (video upload, `temp/video/{job_id}.mp4`, …).
- `VIDEO_JOB_TTL_SECONDS`: user phải tải xong trước khi file bị xóa.
- Băng thông VPS vẫn là giới hạn chính; nginx giúp **giảm CPU** và ổn định hơn, không tăng magic băng thông mạng.
- Muốn giảm dung lượng file: `VIDEO_CRF`, resolution thấp hơn trong pipeline ffmpeg.

## Không dùng nginx

Để trống `NGINX_ACCEL_ENABLED` — app dùng `FileResponse` như trước (phù hợp dev local).
