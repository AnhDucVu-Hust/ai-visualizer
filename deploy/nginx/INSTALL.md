# Cài đặt VPS — Docker (Dockerfile) + Nginx trên host

Hướng dẫn deploy **AI Visualizer** trên VPS Ubuntu/Debian:

- **FastAPI** chạy trong Docker (build từ `Dockerfile` ở thư mục gốc repo)
- **Nginx** cài trực tiếp trên VPS (reverse proxy + tải video lớn)
- Repo mặc định: `**~/ai-visualizer/`**

Không dùng `docker compose` cho flow này. File `nginx.docker.conf` chỉ dành cho compose 2 container.

---

## Kiến trúc

```
Internet :80/:443
    ↓
nginx (trên VPS)
    ├─ proxy_pass → 127.0.0.1:8000 → container Docker (FastAPI)
    └─ X-Accel-Redirect → /opt/ai-visualizer/temp/*.mp4 (sendfile)
```


| Thành phần                | Vị trí                              | Port                                    |
| ------------------------- | ----------------------------------- | --------------------------------------- |
| Nginx                     | Host VPS                            | 80, 443                                 |
| Container `ai-visualizer` | Docker                              | `127.0.0.1:8000` (không mở ra internet) |
| Video tạm                 | `/opt/ai-visualizer/temp/` trên host | mount → `/app/temp` trong container     |

> **Quan trọng:** thư mục temp đặt ở `/opt/ai-visualizer/temp`, **không** để trong `/root`.
> Nginx chạy bằng user `www-data`; `/root` có quyền `700` nên nginx không đọc được file →
> tải video sẽ lỗi **403**. `/opt` mặc định `755` nên `www-data` đọc được.


---

## Yêu cầu

- VPS Ubuntu/Debian
- Docker đã cài
- Git (hoặc copy code lên VPS)
- Domain (tùy chọn, cho HTTPS)

---

## Bước 1 — Code và `.env`

```bash
cd ~
git clone <URL_REPO> ai-visualizer
cd ~/ai-visualizer

cp .env.example .env
nano .env
```

Điền API keys (`OPENAI_API_KEY`, `LLM_*`, …).

Thêm / kiểm tra:

```env
NGINX_ACCEL_ENABLED=1
VIDEO_JOB_TTL_SECONDS=3600
UPLOAD_JOB_TTL_SECONDS=3600
```

```bash
sudo mkdir -p /opt/ai-visualizer/temp
```

---

## Bước 2 — Build Docker image

Từ thư mục gốc repo (có `Dockerfile`):

```bash
cd ~/ai-visualizer
docker build -t ai-visualizer .
```

Image chứa Python 3.11, ffmpeg, FastAPI. Ứng dụng listen **8000** bên trong container (`CMD python run_app.py --host 0.0.0.0`).

---

## Bước 3 — Chạy container

### Cách A — Script (khuyến nghị)

```bash
cd ~/ai-visualizer
bash deploy/docker/run.sh
```

Script sẽ: build image, tạo `/opt/ai-visualizer/temp`, chạy container với volume và `NGINX_ACCEL_ENABLED=1`.

Đổi thư mục temp (nếu cần): `TEMP_HOST_DIR=/path/khac bash deploy/docker/run.sh` — nhớ sửa `alias` nginx cho khớp.

### Cách B — Lệnh tay

```bash
cd ~/ai-visualizer

sudo mkdir -p /opt/ai-visualizer/temp

docker rm -f ai-visualizer 2>/dev/null || true

docker run -d \
  --name ai-visualizer \
  --restart unless-stopped \
  --env-file .env \
  -e NGINX_ACCEL_ENABLED=1 \
  -p 127.0.0.1:8000:8000 \
  -v "/opt/ai-visualizer/temp:/app/temp" \
  ai-visualizer
```

**Giải thích flags:**


| Flag                                | Ý nghĩa                                              |
| ----------------------------------- | ---------------------------------------------------- |
| `-p 127.0.0.1:8000:8000`            | Chỉ localhost VPS gọi được (nginx proxy vào đây)     |
| `-v /opt/ai-visualizer/temp:/app/temp` | Video MP4 trên host để nginx đọc khi tải lớn (ngoài `/root` để tránh 403) |
| `-e NGINX_ACCEL_ENABLED=1`          | App trả `X-Accel-Redirect` thay vì stream qua Python |
| `--restart unless-stopped`          | Tự chạy lại khi reboot VPS                           |


### Kiểm tra app (chưa qua nginx)

```bash
curl -s http://127.0.0.1:8000/api/health
# {"status":"ok"}

docker logs ai-visualizer --tail 50
docker ps
```

---

## Bước 4 — Cài Nginx trên VPS

```bash
sudo apt update
sudo apt install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

---

## Bước 5 — Cấu hình Nginx

### 5.1 Copy file mẫu

```bash
sudo cp ~/ai-visualizer/deploy/nginx/ai-visualizer.conf.example \
        /etc/nginx/sites-available/ai-visualizer
```

### 5.2 Sửa path (nếu cần)

```bash
sudo nano /etc/nginx/sites-available/ai-visualizer
```

**Bắt buộc kiểm tra:**

1. `alias` trong `location /internal-temp/` — phải trùng path volume Docker (mặc định `/opt/ai-visualizer/temp/`):
  ```nginx
   alias /opt/ai-visualizer/temp/;
  ```
   Có `/` cuối. **Không** đặt temp trong `/root` (nginx `www-data` không đọc được → lỗi 403).
2. `**server_name**` — domain hoặc `_` (chấp nhận mọi host / test bằng IP):
  ```nginx
   server_name studio.example.com;
  ```

`**proxy_pass`:** config dùng `upstream ai_visualizer_app { server 127.0.0.1:8000; }` — tương đương proxy thẳng tới container.

### 5.3 Bật site

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/ai-visualizer /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl reload nginx
```

### 5.4 Kiểm tra qua nginx

```bash
curl -s http://127.0.0.1/api/health
curl -s http://IP_HOAC_DOMAIN/api/health
```

---

## Bước 6 — Firewall (ufw)

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Không mở port **8000** ra internet.

---

## Bước 7 — HTTPS (có domain)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d studio.example.com
```

---

## Deploy lại sau khi đổi code

```bash
cd ~/ai-visualizer
git pull

docker build -t ai-visualizer .

docker rm -f ai-visualizer

docker run -d \
  --name ai-visualizer \
  --restart unless-stopped \
  --env-file .env \
  -e NGINX_ACCEL_ENABLED=1 \
  -p 127.0.0.1:8000:8000 \
  -v "/opt/ai-visualizer/temp:/app/temp" \
  ai-visualizer
```

Hoặc: `bash deploy/docker/run.sh`

**Không cần** reload nginx trừ khi đổi domain hoặc path `temp`.

---

## Client / API

Base URL: `http://IP` hoặc `https://domain` (không dùng `:8000` từ ngoài).


| Endpoint                                     | Mục đích                    |
| -------------------------------------------- | --------------------------- |
| `GET /api/health`                            | Kiểm tra sống               |
| `POST /api/video/ffmpeg/jobs/upload`         | Render video (cloud)        |
| `GET /api/jobs/{id}`                         | Poll trạng thái job         |
| `GET /api/jobs/{id}/artifact?client_key=...` | Tải MP4 (sau `stage: done`) |


---

## Checklist hoàn tất

- `curl http://127.0.0.1:8000/api/health` → OK
- `curl http://IP/api/health` → OK
- `alias` nginx = `/opt/ai-visualizer/temp/` (có `/` cuối)
- Docker `-v` trùng path `temp` (`/opt/ai-visualizer/temp:/app/temp`)
- temp **không** nằm trong `/root` (tránh 403)
- `.env` có `NGINX_ACCEL_ENABLED=1`
- `docker ps` → `ai-visualizer` Up

---

## Xác minh tải video lớn

Sau job video xong:

```bash
ls /opt/ai-visualizer/temp/
docker exec ai-visualizer ls /app/temp/

# Quyền đọc của nginx (www-data) — phải KHÔNG báo "Permission denied"
sudo -u www-data ls /opt/ai-visualizer/temp/ && echo "www-data OK"
```

Hai lệnh đầu phải thấy **cùng file**. Nếu host trống → thiếu `-v` khi `docker run`.
Nếu lệnh `www-data` báo `Permission denied` → temp đang nằm chỗ nginx không đọc được (vd `/root`) → sẽ lỗi 403.

---

## Lỗi thường gặp


| Triệu chứng                       | Nguyên nhân                             | Xử lý                                    |
| --------------------------------- | --------------------------------------- | ---------------------------------------- |
| 502 Bad Gateway                   | Container không chạy                    | `docker ps`, `docker logs ai-visualizer` |
| **403 khi tải video**             | nginx (`www-data`) không đọc được file — thường do temp nằm trong `/root` (quyền 700) | Chuyển temp sang `/opt/ai-visualizer/temp` (xem dưới) |
| API OK, tải video lỗi (404)       | `alias` ≠ volume `temp`                 | Sửa nginx + `docker run -v`              |
| 404 artifact                      | Job chưa `done` hoặc file đã bị TTL xóa | Poll job; tăng `VIDEO_JOB_TTL_SECONDS`   |
| Tải lỗi + `NGINX_ACCEL_ENABLED=1` | Nginx chưa có `/internal-temp/`         | Làm lại bước 5                           |

**Khắc phục lỗi 403 (temp đang ở `/root`):**

```bash
sudo mkdir -p /opt/ai-visualizer/temp
cd ~/ai-visualizer
TEMP_HOST_DIR=/opt/ai-visualizer/temp bash deploy/docker/run.sh
sudo sed -i 's#/root/ai-visualizer/temp/#/opt/ai-visualizer/temp/#' /etc/nginx/sites-available/ai-visualizer
sudo nginx -t && sudo systemctl reload nginx
```

Job đã render trước đó (ở `/root/...`) phải combine lại để file mới sinh trong `/opt`.


---

## File liên quan


| File                                      | Mục đích                                             |
| ----------------------------------------- | ---------------------------------------------------- |
| `Dockerfile`                              | Build image backend                                  |
| `deploy/docker/run.sh`                    | Build + run container                                |
| `deploy/nginx/ai-visualizer.conf.example` | Config nginx host (copy sang `/etc/nginx/...`)       |
| `deploy/nginx/nginx.docker.conf`          | Chỉ dùng với `docker-compose.yml` (không dùng ở đây) |
| `.env.example`                            | Biến môi trường mẫu                                  |


---

## Tùy chọn: đổi tên image / container

```bash
IMAGE_NAME=mybackend CONTAINER_NAME=mybackend bash deploy/docker/run.sh
```

Nginx config **không** đổi — vẫn proxy `127.0.0.1:8000`.