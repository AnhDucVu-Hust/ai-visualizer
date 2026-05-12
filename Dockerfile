FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libass9 \
    libfreetype6 \
    libfontconfig1 \
    libharfbuzz0b \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend only (frontend is ignored)
COPY backend /app/backend
COPY video_combine /app/video_combine
COPY run_app.py /app/run_app.py

# Config files
COPY config.yaml /app/config.yaml
COPY video_combine_config.yaml /app/video_combine_config.yaml

EXPOSE 8000

CMD ["python", "run_app.py", "--host", "0.0.0.0", "--port", "8000"]



