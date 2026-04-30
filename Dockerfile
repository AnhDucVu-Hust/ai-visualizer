FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libass9 \
    libfreetype6 \
    libfontconfig1 \
    libharfbuzz0b \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# binary
COPY dist/run_app /app/run_app

# config (để ngoài, không nhét vào binary)
COPY config.yaml /app/config.yaml
COPY video_combine_config.yaml /app/video_combine_config.yaml

RUN chmod +x /app/run_app

EXPOSE 10000

CMD ["./run_app"]