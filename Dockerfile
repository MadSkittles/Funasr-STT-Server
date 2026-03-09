FROM python:3.12-slim

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

ARG http_proxy
ARG https_proxy

RUN pip install --no-cache-dir funasr modelscope torch torchaudio fastapi uvicorn python-multipart httpx

COPY server.py .

EXPOSE 2023

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:2023/health')" || exit 1

CMD ["python", "server.py"]
