FROM python:3.11-slim

ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MODEL_PATH=/app/weights/best.pt \
    PORT=8080

WORKDIR /app

# libglib is required by OpenCV even when the headless wheel is used.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip uninstall -y opencv-python \
    && python -m pip install --no-cache-dir --force-reinstall \
        opencv-python-headless==4.12.0.88

COPY predict.py extractor.py builder.py media.py snapshots.py storage.py config.py entrypoint.py ./
COPY best.pt weights/best.pt
COPY scripts/smoke_test.py scripts/contract_smoke_test.py scripts/

RUN useradd --create-home --uid 10001 model-server \
    && mkdir -p /tmp/ultralytics \
    && chown -R model-server:model-server /tmp/ultralytics /app

USER model-server

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=3)"

# SageMaker appends `serve`; the entrypoint also permits command overrides for
# smoke tests, e.g. `docker run IMAGE python -c "..."`.
ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["serve"]
