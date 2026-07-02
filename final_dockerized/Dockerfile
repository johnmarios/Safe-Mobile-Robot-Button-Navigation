# Headless, reproducible renderer for SafetyRacecarButton2-v0.
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglfw3 \
        libglew2.2 \
        libosmesa6 \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: the package does not require a host GPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY controllers ./controllers
COPY app ./app
COPY models ./models
COPY controller.py ./controller.py

RUN mkdir -p /app/outputs

ENTRYPOINT ["python", "-m", "app.render"]
