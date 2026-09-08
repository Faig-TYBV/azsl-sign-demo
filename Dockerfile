# Recognition backend (src/web_demo/backend.py) — for Render / Fly / any
# container host. This is the full app: MediaPipe + the GRU model + the /ws
# WebSocket, plus the same pages and auth API the Vercel deploy serves.
FROM python:3.12-slim

# MediaPipe needs a graphics stack at runtime even in CPU mode, and pulls
# opencv-contrib-python which needs more of the same. Missing libEGL showed up
# only when a client connected, as:
#   landmarker init failed: libEGL.so.1: cannot open shared object file
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-recognition.txt ./

# Install the CPU-only torch wheel first (the default PyPI build bundles CUDA
# and is several GB — far too big and completely unused here).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.3.0" \
 && pip install --no-cache-dir -r requirements.txt -r requirements-recognition.txt

COPY . .

# Azerbaijani text in the log lines needs UTF-8 stdout.
# MPLCONFIGDIR: matplotlib arrives as a MediaPipe dependency and tries to write
# a config dir under $HOME, which isn't writable on some hosts (HF Spaces).
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port ${PORT}"]
