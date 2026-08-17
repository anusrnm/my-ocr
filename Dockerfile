FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OCR_MODEL_DIR=/models/easyocr

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
       libglib2.0-0 libgl1 tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py ocr_provider.py ./

RUN useradd --create-home --uid 10001 ocr \
    && mkdir -p /models/easyocr /work \
    && chown -R ocr:ocr /app /models /work

USER ocr
VOLUME ["/models", "/work"]

ENTRYPOINT ["python", "app.py"]