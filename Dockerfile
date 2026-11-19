# Hugging Face Spaces (Docker SDK): runs as user 1000 and serves on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE=/home/user/app/.cache/fastembed
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Download the embedding model and build the FAISS index into the image,
# so the container starts fast and needs no network for retrieval.
RUN python ingest.py

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
