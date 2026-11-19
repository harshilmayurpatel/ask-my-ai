# Runs on Render (free web service) or any Docker host.
# Render tells the app which port to use through $PORT; 7860 is the local default.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE=/home/user/app/.cache/fastembed \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Download the embedding model and build the FAISS index into the image,
# so the container starts fast and needs no network for retrieval.
RUN python ingest.py

EXPOSE 7860
# Shell form so $PORT is read at start-up; one worker keeps memory under 512 MB
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
