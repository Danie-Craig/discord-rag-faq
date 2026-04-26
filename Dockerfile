# Dockerfile for the Discord RAG FAQ backend API.
#
# Builds an image containing the FastAPI service + RAG pipeline + indices.
# The Discord bot is intentionally NOT included — in a real deployment you
# would run the API on a stateless host (e.g. Cloud Run, Render) and the bot
# on a separate host that maintains the long-lived Discord WebSocket.
#
# Build:   docker build -t discord-rag-faq .
# Run:     docker run --env-file .env -p 8000:8000 discord-rag-faq
# Health:  curl http://localhost:8000/health
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first to maximize layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so first request isn't slow.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy app source and knowledge base.
COPY . .

# Build the dense + BM25 indices at image build time. If the data folder is
# replaced via a volume mount at runtime, re-run `python ingest.py` inside
# the container to refresh the indices.
RUN python ingest.py

EXPOSE 8000

# Healthcheck so orchestrators (Cloud Run, ECS) can detect a wedged container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
