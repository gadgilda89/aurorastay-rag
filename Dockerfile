# ---------------------------------------------------------------------------
# Dockerfile for the Streamlit frontend (+ optional local RAG pipeline)
# Build :  docker build -t aurorastay-app .
# Run   :  docker run -p 8501:8501 -e BACKEND_URL=https://xxxx.ngrok-free.app aurorastay-app
# ---------------------------------------------------------------------------

# 1. Start from a small official Python image.
FROM python:3.11-slim

# 2. Don't write .pyc files, and print logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# 3. Work inside /app in the container.
WORKDIR /app

# 4. Install dependencies FIRST. Docker caches this layer, so rebuilds are fast
#    unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Pre-download the embedding model into the image so the container starts quickly
#    and works without internet at run time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 6. Copy the application code and the policy PDFs.
COPY rag ./rag
COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY .streamlit ./.streamlit

# 7. Streamlit listens on 8501.
EXPOSE 8501

# 8. Container health check (Streamlit exposes /_stcore/health).
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# 9. Start the app. Settings come from environment variables (BACKEND_URL, API_KEY, ...).
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
