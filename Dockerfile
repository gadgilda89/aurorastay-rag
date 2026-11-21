# Backend image. Build from the backend folder:
#   docker build -t aurorastay-backend .
#   docker run -p 5000:5000 -e LLM_API_KEY=<key> -e API_KEY=<key> aurorastay-backend
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only torch keeps the image much smaller than the default GPU build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download the embedding model at build time so the container starts quickly
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app.py .
COPY vector_db ./vector_db

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "180", "app:app"]
