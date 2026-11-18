# Frontend image. Build from the frontend folder:
#   docker build -t aurorastay-frontend .
#   docker run -p 8501:8501 -e BACKEND_URL=<forwarded url> -e API_KEY=<key> aurorastay-frontend
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
