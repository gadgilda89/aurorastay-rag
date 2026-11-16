"""
backend/api.py
==============
STEP A of deployment: expose the RAG pipeline as a web API (FastAPI).

Run locally :  uvicorn backend.api:app --host 0.0.0.0 --port 8000
Endpoints   :  GET /health      -> is the service alive?
               POST /query      -> {"question": "...", "top_k": 4} -> answer + chunks
"""
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline, TOP_K

# Load the pipeline ONCE when the server starts (reading PDFs + embedding takes a few
# seconds; doing it per request would be far too slow).
pipeline = RAGPipeline()

app = FastAPI(title="AuroraStay Policy Assistant API", version="1.0.0")

# Optional shared secret: set API_KEY in the environment to require an "x-api-key" header.
# Strongly recommended once the API is exposed on a public forwarded URL.
API_KEY = os.getenv("API_KEY")


class QueryRequest(BaseModel):
    """Shape of the JSON body the client must send."""
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(TOP_K, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    """Used by the frontend, Docker, and monitors to check the service is up."""
    return {"status": "ok", "chunks_indexed": len(pipeline.chunks),
            "embedding_backend": pipeline.embedder.backend}


@app.post("/query")
def query(req: QueryRequest, x_api_key: Optional[str] = Header(default=None)) -> dict:
    """Online inference endpoint: question in, grounded answer + relevant chunks out."""
    if API_KEY and x_api_key != API_KEY:                 # reject callers without the key
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key header")
    return pipeline.answer(req.question, req.top_k)
