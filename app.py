# Flask backend for the AuroraStay policy assistant.
# It loads the saved FAISS index, finds the chunks closest to a question and asks the LLM
# to answer from those chunks only.
#   local : python app.py
#   docker: gunicorn app:app   (see Dockerfile)
import json
import os
from pathlib import Path

import faiss
import requests
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

DB_DIR = Path(os.getenv("DB_DIR", Path(__file__).parent / "vector_db"))
LLM_URL = os.getenv("LLM_URL", "https://api.openai.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TOKEN = os.getenv("LLM_API_KEY", "")      # OpenAI API key
API_KEY = os.getenv("API_KEY", "")                # if set, callers must send it as x-api-key
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))

SYSTEM_PROMPT = (
    "You are the AuroraStay Hotels policy assistant for hotel staff. "
    "Answer only from the numbered context passages. Quote fees, times and deadlines exactly. "
    "Properties differ, so never apply one property's rule or fee to another. "
    "If the reservation confirmation can override the baseline rule, say so. "
    "If the context does not contain the answer (for example a property that is not listed), "
    "start your reply with NOT_COVERED: followed by one sentence saying what is missing. "
    "Otherwise answer in a few sentences and name the document and section you used."
)

# the vector database and embedding model are loaded once, when the server starts
config = json.loads((DB_DIR / "config.json").read_text())
index = faiss.read_index(str(DB_DIR / "index.faiss"))
chunks = json.loads((DB_DIR / "chunks.json").read_text())
embedder = SentenceTransformer(config["embedding_model"])

app = Flask(__name__)


def retrieve(question, k):
    vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(vec, k)
    return [{**chunks[i], "score": round(float(s), 4)} for s, i in zip(scores[0], ids[0]) if i >= 0]


def generate(question, hits):
    context = "\n\n".join(f"[{n}] {h['text']}" for n, h in enumerate(hits, 1))
    resp = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {LLM_TOKEN}", "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "temperature": 0, "max_tokens": 500,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def answer(question, k):
    hits = retrieve(question, k)
    shown = [{"doc": h["doc"], "section": h["section"], "score": h["score"], "text": h["body"]} for h in hits]

    if not hits or hits[0]["score"] < MIN_SCORE:           # nothing close enough in the documents
        return {"answer": "Not covered in the policy documents.", "needs_escalation": True,
                "mode": "no_context", "chunks": shown}
    if not LLM_TOKEN:                                      # no LLM configured: show the best passage
        return {"answer": hits[0]["body"], "needs_escalation": False, "mode": "extractive", "chunks": shown}

    text = generate(question, hits)
    escalate = text.upper().startswith("NOT_COVERED")
    if escalate:
        text = text.split(":", 1)[-1].strip()
    return {"answer": text, "needs_escalation": escalate, "mode": "llm", "chunks": shown}


@app.get("/health")
def health():
    return jsonify(status="ok", chunks=len(chunks), embedding_model=config["embedding_model"])


@app.post("/query")
def query():
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        return jsonify(error="invalid api key"), 401
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if len(question) < 3:
        return jsonify(error="question is required"), 400
    k = max(1, min(int(body.get("top_k", TOP_K)), 10))
    try:
        return jsonify(answer(question, k))
    except Exception as exc:                               # LLM or network problem
        return jsonify(error=str(exc)), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
