# AuroraStay Policy Assistant

RAG assistant over the AuroraStay policy documents. Flask backend, Streamlit frontend.

- `AuroraStay_RAG.ipynb` builds the index, evaluates the system and writes the deployment files
- `backend/` Flask API, saved FAISS index in `vector_db/` (created by the notebook), Dockerfile
- `frontend/` Streamlit app, Dockerfile
- `data/` policy PDFs, `eval/` evaluation and test questions

Run both containers after running the notebook:
`GH_MODELS_TOKEN=<token> API_KEY=<key> docker compose up --build`, then open http://localhost:8501
