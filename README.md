# AuroraStay Policy Assistant

RAG assistant over the AuroraStay policy documents. Flask backend, Streamlit frontend, OpenAI LLM.

- `AuroraStay_RAG.ipynb` builds the index, evaluates the system and writes the deployment files
- `backend/` Flask API, Dockerfile, requirements. The saved FAISS index goes in `backend/vector_db/` (created by the notebook)
- `frontend/` Streamlit app, Dockerfile, requirements
- `data/` policy PDFs, `eval/` evaluation and test questions

## Run
1. Run the notebook once. It creates `backend/vector_db/`.
2. Start both containers:
   `LLM_API_KEY=<openai key> API_KEY=<any secret> docker compose up --build`
3. Open http://localhost:8501

Keys are read from environment variables and are never stored in this repository.
