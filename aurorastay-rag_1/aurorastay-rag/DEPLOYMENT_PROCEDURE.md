# AuroraStay RAG - Deployment Procedure

Architecture
```
 User ──> Streamlit frontend (Docker, port 8501) ──HTTPS──> Forwarded URL (ngrok) ──> FastAPI backend ──> RAG pipeline
                                                                                           │ retrieve chunks (embeddings)
                                                                                           └ generate answer (LLM)
```

## Phase 0 - Prerequisites
1. Python 3.11, Git, Docker Desktop.
2. Accounts/keys: GitHub (personal access token, scope `repo`), ngrok authtoken, Anthropic API key (optional: without it the bot returns the best passage verbatim).
3. **Keep the GitHub token secret.** Never paste it in code, chat, or Excel. Use the environment only. If it was shared anywhere, revoke it in GitHub > Settings > Developer settings and create a new one.

## Phase 1 - Build and test locally (rubric: dependencies)
```bash
cd aurorastay-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m rag.pipeline "Is breakfast free at the Miami Brickell property?"   # tests steps 1-6
python evaluate.py eval/evaluation.csv                                         # quality check
```

## Phase 2 - Backend + forwarded URL (rubric: Inference)
```bash
export ANTHROPIC_API_KEY=...      API_KEY=a-long-random-string      NGROK_AUTHTOKEN=...
python serve_with_ngrok.py        # prints  FORWARDED URL : https://xxxx.ngrok-free.app
curl https://xxxx.ngrok-free.app/health -H "ngrok-skip-browser-warning: true"
```
Online inference + printing relevant chunks:
```bash
export FORWARDED_URL=https://xxxx.ngrok-free.app
python inference_client.py "I have a service dog at ASH-BOS Harbor. Do I pay a pet fee?"
```
Colab: run the same commands in cells (`!pip install -r requirements.txt`, set the env vars with `os.environ[...]`, then `!python serve_with_ngrok.py`).

## Phase 3 - Frontend (rubric: Streamlit, dependencies, Dockerfile)
```bash
export BACKEND_URL=https://xxxx.ngrok-free.app API_KEY=a-long-random-string
streamlit run frontend/app.py                      # http://localhost:8501
```
Docker:
```bash
docker build -t aurorastay-app .
docker run -p 8501:8501 -e BACKEND_URL=$BACKEND_URL -e API_KEY=$API_KEY aurorastay-app
```
Leave `BACKEND_URL` empty to run the pipeline inside the Streamlit container (single-container mode).

## Phase 4 - Push to GitHub (rubric: push the files)
1. Create an empty repo (e.g. `aurorastay-rag`) on github.com.
2. ```bash
   export GITHUB_USER=<you> GITHUB_REPO=aurorastay-rag
   read -s -p "GitHub token: " GITHUB_TOKEN; export GITHUB_TOKEN; echo
   bash scripts/push_to_github.sh
   ```
3. Confirm the repo shows `Dockerfile`, `requirements.txt`, `frontend/app.py`, `rag/`, `backend/`, `data/` and does **not** show `.env`, `*.xlsx` or any token.

## Phase 5 - Host the frontend publicly (pick one)
- **Streamlit Community Cloud:** share.streamlit.io > New app > choose the repo, main file `frontend/app.py` > Advanced settings > Secrets: `BACKEND_URL = "..."`, `API_KEY = "..."`.
- **Docker host (Render / Cloud Run / Azure Container Apps):** deploy the repo's `Dockerfile`, set the same variables as environment variables, expose port 8501.
- For a permanent backend (no ngrok), deploy `Dockerfile.backend` on a similar host and use its HTTPS URL as `BACKEND_URL`.

## Phase 6 - Verify and monitor
- `/health` returns `status: ok`.
- Ask: a supported question, a cross-property question (Atlanta vs Miami parking), and an out-of-scope one ("Austin North pool") - the last must say the documents don't cover it.
- Re-run `python evaluate.py eval/test.csv` against the deployed `FORWARDED_URL`.
- ngrok free URLs change on restart: update `BACKEND_URL` each time.

## Troubleshooting
| Symptom | Fix |
|---|---|
| 401 from `/query` | `API_KEY` differs between frontend and backend |
| HTML instead of JSON | missing `ngrok-skip-browser-warning` header |
| Answers are raw passages | `ANTHROPIC_API_KEY` not set on the backend |
| Slow first start | embedding model downloading (baked into the Docker image) |
