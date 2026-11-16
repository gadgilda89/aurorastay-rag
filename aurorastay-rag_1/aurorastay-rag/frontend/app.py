"""
frontend/app.py
===============
STEP B of deployment: the Streamlit user interface.

Two modes (chosen automatically):
  * REMOTE mode - if BACKEND_URL is set (e.g. the ngrok "forwarded URL"), every question
    is sent to that API. This is the real client/server deployment.
  * LOCAL mode  - if BACKEND_URL is empty, the RAG pipeline runs inside Streamlit itself
    (handy for a single-container deployment such as Streamlit Community Cloud).

Run:  streamlit run frontend/app.py
"""
import os
import sys

import requests
import streamlit as st

# Make the project root importable when Streamlit is started from a sub-folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="AuroraStay Policy Assistant", page_icon="🏨", layout="wide")

# ---- configuration -------------------------------------------------------
# st.secrets is used on Streamlit Cloud; os.getenv is used locally / in Docker.
def get_setting(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:                       # no secrets.toml present
        return os.getenv(name, default)

BACKEND_URL = get_setting("BACKEND_URL").rstrip("/")
API_KEY = get_setting("API_KEY")

# ---- local pipeline (cached so it is built only once per server process) ---
@st.cache_resource(show_spinner="Building knowledge base ...")
def get_local_pipeline():
    from rag.pipeline import RAGPipeline
    return RAGPipeline()

def ask(question: str, top_k: int) -> dict:
    """Send the question to the backend (remote mode) or answer it locally."""
    if BACKEND_URL:
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        # ngrok free tier shows an interstitial page unless this header is sent.
        headers["ngrok-skip-browser-warning"] = "true"
        r = requests.post(f"{BACKEND_URL}/query", json={"question": question, "top_k": top_k},
                          headers=headers, timeout=90)
        r.raise_for_status()
        return r.json()
    return get_local_pipeline().answer(question, top_k)

# ---- sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)
    st.caption(f"Mode: **{'Remote API' if BACKEND_URL else 'Local pipeline'}**")
    if BACKEND_URL:
        st.caption(f"Backend: {BACKEND_URL}")
    st.markdown("**Try these**")
    examples = [
        "Is breakfast free at the Miami Brickell property?",
        "I have a service dog at ASH-BOS Harbor. Do I pay a pet fee?",
        "How late can I cancel a flexible reservation?",
        "Is there a pool at AuroraStay Austin North?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

# ---- chat history ---------------------------------------------------------
st.title("🏨 AuroraStay Policy Assistant")
st.caption("Answers come only from the AuroraStay policy documents (AS-01 to AS-04).")

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_chunks(chunks):
    """Show the retrieved evidence under an answer (transparency for the user)."""
    with st.expander(f"Relevant chunks ({len(chunks)})"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**{i}. {c['doc']}** - {c['section']}  \n`similarity {c['score']}`")
            st.write(c["text"].split(" | ", 2)[-1])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("chunks"):
            render_chunks(m["chunks"])

# ---- handle a new question --------------------------------------------------
question = st.chat_input("Ask about bookings, refunds, rooms, or guest rules") \
           or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching policies ..."):
                result = ask(question, top_k)
            st.markdown(result["answer"])
            render_chunks(result["chunks"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "chunks": result["chunks"]})
        except Exception as exc:
            st.error(f"Could not get an answer: {exc}")
