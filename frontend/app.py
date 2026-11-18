# Streamlit frontend for AuroraStay staff. It sends each question to the Flask backend
# (BACKEND_URL) and shows the answer together with the source chunks.
#   streamlit run app.py
import os

import requests
import streamlit as st

st.set_page_config(page_title="AuroraStay Policy Assistant", layout="wide")


def setting(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:                      # no secrets file
        return os.getenv(name, default)


BACKEND_URL = setting("BACKEND_URL", "http://localhost:5000").rstrip("/")
API_KEY = setting("API_KEY")


def ask(question, top_k):
    headers = {"ngrok-skip-browser-warning": "true"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    r = requests.post(f"{BACKEND_URL}/query", json={"question": question, "top_k": top_k},
                      headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()


def show_sources(chunks):
    with st.expander(f"Supporting sources ({len(chunks)})"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**{i}. {c['doc']}**, {c['section']}  (similarity {c['score']})")
            st.write(c["text"])


# running count of answered vs escalated questions in this session
stats = st.session_state.setdefault("stats", {"answered": 0, "escalated": 0})
history = st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", 1, 8, 5)
    st.caption(f"Backend: {BACKEND_URL}")
    counters = st.container()          # filled at the end of the script so it shows the latest counts
    st.subheader("Examples")
    for example in ["Is breakfast free at the Miami Brickell property?",
                    "I have a service dog at ASH-BOS Harbor. Do I pay a pet fee?",
                    "How late can I cancel a flexible reservation?",
                    "Is there a pool at AuroraStay Austin North?"]:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example

st.title("AuroraStay Policy Assistant")
st.caption("Internal proof of concept. Answers come only from the AuroraStay policy documents.")

for m in history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("escalate"):
            st.warning("Not covered by the documents. Check with the property team or Guest Experience.")
        if m.get("chunks"):
            show_sources(m["chunks"])

question = st.chat_input("Ask about bookings, refunds, rooms or guest rules") or st.session_state.pop("pending", None)
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the policy documents..."):
                result = ask(question, top_k)
            escalate = result.get("needs_escalation", False)
            stats["escalated" if escalate else "answered"] += 1
            st.markdown(result["answer"])
            if escalate:
                st.warning("Not covered by the documents. Check with the property team or Guest Experience.")
            show_sources(result["chunks"])
            history.append({"role": "assistant", "content": result["answer"],
                            "escalate": escalate, "chunks": result["chunks"]})
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with counters:
    st.metric("Answered from documents", stats["answered"])
    st.metric("Needs manual lookup", stats["escalated"])
