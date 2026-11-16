"""
inference_client.py
===================
STEP C of deployment - ONLINE INFERENCE through the forwarded URL.

1. Define the forwarded URL (printed by serve_with_ngrok.py, or your cloud URL).
2. Send a question to <URL>/query.
3. Print the generated response AND the relevant chunks it was based on.

Usage:
    export FORWARDED_URL="https://xxxx.ngrok-free.app"
    python inference_client.py "Is breakfast free at ASH-SEA Pike?"
"""
import os
import sys

import requests

# --- 1. DEFINE THE FORWARDED URL (public address that tunnels to the backend) ----------
FORWARDED_URL = os.getenv("FORWARDED_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")


def online_inference(question: str, top_k: int = 4) -> dict:
    """--- 2. Call the deployed model over HTTP and return its JSON response."""
    headers = {"ngrok-skip-browser-warning": "true"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    resp = requests.post(f"{FORWARDED_URL}/query", json={"question": question, "top_k": top_k},
                         headers=headers, timeout=90)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Is breakfast free at the Miami Brickell property?"
    result = online_inference(q)

    # --- 3. PRINT THE RESPONSE AND THE RELEVANT CHUNKS -----------------------------------
    print("\nQUESTION:", q)
    print("\nRESPONSE (" + result["mode"] + "):\n" + result["answer"])
    print("\nRELEVANT CHUNKS:")
    for i, c in enumerate(result["chunks"], 1):
        print(f"\n[{i}] {c['doc']} | {c['section']} | score={c['score']}")
        print("    " + c["text"].split(" | ", 2)[-1][:400].replace("\n", " "))
