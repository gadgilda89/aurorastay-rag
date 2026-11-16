"""
serve_with_ngrok.py
===================
Starts the FastAPI backend and opens a public "forwarded URL" with ngrok.
Ideal for Google Colab / a laptop / any machine without a public IP.

Setup:  pip install -r requirements.txt pyngrok
        export NGROK_AUTHTOKEN=<token from https://dashboard.ngrok.com>
Run  :  python serve_with_ngrok.py
In Colab use:  os.environ["NGROK_AUTHTOKEN"] = "..."   then   !python serve_with_ngrok.py
"""
import os
import threading
import time

import uvicorn
from pyngrok import ngrok

PORT = int(os.getenv("PORT", "8000"))

# 1. Start the API server in a background thread so this script can keep going.
from backend.api import app  # importing builds the RAG index (STEP 1-3)
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning"),
                 daemon=True).start()
time.sleep(3)                                            # give the server a moment to bind

# 2. Open the tunnel. ngrok reads NGROK_AUTHTOKEN from the environment.
ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
tunnel = ngrok.connect(PORT, "http")

# 3. This is the FORWARDED URL. Put it in the Streamlit BACKEND_URL setting.
print("\n==============================================")
print("FORWARDED URL :", tunnel.public_url)
print("Health check  :", tunnel.public_url + "/health")
print("==============================================\n")

while True:                                              # keep the process alive
    time.sleep(60)
