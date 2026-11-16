#!/usr/bin/env bash
# Push this project to GitHub using a Personal Access Token WITHOUT writing the token
# into the repo, the remote URL, or your shell history.
#
# Usage:
#   export GITHUB_USER=your-username
#   export GITHUB_REPO=aurorastay-rag           # repo must already exist (can be empty)
#   read -s -p "GitHub token: " GITHUB_TOKEN; export GITHUB_TOKEN; echo
#   bash scripts/push_to_github.sh
set -euo pipefail

: "${GITHUB_USER:?set GITHUB_USER}"; : "${GITHUB_REPO:?set GITHUB_REPO}"; : "${GITHUB_TOKEN:?set GITHUB_TOKEN}"

# 1. Initialise git if this folder is not a repository yet.
[ -d .git ] || git init -b main

# 2. Stage and commit everything not listed in .gitignore.
git add .
git commit -m "AuroraStay RAG: backend, Streamlit frontend, Dockerfile" || echo "Nothing new to commit"

# 3. Point 'origin' at the clean URL (no token inside it).
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${GITHUB_USER}/${GITHUB_REPO}.git"

# 4. Push, passing the token only for this one command as an HTTP header.
AUTH=$(printf "%s:%s" "x-access-token" "$GITHUB_TOKEN" | base64 -w0)
git -c http.extraheader="Authorization: Basic ${AUTH}" push -u origin main
echo "Pushed to https://github.com/${GITHUB_USER}/${GITHUB_REPO}"
