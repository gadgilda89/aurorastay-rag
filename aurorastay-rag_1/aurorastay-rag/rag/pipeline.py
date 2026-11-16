"""
rag/pipeline.py
===============
The complete Retrieval-Augmented Generation (RAG) pipeline for the AuroraStay
policy assistant.

Big picture (the 6 steps this file implements)
----------------------------------------------
STEP 1  Load      : read the 4 AuroraStay policy PDFs into plain text.
STEP 2  Chunk     : split the text into small, section-aware pieces ("chunks").
STEP 3  Embed     : turn every chunk into a numeric vector.
STEP 4  Retrieve  : embed the user question and find the most similar chunks.
STEP 5  Generate  : give the question + retrieved chunks to an LLM and get an answer.
STEP 6  Guardrail : if nothing relevant is found, say so instead of guessing.

Everything is kept in plain Python + numpy so it is easy to read and to deploy.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# CONFIGURATION  (all values can be overridden with environment variables so the
# same code runs locally, in Docker, and in the cloud without edits)
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))                     # folder holding the PDFs
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert")        # "sbert" or "tfidf"
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")            # model used for answers
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))         # max size of one chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))             # overlap between sub-chunks
TOP_K = int(os.getenv("TOP_K", "4"))                               # chunks given to the LLM
# Below this similarity the bot answers "not found" instead of guessing. Neural (sbert)
# and TF-IDF scores live on different scales, so each gets its own default.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25" if EMBEDDING_BACKEND == "sbert" else "0.05"))

# Friendly document names, keyed by the policy code printed in each PDF header.
DOC_TITLES = {
    "AS-01": "Booking Terms and Cancellation Policy",
    "AS-02": "Room and Property Information Guide",
    "AS-03": "Refunds and Additional Charges Guide",
    "AS-04": "Guest Conduct, Accessibility, Safety, and Resolution Guidelines",
}


# ---------------------------------------------------------------------------
# DATA STRUCTURE: one retrievable piece of knowledge
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    doc: str        # e.g. "Refunds and Additional Charges Guide"
    code: str       # e.g. "AS-03"
    section: str    # e.g. "3. Additional charges"
    text: str       # the chunk text itself
    score: float = 0.0  # similarity to the question (filled in at retrieval time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(float(self.score), 4)
        return d


# ===========================================================================
# STEP 1 - LOAD: PDF  ->  text
# ===========================================================================
def load_pdf_pages(pdf_path: Path) -> List[str]:
    """Extract the text of every page of one PDF."""
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def clean_text(text: str) -> str:
    """Remove the repeating page header ("AuroraStay Hotels | AS-01 | ... Page N")
    and collapse messy whitespace, so headers never pollute the chunks."""
    text = re.sub(r"AuroraStay Hotels \| AS-\d+ \| Internal operating policy Page \d+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ===========================================================================
# STEP 2 - CHUNK: text -> section-aware chunks
# ===========================================================================
# Section headings in these PDFs look like "1. Scope and governing rules".
SECTION_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][^\n]{3,90})$", re.MULTILINE)


def split_into_sections(full_text: str) -> List[tuple[str, str]]:
    """Split a document into (section_title, section_body) pairs using the numbered headings."""
    # Keep only headings that count 1, 2, 3 ... in order. This ignores numbered rows inside
    # tables (e.g. the "1. Locate reservation" workflow steps) that look like headings.
    matches, expected = [], 1
    for m in SECTION_RE.finditer(full_text):
        if int(m.group(1)) == expected:
            matches.append(m)
            expected += 1
    if not matches:                                   # fallback: treat whole doc as one section
        return [("Document", full_text)]
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        title = f"{m.group(1)}. {m.group(2).strip()}"
        sections.append((title, full_text[start:end].strip()))
    return sections


def split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """If a section is longer than max_chars, cut it on line boundaries with a small
    overlap so no sentence loses its context."""
    if len(text) <= max_chars:
        return [text]
    lines, pieces, current = text.split("\n"), [], ""
    for line in lines:
        if len(current) + len(line) + 1 > max_chars and current:
            pieces.append(current.strip())
            current = current[-overlap:] + "\n"        # carry the tail forward as overlap
        current += line + "\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def build_chunks(data_dir: Path = DATA_DIR) -> List[Chunk]:
    """Run STEP 1 + STEP 2 for every PDF in data_dir and return all chunks."""
    chunks: List[Chunk] = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        pages = load_pdf_pages(pdf)
        full = clean_text("\n".join(pages))
        code_match = re.search(r"\bAS-0\d\b", "\n".join(pages))
        code = code_match.group(0) if code_match else pdf.stem
        doc_name = DOC_TITLES.get(code, pdf.stem)

        for title, body in split_into_sections(full):
            if len(body) < 40:                         # skip empty / heading-only sections
                continue
            for piece in split_long_text(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP):
                # Prefix each chunk with doc + section so the meaning travels with the text
                # (this makes retrieval much better for short chunks).
                chunks.append(Chunk(doc=doc_name, code=code, section=title,
                                    text=f"{doc_name} | {title} | {piece}"))
    return chunks


# ===========================================================================
# STEP 3 - EMBED: text -> vectors
# ===========================================================================
class Embedder:
    """Wraps two interchangeable embedding methods behind one interface.

    * "sbert": sentence-transformers neural embeddings (best quality, default in production)
    * "tfidf": scikit-learn TF-IDF (no model download; used for offline tests / fallback)
    """

    def __init__(self, backend: str = EMBEDDING_BACKEND):
        self.backend = backend
        if backend == "sbert":
            from sentence_transformers import SentenceTransformer  # imported lazily (heavy)
            self.model = SentenceTransformer(SBERT_MODEL)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Embed the corpus (chunks). Returns an L2-normalised matrix (n_chunks x dim)."""
        if self.backend == "sbert":
            vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            vecs = self.model.fit_transform(texts).toarray()
        return self._normalise(np.asarray(vecs, dtype=np.float32))

    def transform(self, texts: List[str]) -> np.ndarray:
        """Embed new text (a user question) with the SAME model as the corpus."""
        if self.backend == "sbert":
            vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            vecs = self.model.transform(texts).toarray()
        return self._normalise(np.asarray(vecs, dtype=np.float32))

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        """Unit-length vectors make a dot product equal to cosine similarity."""
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms


# ===========================================================================
# STEP 4 - RETRIEVE, STEP 5 - GENERATE, STEP 6 - GUARDRAIL
# ===========================================================================
SYSTEM_PROMPT = """You are the AuroraStay Hotels policy assistant.
Answer ONLY from the CONTEXT passages provided. Rules:
1. If the context does not contain the answer (for example a property that is not listed),
   say clearly that the policy documents do not cover it and suggest contacting the property.
2. Never invent fees, times, or policies. Quote numbers exactly as written.
3. Properties differ: never apply one property's fee or rule to another.
4. If the reservation confirmation may override the baseline policy, mention it.
5. Be concise and friendly. Mention the policy document and section you relied on."""


class RAGPipeline:
    """Holds the chunks + vectors in memory and exposes retrieve() and answer()."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.chunks = build_chunks(data_dir)                                  # STEP 1 + 2
        if not self.chunks:
            raise RuntimeError(f"No PDF content found in {data_dir.resolve()}")
        self.embedder = Embedder()                                            # STEP 3
        self.matrix = self.embedder.fit_transform([c.text for c in self.chunks])

    # ---- STEP 4 ----------------------------------------------------------
    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Chunk]:
        """Return the top_k chunks most similar to the question."""
        q_vec = self.embedder.transform([question])[0]
        scores = self.matrix @ q_vec                     # cosine similarity for every chunk
        best = np.argsort(-scores)[:top_k]               # indices of the highest scores
        results = []
        for i in best:
            c = Chunk(**{**self.chunks[i].__dict__, "score": float(scores[i])})
            results.append(c)
        return results

    # ---- STEP 5 + 6 ------------------------------------------------------
    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """Full RAG: retrieve -> guardrail -> generate. Returns answer + the chunks used."""
        chunks = self.retrieve(question, top_k)

        # STEP 6 guardrail: if even the best chunk is weakly related, refuse to guess.
        if not chunks or chunks[0].score < MIN_SCORE:
            return {"answer": ("I could not find this in the AuroraStay policy documents. "
                               "Please contact the property or Guest Experience team."),
                    "chunks": [c.to_dict() for c in chunks], "mode": "no_context"}

        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if api_key:                                       # STEP 5: LLM generation
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=LLM_MODEL, max_tokens=500, temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user",
                               "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                return {"answer": text, "chunks": [c.to_dict() for c in chunks], "mode": "llm"}
            except Exception as exc:                      # network/key problem -> fall through
                print(f"[warn] LLM call failed, using extractive fallback: {exc}")

        # Fallback (no API key / LLM error): return the best passages verbatim.
        best = chunks[0]
        return {"answer": f"(Extractive answer - no LLM configured)\n\n{best.text}",
                "chunks": [c.to_dict() for c in chunks], "mode": "extractive"}


# Quick manual test:  python -m rag.pipeline "Is breakfast free in Miami?"
if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "Is breakfast free at the Miami Brickell property?"
    out = RAGPipeline().answer(q)
    print(json.dumps(out, indent=2)[:3000])
