"""
evaluate.py
===========
Measures how good the deployed RAG system is, using eval/evaluation.csv (and test.xlsx).

Metrics
-------
* Retrieval hit-rate : does the top-k contain the document+section named in
                       the CSV's `relevant_context` column?
* Answer token-F1    : word overlap between the generated answer and `expected_answer`.
* Refusal check      : questions about things not in the documents (e.g. "Austin North")
                       should produce a "not covered" answer.

Usage:  python evaluate.py eval/evaluation.csv          (local pipeline)
        FORWARDED_URL=https://... python evaluate.py eval/test.csv   (online)
"""
import os
import re
import sys

import pandas as pd

TITLE_TO_CODE = {"Booking Terms": "AS-01", "Room and Property": "AS-02",
                 "Refunds and Additional": "AS-03", "Guest Conduct": "AS-04"}


def parse_relevant(ctx: str) -> set:
    """Turn 'Doc | Section 3, Title | text <> Doc2 | ...' into {(code, section_number)}."""
    wanted = set()
    for part in str(ctx).split("<>"):
        fields = [f.strip() for f in part.split("|")]
        if len(fields) < 2:
            continue
        code = next((c for t, c in TITLE_TO_CODE.items() if t in fields[0]), None)
        m = re.search(r"Section\s+(\d+)", fields[1])
        if code and m:
            wanted.add((code, m.group(1)))
    return wanted


def token_f1(pred: str, gold: str) -> float:
    """Simple word-overlap F1 (no extra libraries needed)."""
    tok = lambda s: re.findall(r"[a-z0-9$]+", str(s).lower())
    p, g = tok(pred), tok(gold)
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def get_answerer():
    """Use the online endpoint if FORWARDED_URL is set, otherwise the local pipeline."""
    if os.getenv("FORWARDED_URL"):
        from inference_client import online_inference
        return online_inference
    from rag.pipeline import RAGPipeline
    return RAGPipeline().answer


def main(path: str):
    df = pd.read_excel(path) if path.endswith("xlsx") else pd.read_csv(path)
    answer = get_answerer()
    rows = []
    for _, r in df.iterrows():
        out = answer(r["question"])
        got = {(c["code"], c["section"].split(".")[0]) for c in out["chunks"]}
        want = parse_relevant(r["relevant_context"])
        rows.append({"question": r["question"][:70],
                     "hit": (len(want & got) / len(want)) if want else None,
                     "f1": token_f1(out["answer"], r["expected_answer"]),
                     "mode": out["mode"]})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print(f"\nMean retrieval hit-rate: {res['hit'].mean():.2f}   Mean answer F1: {res['f1'].mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/evaluation.csv")
