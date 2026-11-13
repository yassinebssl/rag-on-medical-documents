"""Compare free OpenRouter models on the same retrieval and the same questions.

Retrieval (BGE-M3 + Chroma) is shared, so only the generator changes. Each
model answers every question in data/eval/questions.json. The script writes
results/comparison.json (per answer) and results/summary.csv (per model).

Metrics per model:
- answer_rate: answered / answerable questions (should be high)
- refusal_rate: refused / out-of-corpus questions (should be high)
- cited_rate: answered questions whose answer names a retrieved source
- mean_latency_s: wall time per question, including retries

Usage:
    .venv/bin/python src/compare.py
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from index import load_embedder, open_index  # noqa: E402
from rag import ROOT, answer_question, load_env  # noqa: E402

FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "baidu/ernie-5.5-flash-preview:free",
    "thinkingmachines/inkling-2-small:free",
]
QUESTIONS = os.path.join(ROOT, "data", "eval", "questions.json")
OUT_DIR = os.path.join(ROOT, "results")


def main():
    env = load_env()
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing from .env")
    with open(QUESTIONS, encoding="utf-8") as fh:
        questions = json.load(fh)
    embedder = load_embedder()
    collection = open_index()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows, records = [], []
    for model in FREE_MODELS:
        answered = refused_oos = cited = n_ans = n_oos = 0
        latencies = []
        for q in questions:
            t0 = time.time()
            try:
                res = answer_question(q["question"], embedder, collection, model, api_key)
            except Exception as exc:  # keep going if one free model is down
                res = {"answer": f"ERROR: {exc}", "refused": None, "reason": "error",
                       "best_similarity": None, "hits": []}
            latency = time.time() - t0
            latencies.append(latency)
            records.append({"model": model, "id": q["id"], "answerable": q["answerable"],
                            "question": q["question"], "answer": res["answer"],
                            "refused": res["refused"], "reason": res["reason"],
                            "best_similarity": res["best_similarity"], "latency_s": round(latency, 2),
                            "sources": [h["source"] for h in res["hits"]]})
            if q["answerable"]:
                n_ans += 1
                answered += res["refused"] is False
                cited += res["reason"] == "answered"
            else:
                n_oos += 1
                refused_oos += res["refused"] is True
            print(f"{model} {q['id']}: {res['reason']} ({latency:.1f}s)", flush=True)
        rows.append({"model": model,
                     "answer_rate": round(answered / n_ans, 3) if n_ans else None,
                     "refusal_rate": round(refused_oos / n_oos, 3) if n_oos else None,
                     "cited_rate": round(cited / n_ans, 3) if n_ans else None,
                     "mean_latency_s": round(sum(latencies) / len(latencies), 2)})

    with open(os.path.join(OUT_DIR, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
