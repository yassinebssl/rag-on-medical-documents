"""Retrieve, gate, generate and cite-check answers for medical questions.

Flow: embed the question -> top-5 chunks -> refuse if the best similarity is
below the threshold -> ask a free LLM on OpenRouter with the medical prompt ->
check that the answer cites a retrieved source -> otherwise refuse.

Only free OpenRouter models are allowed (model id ends in ':free', or is the
'openrouter/free' router). This is a project rule, enforced here.
"""
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFUSAL = "I cannot answer this question from the provided documents."
TOP_K = 5
SIM_THRESHOLD = 0.6  # the brief's value; calibrate on the eval set before reporting
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """You are a medical assistant. Answer ONLY using the context below.
For every claim, cite the source as [source: <file name>, p. <page>].
If the context does not contain the answer, say exactly:
"I cannot answer this question from the provided documents."
This is not medical advice. Do not diagnose. For doses, quote the source and
tell the user to check with a clinician.

Context:
{context}

Question: {question}
Answer (with citations):"""


def load_env(path=os.path.join(ROOT, ".env")):
    """Read KEY=VALUE pairs from .env without printing them."""
    env = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.rstrip("\n").split("=", 1)
                    env[k.strip()] = v.strip()
    return env


def check_free(model):
    """Raise unless the model is one of the free OpenRouter models."""
    if not (model.endswith(":free") or model == "openrouter/free"):
        raise ValueError(f"{model!r} is not a free model; the project uses free models only")


def call_llm(prompt, model, api_key, retries=4):
    """Send one chat request to OpenRouter. Retries on 429 with growing pauses."""
    check_free(model)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 600, "temperature": 0}).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(10 * 2 ** attempt)  # free tier is rate-limited upstream
    return ""


def retrieve(question, embedder, collection, k=TOP_K):
    """Return the top-k chunks as dicts with text, source, page and similarity."""
    qvec = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[qvec], n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"],
                     "page": meta["page"] if meta["page"] != -1 else None,
                     "similarity": 1.0 - dist})
    return hits


def cites_retrieved(answer, hits):
    """True if the answer names at least one retrieved source file or title."""
    return any(h["source"] in answer for h in hits)


def answer_question(question, embedder, collection, model, api_key, threshold=SIM_THRESHOLD):
    """Answer one question. Returns a dict with the answer, refusal flag and evidence."""
    hits = retrieve(question, embedder, collection)
    best = hits[0]["similarity"] if hits else 0.0
    if best < threshold:
        return {"answer": REFUSAL, "refused": True, "reason": "below_threshold",
                "best_similarity": best, "hits": hits}
    context = "\n\n".join(f"[source: {h['source']}, p. {h['page']}]\n{h['text']}" for h in hits)
    raw = call_llm(PROMPT.format(context=context, question=question), model, api_key)
    if REFUSAL.lower() in raw.lower() or not cites_retrieved(raw, hits):
        return {"answer": REFUSAL, "refused": True, "reason": "model_refused_or_uncited",
                "best_similarity": best, "hits": hits, "raw": raw}
    return {"answer": raw.strip(), "refused": False, "reason": "answered",
            "best_similarity": best, "hits": hits}


if __name__ == "__main__":  # .venv/bin/python src/rag.py "question" [model]
    import sys

    from index import load_embedder, open_index

    model = sys.argv[2] if len(sys.argv) > 2 else "nvidia/nemotron-3-ultra-550b-a55b:free"
    res = answer_question(sys.argv[1], load_embedder(), open_index(), model,
                          load_env()["OPENROUTER_API_KEY"])
    print(f"reason={res['reason']} best_similarity={res['best_similarity']:.3f}")
    for h in res["hits"]:
        print(f"  {h['similarity']:.3f}  {h['source'][:80]}  p.{h['page']}")
    print(res.get("raw", res["answer"]) if res["reason"] != "below_threshold" else res["answer"])
