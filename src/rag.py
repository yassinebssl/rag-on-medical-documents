"""Retrieve, gate, generate and cite-check answers for medical questions.

Flow: embed the question -> top-5 chunks -> refuse if the best similarity is
below the threshold -> ask a free LLM on OpenRouter with the passages numbered
[1]..[k] -> keep the answer only if it cites passages and every cited number is
one of the given passages -> replace the numbers with source labels.

Only free OpenRouter models are allowed (model id ends in ':free', or is the
'openrouter/free' router). This is a project rule, enforced here.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFUSAL = "I cannot answer this question from the provided documents."
TOP_K = 5
SIM_THRESHOLD = 0.6  # the brief's value; calibrate on the eval set before reporting
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "baidu/ernie-5.5-flash-preview:free",
    "thinkingmachines/inkling-2-small:free",
]
DEFAULT_MODEL = FREE_MODELS[0]
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2] or [1, 4]

PROMPT = """You are a medical assistant. Answer ONLY using the numbered passages below.
After every claim, cite the passage it comes from by its number in square
brackets, like [1] or [2][4]. Use only the passage numbers given.
If the passages do not contain the answer, say exactly:
"I cannot answer this question from the provided documents."
This is not medical advice. Do not diagnose. For doses, quote the passage and
tell the user to check with a clinician.

Passages:
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
    """Send one chat request to OpenRouter. Retries empty replies, 429 and 5xx with growing pauses."""
    check_free(model)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 600, "temperature": 0}).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if text.strip():
                return text
        except urllib.error.HTTPError as exc:
            if not (exc.code == 429 or exc.code >= 500) or attempt == retries - 1:
                raise
        if attempt < retries - 1:
            time.sleep(10 * 2 ** attempt)  # free tier is rate-limited upstream
    raise RuntimeError(f"empty response from {model}")


def retrieve(question, embedder, collection, k=TOP_K):
    """Return the top-k chunks as dicts with id, text, source, page and similarity."""
    qvec = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[qvec], n_results=k)
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        hits.append({"id": cid, "text": doc, "source": meta["source"],
                     "page": meta["page"] if meta["page"] != -1 else None,
                     "similarity": 1.0 - dist})
    return hits


def source_label(hit):
    """Return 'file, p. N' for a paged source, or the source alone (PubMed title)."""
    return f"{hit['source']}, p. {hit['page']}" if hit["page"] is not None else hit["source"]


def format_context(hits):
    """Number the passages [1]..[k] for the prompt, each under its source label."""
    return "\n\n".join(f"[{i}] {source_label(h)}\n{h['text']}" for i, h in enumerate(hits, 1))


def cited_numbers(answer):
    """Return the sorted, unique passage numbers cited as [n] or [n, m] in the answer."""
    return sorted({int(n) for m in CITATION.findall(answer) for n in m.split(",")})


def expand_citations(answer, hits):
    """Replace each [n, m] marker with one [source label] per number, in order."""
    return CITATION.sub(lambda m: "".join(f"[{source_label(hits[int(n) - 1])}]"
                                          for n in m.group(1).split(",")), answer)


def generate_answer(question, hits, model, api_key):
    """Ask the LLM about the given passages; keep the answer only if its citations are valid."""
    if not hits:
        return {"answer": REFUSAL, "refused": True, "reason": "no_context", "hits": hits, "cited": []}
    raw = call_llm(PROMPT.format(context=format_context(hits), question=question), model, api_key)
    refused = {"answer": REFUSAL, "refused": True, "hits": hits, "cited": [], "raw": raw}
    if REFUSAL.lower() in raw.lower():
        return {**refused, "reason": "model_refused"}
    cited = cited_numbers(raw)
    if not cited or not all(1 <= n <= len(hits) for n in cited):
        return {**refused, "reason": "uncited"}
    return {"answer": expand_citations(raw, hits).strip(), "refused": False, "reason": "answered",
            "hits": hits, "cited": cited, "raw": raw}


def answer_question(question, embedder, collection, model, api_key, threshold=SIM_THRESHOLD):
    """Answer one question: retrieve, refuse below the threshold, else generate_answer."""
    hits = retrieve(question, embedder, collection)
    best = hits[0]["similarity"] if hits else 0.0
    if best < threshold:
        res = {"answer": REFUSAL, "refused": True, "reason": "below_threshold", "hits": hits, "cited": []}
    else:
        res = generate_answer(question, hits, model, api_key)
    return {**res, "best_similarity": best}


if __name__ == "__main__":  # .venv/bin/python src/rag.py "question" [model]
    import sys

    from index import load_embedder, open_index

    env = load_env()
    model = sys.argv[2] if len(sys.argv) > 2 else env.get("OPENROUTER_MODEL") or DEFAULT_MODEL
    res = answer_question(sys.argv[1], load_embedder(), open_index(), model, env["OPENROUTER_API_KEY"])
    print(f"reason={res['reason']} best_similarity={res['best_similarity']:.3f}")
    for h in res["hits"]:
        print(f"  {h['similarity']:.3f}  {h['source'][:80]}  p.{h['page']}")
    print(res.get("raw", res["answer"]))
