"""Retrieve, gate, generate and cite-check answers for medical questions.

Flow: embed the question -> top-5 chunks -> refuse if the best similarity is
below the threshold -> ask a free LLM on OpenRouter with the passages numbered
[1]..[k] -> keep the answer only if it cites passages and every cited number is
one of the given passages -> replace the numbers with source labels.

Hybrid mode (bm25 given): the top-20 dense chunks and the top-20 BM25 chunks
are fused by reciprocal rank fusion and the first 5 are the passages; the
refusal gate still uses the best dense similarity.

Cross-encoder gate (reranker given): the passages stay the same, but the gate
compares the cross-encoder's score over them (rerank.gate_score, 0..1) with
RERANK_THRESHOLD instead of comparing the dense similarity with the threshold.
It refuses when no passage reaches the threshold.

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
CANDIDATES = 20  # chunks taken from each ranking before fusion in hybrid mode
SIM_THRESHOLD = 0.6  # the brief's value; calibrate on the eval set before reporting
RERANK_THRESHOLD = 0.5  # cross-encoder gate: the model's own boundary (logit 0), not tuned on the eval set
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LENGTH_RETRY_TOKENS = 4000  # budget of the one resend after a reasoning model used max_tokens up
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


class QuotaExhausted(RuntimeError):
    """The account's daily free-model requests are used up; a retry before 00:00 UTC fails too."""


def check_free(model):
    """Raise unless the model is one of the free OpenRouter models."""
    if not (model.endswith(":free") or model == "openrouter/free"):
        raise ValueError(f"{model!r} is not a free model; the project uses free models only")


def call_llm(prompt, model, api_key, retries=4, max_tokens=600):
    """Send one chat request to OpenRouter. Retries empty replies, 429 and 5xx with growing pauses.

    A 429 whose body names the free-models-per-day limit raises QuotaExhausted at once,
    with no retry and no pause.

    If every reply is empty, raise RuntimeError; it carries the last reply's `error`
    field when the JSON body has one (a 200 reply can hold a provider error).
    An empty reply cut at max_tokens means a reasoning model spent the whole budget on
    hidden reasoning, and the same request at temperature 0 would do the same: it is
    sent once more at once with LENGTH_RETRY_TOKENS, and raises if that is cut too (or
    if max_tokens was already that large). A reply that fits in max_tokens is unchanged.
    """
    check_free(model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    error = None
    for attempt in range(retries):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            choice = (data.get("choices") or [{}])[0]
            text = choice.get("message", {}).get("content") or ""
            if text.strip():
                return text
            if choice.get("finish_reason") == "length":
                if max_tokens >= LENGTH_RETRY_TOKENS:
                    raise RuntimeError(f"{model} used all max_tokens={max_tokens} before writing an answer")
                max_tokens = LENGTH_RETRY_TOKENS
                continue
            error = data.get("error")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:  # no or unreadable body: an ordinary 429
                    detail = ""
                if "free-models-per-day" in detail:
                    raise QuotaExhausted("the daily free-model limit of the OpenRouter account is used up; "
                                         "it resets at 00:00 UTC") from exc
            if not (exc.code == 429 or exc.code >= 500) or attempt == retries - 1:
                raise
        if attempt < retries - 1:
            time.sleep(10 * 2 ** attempt)  # free tier is rate-limited upstream
    raise RuntimeError(f"empty response from {model}" + (f": {error}" if error else ""))


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
    """Number the passages [1]..[k] for the prompt, each under its source label.

    A passage's own reference markers ([12], [3, 4]) become (12), (3, 4) in the prompt,
    so that the model cannot copy them as citations. The hits are not changed.
    """
    return "\n\n".join(f"[{i}] {source_label(h)}\n{CITATION.sub(lambda m: f'({m.group(1)})', h['text'])}"
                         for i, h in enumerate(hits, 1))


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


def rrf(rankings, k=60):
    """Fuse ranked hit lists by reciprocal rank: an id scores the sum of 1 / (k + rank).

    Ties keep the order of first appearance; the dict kept for an id is the one
    from the earliest ranking that has it.
    """
    # Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and
    # individual rank learning methods", SIGIR 2009: https://doi.org/10.1145/1571941.1572114
    # The structure follows reciprocal_rank_fusion in the rag-from-scratch tutorial (Part 6,
    # RAG-Fusion): https://github.com/langchain-ai/rag-from-scratch. Unlike it, ranks count
    # from 1 as in the paper, and hits are keyed by their chunk id.
    scores, first = {}, {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (k + rank)
            first.setdefault(hit["id"], hit)
    return [first[i] for i in sorted(scores, key=lambda i: -scores[i])]


def answer_question(question, embedder, collection, model, api_key, threshold=SIM_THRESHOLD,
                    bm25=None, reranker=None, rerank_threshold=RERANK_THRESHOLD):
    """Retrieve (fused with BM25 if given), refuse below the gate's threshold, else generate_answer.

    The gate is the best dense similarity against threshold, or, with a reranker,
    the cross-encoder gate score against rerank_threshold. The cross-encoder stops at
    the first passage that reaches rerank_threshold, so rerank_score is that passage's
    score when the question is answered and the best score when it is refused.
    """
    if bm25 is None:
        hits = retrieve(question, embedder, collection)
        best = hits[0]["similarity"] if hits else 0.0
    else:
        dense = retrieve(question, embedder, collection, k=CANDIDATES)
        best = dense[0]["similarity"] if dense else 0.0
        hits = rrf([dense, bm25.search(question, CANDIDATES)])[:TOP_K]
    rerank_score = None
    if reranker is not None:
        from rerank import gate_score

        rerank_score = gate_score(question, hits, reranker, stop_at=rerank_threshold)
        refuse, reason = rerank_score < rerank_threshold, "below_rerank_threshold"
    else:
        refuse, reason = best < threshold, "below_threshold"
    if refuse:
        res = {"answer": REFUSAL, "refused": True, "reason": reason, "hits": hits, "cited": []}
    else:
        res = generate_answer(question, hits, model, api_key)
    return {**res, "best_similarity": best, "rerank_score": rerank_score}


def load_context():
    """Return the embedder, collection, BM25 index, gate reranker and API key; exit with one line if one is missing."""
    import index  # imported here so that importing rag stays light
    from bm25 import BM25
    from rerank import GATE_RERANKER, load_reranker
    from chromadb.errors import NotFoundError

    api_key = load_env().get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing: copy .env.example to .env and add your key.")
    empty = "The index is empty: run .venv/bin/python src/index.py first."
    try:
        collection = index.open_index()
    except NotFoundError:  # the collection does not exist; other errors propagate unchanged
        raise SystemExit(empty)
    if collection.count() == 0:
        raise SystemExit(empty)
    return {"embedder": index.load_embedder(), "collection": collection,
            "bm25": BM25(index.all_chunks(collection)), "reranker": load_reranker(GATE_RERANKER),
            "api_key": api_key}


if __name__ == "__main__":  # .venv/bin/python src/rag.py "question" [model] [--hybrid] [--rerank]
    import sys

    args = [a for a in sys.argv[1:] if a not in ("--hybrid", "--rerank")]
    ctx = load_context()
    model = args[1] if len(args) > 1 else load_env().get("OPENROUTER_MODEL") or DEFAULT_MODEL
    bm25 = ctx["bm25"] if "--hybrid" in sys.argv or "--rerank" in sys.argv else None
    reranker = ctx["reranker"] if "--rerank" in sys.argv else None
    res = answer_question(args[0], ctx["embedder"], ctx["collection"], model, ctx["api_key"],
                          bm25=bm25, reranker=reranker)
    gate = "" if res["rerank_score"] is None else f" rerank_score={res['rerank_score']:.3f}"
    print(f"reason={res['reason']} best_similarity={res['best_similarity']:.3f}{gate}")
    for h in res["hits"]:
        sim = "-" if h["similarity"] is None else f"{h['similarity']:.3f}"
        print(f"  {sim:>5}  {source_label(h)}")
    print(res.get("raw", res["answer"]))
