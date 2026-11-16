"""Web app over the RAG pipeline: a FastAPI server for the question page in src/web/.

Launch with .venv/bin/python src/app.py, then open http://127.0.0.1:7860.
GET /api/config gives the page its choices; POST /api/ask answers one question of a conversation.
The server keeps no conversation: the page sends the earlier turns with each question.
HOST and PORT in the environment override 127.0.0.1:7860 (the Hugging Face Space sets 0.0.0.0).
"""
import os
import re
import threading
import time
from html import escape
from pathlib import Path

import rag
from rag import load_context

WEB = Path(__file__).resolve().parent / "web"
DISCLAIMER = ("This tool is a student project. It is not medical advice. Check every answer with a clinician. "
              "Questions are sent to OpenRouter: do not type anything that identifies a patient.")
DENSE, HYBRID, RERANKED = "Dense", "Hybrid (dense + BM25)", "Hybrid + cross-encoder gate"
MODES = {DENSE: rag.SIM_THRESHOLD, HYBRID: rag.SIM_THRESHOLD, RERANKED: rag.RERANK_THRESHOLD}  # default threshold
DEFAULT_MODE = RERANKED
NOT_IN_PASSAGES = "The retrieved passages do not contain the answer."
REASONS = {"below_threshold": "Best similarity {best:.2f} is below the threshold.",
           "below_rerank_threshold": "The cross-encoder rates the best passage {rerank:.2f}, below the threshold.",
           "uncited": "The model's answer did not cite the retrieved passages.",
           "model_refused": NOT_IN_PASSAGES,
           "no_context": NOT_IN_PASSAGES}
# reason -> (kind, label) of the verdict; the kind picks its colour on the page
VERDICTS = {"answered": ("answered", "Answered"),
            "below_threshold": ("refused", "Refused by the gate"),
            "below_rerank_threshold": ("refused", "Refused by the gate"),
            "model_refused": ("refused", "Refused by the model"),
            "no_context": ("refused", "Refused"),
            "uncited": ("uncited", "Uncited, refused")}
EXAMPLES = ["How long should parenteral artesunate be given to a patient with severe malaria?",
            "When should antiretroviral therapy be started in a person living with HIV who begins tuberculosis treatment?",
            "What is the first-line topical treatment for scabies?"]
INTRO = ("Answers come from 50 documents: 20 WHO guidelines and 30 PubMed Central articles. "
         "Each claim cites its source, and a question the documents do not cover is refused.")


def verdict(res):
    """Return the kind, the label and the detail of the verdict line for a pipeline result."""
    kind, label = VERDICTS[res["reason"]]
    if res["refused"]:
        return kind, label, REASONS[res["reason"]].format(best=res.get("best_similarity"),
                                                          rerank=res.get("rerank_score"))
    return kind, label, f"{len(res['cited'])} of {len(res['hits'])} passages cited."


def answer_html(answer, hits):
    """Return the answer as HTML: escaped, **bold** kept, paragraphs kept, citations as numbered markers.

    Each [source, p. N] becomes the rank of the first passage with that label, the number of its row
    in the evidence table; adjacent citations share one marker ("1, 3, 4") titled with their labels.
    """
    text = escape(answer)
    ranks = {}
    for rank, hit in enumerate(hits, 1):
        ranks.setdefault(rag.source_label(hit), rank)
    for label, rank in ranks.items():
        text = text.replace(f"[{escape(label)}]", f"\x00{rank}\x01")
    labels = {rank: escape(label) for label, rank in ranks.items()}

    def marker(m):
        cited = [int(n) for n in re.findall(r"\x00(\d+)\x01", m.group(0))]
        links = ", ".join(f'<a data-row="{n}">{n}</a>' for n in cited)
        return f'<span class="cite" title="{"; ".join(labels[n] for n in cited)}">{links}</span>'
    text = re.sub(r"(?:\x00\d+\x01)+", marker, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return "".join(f"<p>{p.strip().replace(chr(10), '<br>')}</p>" for p in text.split("\n\n") if p.strip())


def sources(res):
    """Return the retrieved passages, best first, with their rank, label, similarity and whether they are cited."""
    return [{"rank": rank, "label": rag.source_label(hit), "source": hit["source"], "page": hit["page"],
             "similarity": hit["similarity"], "cited": rank in res["cited"]}
            for rank, hit in enumerate(res["hits"], 1)]


def respond(question, model, mode, threshold, ctx, history=()):
    """Answer one question of the page: return the verdict, the answer, its passages and the step times.

    An empty question, an unknown model or a failed LLM call becomes a verdict with no answer.
    history holds the earlier (question, answer) pairs; only the last rag.HISTORY_TURNS are used.
    """
    if not question.strip():
        return {"kind": "empty", "label": "", "detail": "Type a question first."}
    if model not in rag.FREE_MODELS:
        return {"kind": "error", "label": "Not answered", "detail": f"Unknown model: {model}."}
    mode = mode if mode in MODES else DEFAULT_MODE
    threshold = MODES[mode] if threshold is None else threshold
    bm25 = ctx["bm25"] if mode in (HYBRID, RERANKED) else None
    reranker = ctx["reranker"] if mode == RERANKED else None
    t0 = time.perf_counter()
    try:  # the threshold is the one of whichever gate the mode uses
        res = rag.answer_question(question, ctx["embedder"], ctx["collection"], model, ctx["api_key"],
                                  threshold=threshold, bm25=bm25, reranker=reranker, rerank_threshold=threshold,
                                  history=list(history)[-rag.HISTORY_TURNS:])
    except Exception as exc:  # network, rate limit, used-up quota or empty reply: show it instead of crashing
        part = getattr(exc, "partial", None)
        res = {"kind": "error", "label": "Not answered",
               "detail": f"The model call failed: {exc}. Ask again, or choose another model."}
        if part:  # the search ran: keep its passages and its time
            res.update({"sources": sources({**part, "cited": []}), "best_similarity": part["best_similarity"],
                        "rerank_score": part["rerank_score"], "search_s": part["search_s"], "model_s": None,
                        "follow_up": part["follow_up"], "answer_html": ""})
        return res
    kind, label, detail = verdict(res)
    return {"kind": kind, "label": label, "detail": detail, "refused": res["refused"], "reason": res["reason"],
            "answer": res["answer"], "answer_html": answer_html(res["answer"], res["hits"]),
            "sources": sources(res), "best_similarity": res["best_similarity"],
            "rerank_score": res["rerank_score"], "search_s": res.get("search_s"), "model_s": res.get("model_s"),
            "follow_up": res.get("follow_up", False), "total_s": time.perf_counter() - t0}


def config():
    """Return what the page needs to draw its controls and its empty state."""
    return {"models": rag.FREE_MODELS, "default_model": rag.DEFAULT_MODEL,
            "modes": [{"name": name, "threshold": value} for name, value in MODES.items()],
            "default_mode": DEFAULT_MODE, "examples": EXAMPLES, "disclaimer": DISCLAIMER, "intro": INTRO,
            "history_turns": rag.HISTORY_TURNS}


def create_app(ctx):
    """Return the FastAPI app: the page, its assets, /api/config and /api/ask."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    class Turn(BaseModel):
        question: str
        answer: str

    class Question(BaseModel):
        question: str = Field(max_length=1000)
        model: str = rag.DEFAULT_MODEL
        mode: str = DEFAULT_MODE
        threshold: float | None = Field(default=None, ge=0.0, le=1.0)
        history: list[Turn] = []

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()  # one question at a time: the models share one CPU

    @app.get("/")
    def page():
        return FileResponse(WEB / "index.html")

    @app.get("/api/config")
    def get_config():
        return config()

    @app.post("/api/ask")
    def ask(q: Question):
        with lock:
            return respond(q.question, q.model, q.mode, q.threshold, ctx,
                           [(t.question, t.answer) for t in q.history])

    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")
    import gradio as gr
    import gradio_app  # the brief's Gradio app, same pipeline, at /gradio

    return gr.mount_gradio_app(app, gradio_app.build_app(ctx), path="/gradio", footer_links=[],
                               theme=gradio_app.build_theme(), css=gradio_app.CSS)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(load_context()), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "7860")), log_level="warning")
