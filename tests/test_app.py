"""Tests for the web app: verdicts, answer HTML, respond, the HTTP routes and load_context. Fakes only."""
import pytest
from chromadb.errors import NotFoundError
from fastapi.testclient import TestClient

import app
import index
import rag
from conftest import FakeCollection

HITS = [{"id": "a", "text": "Artemether text", "source": "WHO_09.pdf", "page": 42, "similarity": 0.81234},
        {"id": "b", "text": "Other text", "source": "A <b> title", "page": None, "similarity": None}]
CTX = {"embedder": None, "collection": None, "bm25": "BM25", "reranker": "CE", "api_key": "k"}
ANSWER = {"answer": "Twice **daily** [WHO_09.pdf, p. 42].", "refused": False, "reason": "answered", "hits": HITS,
          "cited": [1], "best_similarity": 0.81, "rerank_score": 0.99, "search_s": 0.3, "model_s": 9.5}


def fake_pipeline(monkeypatch, res=ANSWER, calls=None):
    """Replace rag.answer_question with one that records its keyword arguments and returns res."""
    def fake(question, *args, **kwargs):
        if calls is not None:
            calls.append((question, kwargs))
        return res
    monkeypatch.setattr(rag, "answer_question", fake)

def test_verdict_for_a_refusal_says_why():
    kind, label, detail = app.verdict({"answer": rag.REFUSAL, "refused": True, "reason": "below_threshold",
                                       "hits": HITS, "cited": [], "best_similarity": 0.41})
    assert kind == "refused" and "gate" in label and "0.41" in detail and "threshold" in detail

def test_verdict_for_an_answer_counts_the_cited_passages():
    assert app.verdict(ANSWER) == ("answered", "Answered", "1 of 2 passages cited.")

def test_verdict_for_a_cross_encoder_refusal_gives_its_score():
    _, _, detail = app.verdict({"answer": rag.REFUSAL, "refused": True, "reason": "below_rerank_threshold",
                                "hits": HITS, "cited": [], "best_similarity": 0.7, "rerank_score": 0.123})
    assert "0.12" in detail and "cross-encoder" in detail.lower()

def test_every_refusal_reason_has_a_verdict():
    assert set(app.VERDICTS) == set(app.REASONS) | {"answered"}

def test_answer_html_escapes_keeps_bold_and_wraps_each_citation():
    html = app.answer_html("A <b> **dose** [WHO_09.pdf, p. 42][A <b> title].\n\nSecond [A <b> title].", HITS)
    assert html.startswith("<p>A &lt;b&gt; <strong>dose</strong> ")
    assert ('<span class="cite" title="WHO_09.pdf, p. 42; A &lt;b&gt; title">'
            '<a data-row="1">1</a>, <a data-row="2">2</a></span>') in html
    assert html.endswith('<p>Second <span class="cite" title="A &lt;b&gt; title"><a data-row="2">2</a></span>.</p>')

def test_sources_marks_the_cited_passages_by_rank():
    rows = app.sources(ANSWER)
    assert [(r["rank"], r["label"], r["cited"]) for r in rows] == [(1, "WHO_09.pdf, p. 42", True),
                                                                 (2, "A <b> title", False)]

def test_respond_with_an_empty_question_does_not_call_the_pipeline(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda *a, **k: 1 / 0)
    assert app.respond("   ", rag.DEFAULT_MODEL, app.DENSE, None, CTX) == {
        "kind": "empty", "label": "", "detail": "Type a question first."}

def test_respond_refuses_an_unknown_model_without_calling_the_pipeline(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda *a, **k: 1 / 0)
    res = app.respond("q?", "openai/gpt-paid", app.DENSE, None, CTX)
    assert res["kind"] == "error" and "openai/gpt-paid" in res["detail"]

def test_respond_passes_bm25_in_both_hybrid_modes_and_the_reranker_only_with_the_gate(monkeypatch):
    calls = []
    fake_pipeline(monkeypatch, calls=calls)
    for mode in (app.DENSE, app.HYBRID, app.RERANKED):
        app.respond("q?", rag.DEFAULT_MODEL, mode, 0.55, CTX)
    assert [(k["bm25"], k["reranker"], k["threshold"], k["rerank_threshold"]) for _, k in calls] == [
        (None, None, 0.55, 0.55), ("BM25", None, 0.55, 0.55), ("BM25", "CE", 0.55, 0.55)]

def test_respond_uses_the_mode_threshold_when_none_is_given_and_the_gate_for_an_unknown_mode(monkeypatch):
    calls = []
    fake_pipeline(monkeypatch, calls=calls)
    app.respond("q?", rag.DEFAULT_MODEL, "nonsense", None, CTX)
    assert calls[0][1]["reranker"] == "CE" and calls[0][1]["threshold"] == rag.RERANK_THRESHOLD

def test_the_app_defaults_to_the_cross_encoder_gate_and_its_threshold():
    assert app.MODES[app.DEFAULT_MODE] == rag.RERANK_THRESHOLD and app.DEFAULT_MODE == app.RERANKED

def test_respond_sends_only_the_last_turns_of_the_history(monkeypatch):
    calls = []
    fake_pipeline(monkeypatch, calls=calls)
    history = [(f"q{i}", f"a{i}") for i in range(6)]
    app.respond("q?", rag.DEFAULT_MODEL, app.RERANKED, None, CTX, history)
    assert calls[0][1]["history"] == history[-rag.HISTORY_TURNS:]

def test_respond_returns_the_verdict_answer_sources_and_times(monkeypatch):
    fake_pipeline(monkeypatch)
    res = app.respond("q?", rag.DEFAULT_MODEL, app.RERANKED, None, CTX)
    assert (res["kind"], res["detail"]) == ("answered", "1 of 2 passages cited.")
    assert res["answer"] == ANSWER["answer"] and "<strong>daily</strong>" in res["answer_html"]
    assert (res["search_s"], res["model_s"], res["follow_up"]) == (0.3, 9.5, False) and res["total_s"] >= 0
    assert [s["cited"] for s in res["sources"]] == [True, False]

def test_an_llm_error_after_the_search_keeps_the_passages_and_the_search_time(monkeypatch):
    def boom(*a, **k):
        """Fail like a rate-limited model after a successful search."""
        exc = RuntimeError("429")
        exc.partial = {"hits": HITS, "best_similarity": 0.8, "rerank_score": 0.9, "search_s": 0.4, "follow_up": False}
        raise exc
    monkeypatch.setattr(rag, "answer_question", boom)
    res = app.respond("q?", rag.DEFAULT_MODEL, app.DENSE, None, CTX)
    assert res["kind"] == "error" and res["search_s"] == 0.4 and res["model_s"] is None
    assert [s["cited"] for s in res["sources"]] == [False, False]

def test_the_gradio_page_is_mounted_without_its_footer_links():
    client = TestClient(app.create_app(CTX))
    page = client.get("/gradio/")
    assert page.status_code == 200 and "gradio" in page.text.lower()

def test_respond_reports_an_llm_error_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        """Fail like an empty LLM reply."""
        raise RuntimeError("empty response from m:free")
    monkeypatch.setattr(rag, "answer_question", boom)
    res = app.respond("q?", rag.DEFAULT_MODEL, app.DENSE, None, CTX)
    assert res["kind"] == "error" and "empty response from m:free" in res["detail"]

def test_the_routes_serve_the_page_the_config_and_answers(monkeypatch):
    calls = []
    fake_pipeline(monkeypatch, calls=calls)
    client = TestClient(app.create_app(CTX))
    assert client.get("/").status_code == 200 and "text/html" in client.get("/").headers["content-type"]
    cfg = client.get("/api/config").json()
    assert cfg["default_mode"] == app.RERANKED and cfg["models"] == rag.FREE_MODELS
    res = client.post("/api/ask", json={"question": "and for children?", "history": [
        {"question": "Max sodium?", "answer": "Under 2 g."}]}).json()
    assert res["kind"] == "answered" and calls[0][0] == "and for children?"
    assert calls[0][1]["history"] == [("Max sodium?", "Under 2 g.")]

def test_the_ask_route_rejects_a_threshold_out_of_range(monkeypatch):
    fake_pipeline(monkeypatch)
    client = TestClient(app.create_app(CTX))
    assert client.post("/api/ask", json={"question": "q?", "threshold": 2}).status_code == 422

def test_the_app_has_no_api_documentation_pages():
    client = TestClient(app.create_app(CTX))
    assert [client.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]

def test_load_context_exits_with_one_line_when_the_key_is_missing(monkeypatch):
    monkeypatch.setattr(rag, "load_env", lambda: {})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        app.load_context()
    assert "OPENROUTER_API_KEY" in str(exc.value) and ".env" in str(exc.value)

def test_load_context_exits_when_the_index_is_missing_or_empty(monkeypatch):
    def missing():
        """Fail like Chroma when the collection does not exist."""
        raise NotFoundError("Collection [medical_docs] does not exist")
    def no_embedder():
        """Fail the test if the embedder is loaded."""
        raise AssertionError("load_embedder must not be called")
    monkeypatch.setattr(rag, "load_env", lambda: {"OPENROUTER_API_KEY": "k"})
    monkeypatch.setattr(index, "load_embedder", no_embedder)
    for opener in (missing, lambda: FakeCollection([])):
        monkeypatch.setattr(index, "open_index", opener)
        with pytest.raises(SystemExit) as exc:
            app.load_context()
        assert "src/index.py" in str(exc.value) and "\n" not in str(exc.value)

def test_load_context_lets_other_index_errors_through(monkeypatch):
    def locked():
        """Fail with an error that is not a missing index."""
        raise PermissionError("locked")
    def no_embedder():
        """Fail the test if the embedder is loaded."""
        raise AssertionError("load_embedder must not be called")
    monkeypatch.setattr(rag, "load_env", lambda: {"OPENROUTER_API_KEY": "k"})
    monkeypatch.setattr(index, "load_embedder", no_embedder)
    monkeypatch.setattr(index, "open_index", locked)
    with pytest.raises(PermissionError, match="locked"):
        app.load_context()

def test_load_context_reads_the_key_from_the_environment_without_a_dotenv(monkeypatch):
    def stop():
        """Stop after the key check: the index is not under test."""
        raise PermissionError("index reached")
    monkeypatch.setattr(rag, "load_env", lambda: {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    monkeypatch.setattr(index, "open_index", stop)
    with pytest.raises(PermissionError, match="index reached"):
        app.load_context()
