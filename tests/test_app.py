"""Tests for the Gradio app: result formatting, respond, load_context and build_app. Fakes only."""
import gradio as gr
import pytest

import app
import index
import rag
from conftest import FakeCollection

HITS = [{"id": "a", "text": "Artemether text", "source": "WHO_09.pdf", "page": 42, "similarity": 0.81234},
        {"id": "b", "text": "Other text", "source": "Title.", "page": None, "similarity": None}]

def test_format_result_for_an_answer_marks_cited_sources():
    md, table = app.format_result({"answer": "Twice daily [WHO_09.pdf, p. 42].", "refused": False,
                                   "reason": "answered", "hits": HITS, "cited": [1], "best_similarity": 0.81234})
    assert "Twice daily [WHO_09.pdf, p. 42]." in md and app.DISCLAIMER in md
    assert table == [[1, "WHO_09.pdf", "42", "0.812", "yes"], [2, "Title.", "-", "-", ""]]

def test_format_result_for_a_refusal_says_why():
    md, table = app.format_result({"answer": rag.REFUSAL, "refused": True, "reason": "below_threshold",
                                   "hits": HITS, "cited": [], "best_similarity": 0.41})
    assert rag.REFUSAL in md and "0.41" in md and "threshold" in md
    assert [row[4] for row in table] == ["", ""]

def test_respond_with_an_empty_question_does_not_call_the_pipeline(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda *a, **k: 1 / 0)
    md, table = app.respond("   ", rag.DEFAULT_MODEL, "Dense", 0.6, {})
    assert md == "Type a question first." and table == []

def test_respond_passes_bm25_only_in_hybrid_mode(monkeypatch):
    calls = []
    def fake(question, embedder, collection, model, api_key, threshold, bm25):
        """Record the bm25 argument and refuse."""
        calls.append(bm25)
        return {"answer": rag.REFUSAL, "refused": True, "reason": "model_refused", "hits": [], "cited": [], "best_similarity": 0.7}
    monkeypatch.setattr(rag, "answer_question", fake)
    ctx = {"embedder": None, "collection": None, "bm25": "BM25", "api_key": "k"}
    app.respond("q?", rag.DEFAULT_MODEL, "Dense", 0.6, ctx)
    app.respond("q?", rag.DEFAULT_MODEL, "Hybrid (dense + BM25)", 0.6, ctx)
    assert calls == [None, "BM25"]

def test_respond_reports_an_llm_error_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        """Fail like an empty LLM reply."""
        raise RuntimeError("empty response from m:free")
    monkeypatch.setattr(rag, "answer_question", boom)
    md, table = app.respond("q?", rag.DEFAULT_MODEL, "Dense", 0.6, {"embedder": None, "collection": None, "bm25": None, "api_key": "k"})
    assert "empty response from m:free" in md and table == []

def test_load_context_exits_with_one_line_when_the_key_is_missing(monkeypatch):
    monkeypatch.setattr(rag, "load_env", lambda: {})
    with pytest.raises(SystemExit) as exc:
        app.load_context()
    assert "OPENROUTER_API_KEY" in str(exc.value) and ".env" in str(exc.value)

def test_load_context_exits_when_the_index_is_missing_or_empty(monkeypatch):
    def missing():
        """Fail like Chroma when the collection does not exist."""
        raise ValueError("Collection medical_docs does not exist.")
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

def test_build_app_constructs_without_launching():
    assert isinstance(app.build_app({"embedder": None, "collection": None, "bm25": None, "api_key": "k"}), gr.Blocks)
