"""Tests for the Gradio page mounted at /gradio: result formatting, respond and build_app. Fakes only."""
import gradio as gr
import pytest
from chromadb.errors import NotFoundError

import gradio_app as app
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

def test_format_result_for_a_refusal_keeps_the_refusal_sentence_and_cites_nothing():
    md, table = app.format_result({"answer": rag.REFUSAL, "refused": True, "reason": "below_threshold",
                                   "hits": HITS, "cited": [], "best_similarity": 0.41})
    assert rag.REFUSAL in md and app.DISCLAIMER in md
    assert [row[4] for row in table] == ["", ""]

def test_verdict_for_a_refusal_says_why():
    kind, label, detail = app.verdict({"answer": rag.REFUSAL, "refused": True, "reason": "below_threshold",
                                       "hits": HITS, "cited": [], "best_similarity": 0.41})
    assert kind == "refused" and "gate" in label and "0.41" in detail and "threshold" in detail

def test_verdict_for_an_answer_counts_the_cited_passages():
    assert app.verdict({"answer": "x", "refused": False, "reason": "answered", "hits": HITS, "cited": [1],
                        "best_similarity": 0.8}) == ("answered", "Answered", "1 of 2 passages cited.")

def test_verdict_for_an_uncited_answer_has_its_own_kind():
    kind, _, detail = app.verdict({"answer": rag.REFUSAL, "refused": True, "reason": "uncited", "hits": HITS,
                                   "cited": []})
    assert kind == "uncited" and "did not cite" in detail

def test_every_refusal_reason_has_a_verdict():
    assert set(app.VERDICTS) == set(app.REASONS) | {"answered"}

def test_sources_html_marks_cited_rows_and_escapes_the_source():
    html = app.sources_html([[1, "A <b> title", "-", "0.812", "yes"], [2, "WHO_09.pdf", "42", "-", ""]])
    assert html.count("<tr class=cited>") == 1 and "A &lt;b&gt; title" in html and "<b>" not in html
    assert all(f">{h}</th>" in html for h in app.HEADERS)
    assert app.sources_html([]) == ""

def test_respond_with_an_empty_question_does_not_call_the_pipeline(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda *a, **k: 1 / 0)
    status, md, table = app.respond("   ", rag.DEFAULT_MODEL, "Dense", 0.6, {})
    assert "Type a question first." in status and md == "" and table == ""

def test_respond_passes_bm25_in_both_hybrid_modes_and_the_reranker_only_with_the_gate(monkeypatch):
    calls = []
    def fake(question, embedder, collection, model, api_key, threshold, bm25, reranker, rerank_threshold):
        """Record the bm25, reranker and threshold arguments and refuse."""
        calls.append((bm25, reranker, threshold, rerank_threshold))
        return {"answer": rag.REFUSAL, "refused": True, "reason": "model_refused", "hits": [], "cited": [],
                "best_similarity": 0.7, "rerank_score": None}
    monkeypatch.setattr(rag, "answer_question", fake)
    ctx = {"embedder": None, "collection": None, "bm25": "BM25", "reranker": "CE", "api_key": "k"}
    for mode in (app.DENSE, app.HYBRID, app.RERANKED):
        app.respond("q?", rag.DEFAULT_MODEL, mode, 0.55, ctx)
    assert calls == [(None, None, 0.55, 0.55), ("BM25", None, 0.55, 0.55), ("BM25", "CE", 0.55, 0.55)]

def test_verdict_for_a_cross_encoder_refusal_gives_its_score():
    _, _, detail = app.verdict({"answer": rag.REFUSAL, "refused": True, "reason": "below_rerank_threshold",
                                "hits": HITS, "cited": [], "best_similarity": 0.7, "rerank_score": 0.123})
    assert "0.12" in detail and "cross-encoder" in detail.lower()

def test_the_app_defaults_to_the_cross_encoder_gate_and_its_threshold():
    assert app.MODES[app.DEFAULT_MODE] == rag.RERANK_THRESHOLD and app.DEFAULT_MODE == app.RERANKED
    assert app.MODES[app.DENSE] == app.MODES[app.HYBRID] == rag.SIM_THRESHOLD

def test_respond_reports_an_llm_error_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        """Fail like an empty LLM reply."""
        raise RuntimeError("empty response from m:free")
    monkeypatch.setattr(rag, "answer_question", boom)
    status, md, table = app.respond("q?", rag.DEFAULT_MODEL, "Dense", 0.6, {"embedder": None, "collection": None, "bm25": None, "api_key": "k"})
    assert "empty response from m:free" in status and "verdict-error" in status and md == "" and table == ""

def test_respond_escapes_the_error_text(monkeypatch):
    def boom(*a, **k):
        """Fail with markup in the message."""
        raise RuntimeError("<script>x</script>")
    monkeypatch.setattr(rag, "answer_question", boom)
    status, _, _ = app.respond("q?", rag.DEFAULT_MODEL, "Dense", 0.6, {"embedder": None, "collection": None, "bm25": None, "api_key": "k"})
    assert "<script>" not in status and "&lt;script&gt;" in status

def test_respond_returns_the_status_the_answer_and_the_sources_table(monkeypatch):
    res = {"answer": "Twice daily [WHO_09.pdf, p. 42].", "refused": False, "reason": "answered", "hits": HITS,
           "cited": [1], "best_similarity": 0.81}
    monkeypatch.setattr(rag, "answer_question", lambda *a, **k: res)
    status, md, table = app.respond("q?", rag.DEFAULT_MODEL, "Dense", 0.6, {"embedder": None, "collection": None, "bm25": None, "api_key": "k"})
    assert "verdict-answered" in status and "1 of 2 passages cited." in status
    assert md.startswith("Twice daily") and "<table" in table and "WHO_09.pdf" in table

def test_mark_citations_wraps_each_source_label_once_and_escapes_it():
    hits = [{"source": "WHO_09.pdf", "page": 42}, {"source": "A <b> title", "page": None}]
    md = app.mark_citations("Twice daily [WHO_09.pdf, p. 42][A <b> title]. Again [WHO_09.pdf, p. 42].", hits)
    assert md.count('<span class="cite">[WHO_09.pdf, p. 42]</span>') == 2
    assert '<span class="cite">[A &lt;b&gt; title]</span>' in md and "<b>" not in md

def test_build_theme_has_one_look_in_light_and_dark():
    theme = app.build_theme()
    assert theme.body_background_fill_dark == theme.body_background_fill == "#FFFFFF"
    assert theme.button_primary_background_fill_dark == theme.button_primary_background_fill

def test_build_app_constructs_without_launching():
    assert isinstance(app.build_app({"embedder": None, "collection": None, "bm25": None, "api_key": "k"}), gr.Blocks)
