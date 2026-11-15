"""Tests for numbered citations, the refusal gate, retrieval and call_llm (no network)."""
import io
import json
import math
import urllib.error

import pytest

import rag
from conftest import FakeCollection, FakeEmbedder

WHO = {"id": "a", "text": "Artemether is given twice daily.", "source": "WHO_09.pdf", "page": 42, "similarity": 0.8}
PMC = {"id": "b", "text": "Paracetamol 1 g.", "source": "Paracetamol dosing in adults.", "page": None, "similarity": 0.7}


def test_source_label_with_and_without_page():
    assert rag.source_label(WHO) == "WHO_09.pdf, p. 42"
    assert rag.source_label(PMC) == "Paracetamol dosing in adults."


def test_format_context_numbers_passages_from_one():
    assert rag.format_context([WHO, PMC]) == (
        "[1] WHO_09.pdf, p. 42\nArtemether is given twice daily.\n\n"
        "[2] Paracetamol dosing in adults.\nParacetamol 1 g.")


def test_format_context_turns_reference_markers_of_the_text_into_parentheses():
    hit = {**PMC, "text": "Paracetamol 1 g [3]. See also [12] and [3, 4]."}
    out = rag.format_context([WHO, hit])
    assert "[1] WHO_09.pdf" in out and "[2] Paracetamol dosing in adults." in out
    assert "Paracetamol 1 g (3). See also (12) and (3, 4)." in out
    assert rag.cited_numbers(out) == [1, 2]
    assert hit["text"] == "Paracetamol 1 g [3]. See also [12] and [3, 4]."


def test_cited_numbers_accepts_single_adjacent_and_comma_forms():
    assert rag.cited_numbers("A [2]. B [1][3]. C [1, 4].") == [1, 2, 3, 4]
    assert rag.cited_numbers("no citation, dose is 500 mg") == []


def test_expand_citations_replaces_numbers_with_labels():
    out = rag.expand_citations("Twice daily [1]. One gram [1, 2].", [WHO, PMC])
    assert out == ("Twice daily [WHO_09.pdf, p. 42]. "
                   "One gram [WHO_09.pdf, p. 42][Paracetamol dosing in adults.].")


def llm(text):
    """Return a fake call_llm that always answers with text."""
    return lambda prompt, model, api_key: text


def test_generate_answer_answers_when_every_citation_is_a_given_passage(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", llm("Twice daily [1]."))
    res = rag.generate_answer("q", [WHO, PMC], rag.DEFAULT_MODEL, "k")
    assert (res["refused"], res["reason"], res["cited"]) == (False, "answered", [1])
    assert res["answer"] == "Twice daily [WHO_09.pdf, p. 42]."


def test_generate_answer_refuses_uncited_and_out_of_range(monkeypatch):
    for text in ["Twice daily.", "Twice daily [3].", "Twice daily [0]."]:
        monkeypatch.setattr(rag, "call_llm", llm(text))
        res = rag.generate_answer("q", [WHO, PMC], rag.DEFAULT_MODEL, "k")
        assert (res["refused"], res["reason"], res["answer"]) == (True, "uncited", rag.REFUSAL)
        assert res["raw"] == text and res["cited"] == []


def test_generate_answer_passes_on_the_model_refusal(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", llm("i cannot answer this question from the provided documents. [1]"))
    res = rag.generate_answer("q", [WHO], rag.DEFAULT_MODEL, "k")
    assert (res["refused"], res["reason"], res["answer"]) == (True, "model_refused", rag.REFUSAL)


def test_generate_answer_with_no_hits_refuses_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda *a: 1 / 0)
    res = rag.generate_answer("q", [], rag.DEFAULT_MODEL, "k")
    assert (res["refused"], res["reason"]) == (True, "no_context")


def rows(distance):
    """Return two fake index rows; the first one's distance is given."""
    return [("a", "Artemether text", {"source": "WHO_09.pdf", "kind": "WHO", "page": 42}, distance),
            ("b", "Paracetamol text", {"source": "Title.", "kind": "PubMed", "page": -1}, 0.5)]


def test_retrieve_returns_id_similarity_and_none_page():
    hits = rag.retrieve("q", FakeEmbedder(), FakeCollection(rows(0.25)))
    assert hits[0] == {"id": "a", "text": "Artemether text", "source": "WHO_09.pdf", "page": 42, "similarity": 0.75}
    assert hits[1]["page"] is None


def test_answer_question_refuses_below_threshold_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda *a: 1 / 0)
    res = rag.answer_question("q", FakeEmbedder(), FakeCollection(rows(0.45)), rag.DEFAULT_MODEL, "k", threshold=0.6)
    assert (res["refused"], res["reason"], res["answer"]) == (True, "below_threshold", rag.REFUSAL)
    assert abs(res["best_similarity"] - 0.55) < 1e-9


def test_answer_question_answers_above_threshold(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", llm("Given twice daily [1]."))
    res = rag.answer_question("q", FakeEmbedder(), FakeCollection(rows(0.25)), rag.DEFAULT_MODEL, "k", threshold=0.6)
    assert res["reason"] == "answered" and abs(res["best_similarity"] - 0.75) < 1e-9


def test_check_free_rejects_paid_models():
    rag.check_free("openrouter/free")
    with pytest.raises(ValueError):
        rag.check_free("openai/gpt-4o-mini")


def http_error(code, body=b""):
    """Return the HTTPError urlopen raises for this status code, with this response body."""
    return urllib.error.HTTPError(rag.OPENROUTER_URL, code, "error", {}, io.BytesIO(body))


DAILY_LIMIT = (b'{"error":{"code":429,"message":"Rate limit exceeded: free-models-per-day. '
               b'Add 10 credits to unlock 1000 free model requests per day"}}')


def fake_urlopen(monkeypatch, outcomes):
    """Patch urlopen to play outcomes in order (message text, raw JSON body dict or exception); return the calls made."""
    calls = []

    def urlopen(req, timeout):
        """Return an OpenRouter-shaped body, or raise, for the next outcome."""
        calls.append(req)
        out = outcomes[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        body = out if isinstance(out, dict) else {"choices": [{"message": {"content": out}}]}
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(rag.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(rag.time, "sleep", lambda seconds: None)
    return calls


def test_call_llm_retries_empty_content_then_returns_text(monkeypatch):
    calls = fake_urlopen(monkeypatch, ["", "ok"])
    assert rag.call_llm("p", rag.DEFAULT_MODEL, "k") == "ok"
    assert len(calls) == 2


def test_call_llm_retries_429_and_503(monkeypatch):
    calls = fake_urlopen(monkeypatch, [http_error(429), http_error(503), "ok"])
    assert rag.call_llm("p", rag.DEFAULT_MODEL, "k") == "ok"
    assert len(calls) == 3


def test_call_llm_stops_at_once_when_the_daily_free_limit_is_reached(monkeypatch):
    calls = fake_urlopen(monkeypatch, [http_error(429, DAILY_LIMIT), "ok"])
    monkeypatch.setattr(rag.time, "sleep", lambda seconds: 1 / 0)
    with pytest.raises(rag.QuotaExhausted) as excinfo:
        rag.call_llm("p", rag.DEFAULT_MODEL, "k")
    assert "daily free-model limit" in str(excinfo.value) and "00:00 UTC" in str(excinfo.value)
    assert len(calls) == 1


def test_call_llm_retries_a_429_whose_body_is_another_limit_or_unreadable(monkeypatch):
    per_minute = http_error(429, b'{"error":{"code":429,"message":"Rate limit exceeded: free-models-per-min."}}')
    unreadable = urllib.error.HTTPError(rag.OPENROUTER_URL, 429, "error", {}, None)
    calls = fake_urlopen(monkeypatch, [per_minute, unreadable, "ok"])
    assert rag.call_llm("p", rag.DEFAULT_MODEL, "k") == "ok"
    assert len(calls) == 3


def test_call_llm_raises_on_401_at_once(monkeypatch):
    calls = fake_urlopen(monkeypatch, [http_error(401), "ok"])
    with pytest.raises(urllib.error.HTTPError):
        rag.call_llm("p", rag.DEFAULT_MODEL, "k")
    assert len(calls) == 1


def test_call_llm_raises_runtime_error_when_always_empty(monkeypatch):
    calls = fake_urlopen(monkeypatch, ["", ""])
    with pytest.raises(RuntimeError) as excinfo:
        rag.call_llm("p", rag.DEFAULT_MODEL, "k", retries=2)
    assert rag.DEFAULT_MODEL in str(excinfo.value)
    assert len(calls) == 2


def test_call_llm_reports_the_error_field_of_a_200_reply(monkeypatch):
    error = {"error": {"message": "rate limited", "code": 429}}
    calls = fake_urlopen(monkeypatch, [error, error])
    with pytest.raises(RuntimeError) as excinfo:
        rag.call_llm("p", rag.DEFAULT_MODEL, "k", retries=2)
    assert "rate limited" in str(excinfo.value) and rag.DEFAULT_MODEL in str(excinfo.value)
    assert len(calls) == 2


def test_call_llm_sends_the_token_budget(monkeypatch):
    calls = fake_urlopen(monkeypatch, ["ok", "ok"])
    rag.call_llm("p", rag.DEFAULT_MODEL, "k")
    rag.call_llm("p", rag.DEFAULT_MODEL, "k", max_tokens=4000)
    assert [json.loads(c.data)["max_tokens"] for c in calls] == [600, 4000]


CUT = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}


def test_call_llm_resends_once_with_a_larger_budget_when_reasoning_uses_the_whole_budget(monkeypatch):
    calls = fake_urlopen(monkeypatch, [CUT, "ok"])
    monkeypatch.setattr(rag.time, "sleep", lambda seconds: 1 / 0)
    assert rag.call_llm("p", rag.DEFAULT_MODEL, "k") == "ok"
    assert [json.loads(c.data)["max_tokens"] for c in calls] == [600, rag.LENGTH_RETRY_TOKENS]


def test_call_llm_raises_when_the_larger_budget_is_used_up_too(monkeypatch):
    calls = fake_urlopen(monkeypatch, [CUT, CUT, "ok"])
    with pytest.raises(RuntimeError) as excinfo:
        rag.call_llm("p", rag.DEFAULT_MODEL, "k")
    assert f"max_tokens={rag.LENGTH_RETRY_TOKENS}" in str(excinfo.value) and rag.DEFAULT_MODEL in str(excinfo.value)
    assert len(calls) == 2


def test_call_llm_raises_at_once_when_a_large_budget_is_used_up(monkeypatch):
    calls = fake_urlopen(monkeypatch, [CUT, "ok"])
    with pytest.raises(RuntimeError):
        rag.call_llm("p", rag.DEFAULT_MODEL, "k", max_tokens=rag.LENGTH_RETRY_TOKENS)
    assert len(calls) == 1


def test_all_chunks_reads_every_chunk_from_the_collection():
    from index import all_chunks
    assert all_chunks(FakeCollection(rows(0.25))) == [
        {"id": "a", "text": "Artemether text", "source": "WHO_09.pdf", "page": 42},
        {"id": "b", "text": "Paracetamol text", "source": "Title.", "page": None}]


def test_rrf_fuses_by_reciprocal_rank_and_keeps_the_first_dict():
    def h(i, sim):
        """Return a minimal hit with an id and a similarity."""
        return {"id": i, "similarity": sim}
    dense = [h("A", 0.9), h("B", 0.8), h("C", 0.7)]
    sparse = [h("C", None), h("A", None), h("D", None)]
    fused = rag.rrf([dense, sparse])
    assert [x["id"] for x in fused] == ["A", "C", "B", "D"]
    assert fused[1]["similarity"] == 0.7      # the dense dict wins for C


def test_hybrid_answer_question_gates_on_dense_similarity_and_fuses(monkeypatch):
    from bm25 import BM25
    from index import all_chunks
    coll = FakeCollection(rows(0.25))
    seen = {}

    def fake_llm(prompt, model, api_key):
        """Record the prompt and answer citing passage 1."""
        seen["prompt"] = prompt
        return "Paracetamol [1]."
    monkeypatch.setattr(rag, "call_llm", fake_llm)
    res = rag.answer_question("paracetamol", FakeEmbedder(), coll, rag.DEFAULT_MODEL, "k",
                              threshold=0.6, bm25=BM25(all_chunks(coll)))
    assert res["reason"] == "answered" and abs(res["best_similarity"] - 0.75) < 1e-9
    assert [h["id"] for h in res["hits"]] == ["b", "a"]   # b: ranks 2 and 1; a: rank 1 only
    assert len(res["hits"]) <= rag.TOP_K


def test_hybrid_answer_question_refuses_below_threshold_even_when_bm25_matches(monkeypatch):
    from bm25 import BM25
    from index import all_chunks
    coll = FakeCollection(rows(0.45))   # best dense similarity 0.55
    bm25 = BM25(all_chunks(coll))
    assert bm25.search("paracetamol", rag.CANDIDATES)[0]["id"] == "b"
    monkeypatch.setattr(rag, "call_llm", lambda *a: 1 / 0)
    res = rag.answer_question("paracetamol", FakeEmbedder(), coll, rag.DEFAULT_MODEL, "k",
                              threshold=0.6, bm25=bm25)
    assert (res["refused"], res["reason"], res["answer"]) == (True, "below_threshold", rag.REFUSAL)
    assert abs(res["best_similarity"] - 0.55) < 1e-9


def test_reranker_gate_refuses_a_low_cross_encoder_score_even_when_similarity_is_high(monkeypatch):
    from conftest import FakeReranker
    monkeypatch.setattr(rag, "call_llm", lambda *a, **k: 1 / 0)
    res = rag.answer_question("q", FakeEmbedder(), FakeCollection(rows(0.1)), rag.DEFAULT_MODEL, "k",
                              reranker=FakeReranker([-2.0, -3.0]))
    assert (res["refused"], res["reason"], res["answer"]) == (True, "below_rerank_threshold", rag.REFUSAL)
    assert abs(res["best_similarity"] - 0.9) < 1e-9 and abs(res["rerank_score"] - 1 / (1 + math.exp(2.0))) < 1e-9


def test_reranker_gate_answers_a_high_cross_encoder_score_even_when_similarity_is_low(monkeypatch):
    from conftest import FakeReranker
    monkeypatch.setattr(rag, "call_llm", lambda *a, **k: "Twice daily [1].")
    res = rag.answer_question("q", FakeEmbedder(), FakeCollection(rows(0.45)), rag.DEFAULT_MODEL, "k",
                              reranker=FakeReranker([-1.0, 2.0]))
    assert (res["refused"], res["reason"]) == (False, "answered")
    assert [h["id"] for h in res["hits"]] == ["a", "b"]          # passages keep the retrieval order
    assert abs(res["rerank_score"] - 1 / (1 + math.exp(-2.0))) < 1e-9


def test_reranker_gate_stops_at_the_first_passage_that_reaches_the_threshold(monkeypatch):
    from conftest import FakeReranker
    monkeypatch.setattr(rag, "call_llm", lambda *a, **k: "Twice daily [1].")
    reranker = FakeReranker([-1.0, 5.0])
    res = rag.answer_question("q", FakeEmbedder(), FakeCollection(rows(0.1)), rag.DEFAULT_MODEL, "k",
                              reranker=reranker, rerank_threshold=0.2)   # sigmoid(-1.0) is 0.269
    assert res["reason"] == "answered" and len(reranker.calls) == 1
    assert abs(res["rerank_score"] - 1 / (1 + math.exp(1.0))) < 1e-9


def test_answer_question_without_a_reranker_has_no_rerank_score(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda *a, **k: "Twice daily [1].")
    res = rag.answer_question("q", FakeEmbedder(), FakeCollection(rows(0.1)), rag.DEFAULT_MODEL, "k")
    assert res["rerank_score"] is None
