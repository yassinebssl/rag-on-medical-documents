"""Tests for the evaluation metrics, the four systems, the judge and resumable runs (no network)."""
import evaluate as ev
import rag

def test_average_precision():
    assert abs(ev.average_precision([True, False, True]) - (1 + 2 / 3) / 2) < 1e-9
    assert ev.average_precision([False, False]) == 0.0
    assert ev.average_precision([]) == 0.0

def test_sweep_and_best_threshold_prefers_the_higher_threshold_on_a_tie():
    points = [(0.7, True), (0.55, True), (0.5, False), (0.3, False)]   # (best similarity, answerable)
    rows = ev.sweep_thresholds(points, [0.4, 0.6])
    assert rows == [{"threshold": 0.4, "answer_rate": 1.0, "refusal_rate": 0.5, "balanced": 0.75},
                    {"threshold": 0.6, "answer_rate": 0.5, "refusal_rate": 1.0, "balanced": 0.75}]
    assert ev.best_threshold(rows) == 0.6

def test_judge_faithfulness_parses_the_json_in_the_reply(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda p, m, k: 'Result: {"supported": 3, "total": 4}')
    assert ev.judge_faithfulness("answer", [{"text": "t", "source": "s", "page": 1}], "m:free", "k") == 0.75

def test_judge_faithfulness_returns_none_on_garbage_or_zero_total(monkeypatch):
    for reply in ["no json here", '{"supported": 0, "total": 0}', '{"supported": "x"}']:
        monkeypatch.setattr(rag, "call_llm", lambda p, m, k, reply=reply: reply)
        assert ev.judge_faithfulness("a", [], "m:free", "k") is None

def rec(system, qid, answerable, refused, relevant, faith=None, latency=1.0):
    """Build a minimal record with the keys summarize reads."""
    return {"system": system, "model": "m:free", "id": qid, "answerable": answerable, "refused": refused,
            "relevant": relevant, "faithfulness": faith, "latency_s": latency}

def test_summarize_computes_rates_per_system_and_model():
    records = [rec("rag_dense", "A01", True, False, [True, False], 1.0),
               rec("rag_dense", "A02", True, True, [False, False]),
               rec("rag_dense", "O01", False, True, [False, False]),
               rec("rag_dense", "O02", False, False, [False, False], 0.5, 3.0),
               rec("no_rag", "A01", True, False, [])]
    dense, no_rag = ev.summarize(records)
    assert dense == {"system": "rag_dense", "model": "m:free", "n": 4, "answer_rate": 0.5,
                     "refusal_rate": 0.5, "hit_rate": 0.5, "context_precision": 0.5,
                     "faithfulness": 0.75, "mean_latency_s": 1.5}
    assert (no_rag["answer_rate"], no_rag["refusal_rate"], no_rag["hit_rate"],
            no_rag["context_precision"], no_rag["faithfulness"]) == (1.0, None, None, None, None)

def test_run_system_no_rag_uses_no_passages(monkeypatch):
    seen = []
    monkeypatch.setattr(rag, "call_llm", lambda p, m, k: seen.append(p) or "Take 1 g.")
    res = ev.run_system("no_rag", "Dose?", {}, "m:free", "k", 0.6)
    assert (res["refused"], res["reason"], res["hits"], res["best_similarity"]) == (False, "answered", [], None)
    assert "Passages" not in seen[0] and "Dose?" in seen[0]

def test_run_system_bm25_rag_has_no_similarity_gate(monkeypatch):
    from bm25 import BM25
    monkeypatch.setattr(rag, "call_llm", lambda p, m, k: "Twice daily [1].")
    ctx = {"bm25": BM25([{"id": "a", "text": "artemether twice daily", "source": "W.pdf", "page": 2}])}
    res = ev.run_system("bm25_rag", "artemether", ctx, "m:free", "k", 0.99)
    assert res["reason"] == "answered" and res["best_similarity"] is None

def test_run_resumes_and_skips_finished_records(tmp_path, capsys):
    out = str(tmp_path / "records.jsonl")
    questions = [{"id": "A01", "question": "Dose?", "answerable": True, "source": "W.pdf"},
                 {"id": "O01", "question": "Capital?", "answerable": False, "source": None}]
    calls = []

    def runner(system, question, ctx, model, api_key, threshold):
        """Count the call and return an answer with one hit from W.pdf."""
        calls.append((system, question))
        hit = {"id": "a", "text": "t", "source": "W.pdf", "page": 2, "similarity": 0.7}
        return {"answer": "Take 1 g [W.pdf, p. 2]", "refused": False, "reason": "answered",
                "hits": [hit], "cited": [1], "best_similarity": 0.7}

    def flaky(system, question, *rest):
        """Raise for the answerable question, else behave like runner."""
        if question == "Dose?":
            raise RuntimeError("model down")
        return runner(system, question, *rest)

    first = ev.run_questions(questions, ["rag_dense"], ["m:free"], {}, "k", 0.6, False, out, runner)
    assert len(calls) == 2
    assert [(r["system"], r["id"], r["relevant"], r["sources"]) for r in first] == [
        ("rag_dense", "A01", [True], ["W.pdf, p. 2"]), ("rag_dense", "O01", [False], ["W.pdf, p. 2"])]

    second = ev.run_questions(questions, ["rag_dense"], ["m:free"], {}, "k", 0.6, False, out, runner)
    assert len(calls) == 2 and second == first

    third = ev.run_questions(questions, ["no_rag"], ["m:free"], {}, "k", 0.6, False, out, flaky)
    assert calls[2:] == [("no_rag", "Capital?")]
    assert [(r["system"], r["id"]) for r in third] == [
        ("rag_dense", "A01"), ("rag_dense", "O01"), ("no_rag", "O01")]
    assert "model down" in capsys.readouterr().out
    with open(out, encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == 3
