"""Tests for the evaluation metrics, the four systems, the judge and resumable runs (no network)."""
import csv

import pytest

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

def test_judge_faithfulness_returns_none_when_a_count_is_out_of_range(monkeypatch):
    for reply in ['{"supported": 5, "total": 4}', '{"supported": -1, "total": 4}', '{"supported": 0, "total": -2}']:
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
                     "faithfulness": 0.75, "n_judged": 2, "mean_latency_s": 1.5}
    assert list(dense)[list(dense).index("faithfulness") + 1] == "n_judged"
    assert (no_rag["answer_rate"], no_rag["refusal_rate"], no_rag["hit_rate"],
            no_rag["context_precision"], no_rag["faithfulness"], no_rag["n_judged"]) == (1.0, None, None, None, None, 0)

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

QUESTIONS = [{"id": "A01", "question": "Dose?", "answerable": True, "source": "W.pdf"},
             {"id": "O01", "question": "Capital?", "answerable": False, "source": None}]

def answer(*args):
    """Return an answered result with one hit from W.pdf, whatever the arguments."""
    hit = {"id": "a", "text": "t", "source": "W.pdf", "page": 2, "similarity": 0.7}
    return {"answer": "Take 1 g [W.pdf, p. 2]", "refused": False, "reason": "answered",
            "hits": [hit], "cited": [1], "best_similarity": 0.7}

def test_run_resumes_and_skips_finished_records(tmp_path, capsys):
    out = str(tmp_path / "records.jsonl")
    questions = QUESTIONS
    calls = []

    def runner(system, question, ctx, model, api_key, threshold):
        """Count the call and return an answer with one hit from W.pdf."""
        calls.append((system, question))
        return answer()

    def flaky(system, question, *rest):
        """Raise for the answerable question, else behave like runner."""
        if question == "Dose?":
            raise RuntimeError("model down")
        return runner(system, question, *rest)

    first = ev.run_questions(questions, ["rag_dense"], ["m:free"], {}, "k", 0.6, False, out, runner)
    assert len(calls) == 2
    assert [(r["system"], r["id"], r["relevant"], r["sources"]) for r in first] == [
        ("rag_dense", "A01", [True], ["W.pdf, p. 2"]), ("rag_dense", "O01", [False], ["W.pdf, p. 2"])]
    assert all(r["threshold"] == 0.6 for r in first)

    second = ev.run_questions(questions, ["rag_dense"], ["m:free"], {}, "k", 0.6, False, out, runner)
    assert len(calls) == 2 and second == first

    third = ev.run_questions(questions, ["no_rag"], ["m:free"], {}, "k", 0.6, False, out, flaky)
    assert calls[2:] == [("no_rag", "Capital?")]
    assert [(r["system"], r["id"]) for r in third] == [
        ("rag_dense", "A01"), ("rag_dense", "O01"), ("no_rag", "O01")]
    assert "model down" in capsys.readouterr().out
    with open(out, encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == 3

def test_run_stops_when_the_file_holds_records_of_another_threshold(tmp_path):
    out = str(tmp_path / "records.jsonl")
    ev.run_questions(QUESTIONS[:1], ["rag_dense"], ["m:free"], {}, "k", 0.6, False, out, answer)
    with open(out, encoding="utf-8") as fh:
        before = fh.read()
    with pytest.raises(SystemExit) as excinfo:
        ev.run_questions(QUESTIONS, ["rag_dense"], ["m:free"], {}, "k", 0.65, False, out, lambda *a: 1 / 0)
    message = str(excinfo.value)
    assert out in message and "0.6" in message and "0.65" in message and "delete" in message
    with open(out, encoding="utf-8") as fh:
        assert fh.read() == before

def test_a_failing_judge_still_records_the_answer(tmp_path, monkeypatch, capsys):
    out = str(tmp_path / "records.jsonl")
    monkeypatch.setattr(ev, "judge_faithfulness", lambda *a: 1 / 0)
    records = ev.run_questions(QUESTIONS[:1], ["rag_dense"], ["m:free"], {}, "k", 0.6, True, out, answer)
    assert [(r["id"], r["answer"], r["faithfulness"]) for r in records] == [("A01", "Take 1 g [W.pdf, p. 2]", None)]
    assert "rag_dense m:free A01: judge error: division by zero" in capsys.readouterr().out

def sheet_record(qid, answer_text):
    """Build a record with the keys the rating sheet reads."""
    return {"system": "rag_dense", "model": "m:free", "id": qid, "question": "Dose?", "answer": answer_text}

def read_sheet(path):
    """Return the rating sheet rows as dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def test_rating_sheet_keeps_ratings_of_unchanged_answers(tmp_path):
    path = str(tmp_path / "human_rating_sheet.csv")
    ev.write_rating_sheet(path, [sheet_record("A01", "Take 1 g [W.pdf, p. 2]"), sheet_record("A02", "Old")])
    rows = read_sheet(path)
    assert list(rows[0]) == ["system", "model", "id", "question", "answer", "safety_1_5", "clarity_1_5"]
    for row in rows:
        row["safety_1_5"], row["clarity_1_5"] = "4", "5"
    ev.write_csv(path, rows)
    ev.write_rating_sheet(path, [sheet_record("A01", "Take 1 g [W.pdf, p. 2]"), sheet_record("A02", "New"),
                                 sheet_record("A03", "Other")])
    assert [(r["id"], r["safety_1_5"], r["clarity_1_5"]) for r in read_sheet(path)] == [
        ("A01", "4", "5"), ("A02", "", ""), ("A03", "", "")]

def test_run_and_sweep_exit_with_the_load_context_message(tmp_path, monkeypatch):
    def missing():
        """Fail as load_context does when the index is missing."""
        raise SystemExit("The index is empty: run .venv/bin/python src/index.py first.")
    monkeypatch.setattr(rag, "load_context", missing)
    monkeypatch.setattr(rag, "load_env", lambda: {})
    monkeypatch.setattr(ev, "RESULTS", str(tmp_path / "results"))
    for call in [lambda: ev.run(["rag_dense"], ["m:free"], False, None), ev.sweep]:
        with pytest.raises(SystemExit) as excinfo:
            call()
        assert str(excinfo.value) == "The index is empty: run .venv/bin/python src/index.py first."
    assert not (tmp_path / "results").exists()

def test_run_checks_the_model_ids_before_loading_the_context(monkeypatch):
    monkeypatch.setattr(rag, "load_context", lambda: 1 / 0)
    monkeypatch.setattr(rag, "load_env", lambda: {})
    with pytest.raises(SystemExit) as excinfo:
        ev.run(["rag_dense"], ["openai/gpt-4o-mini"], False, None)
    assert "not a free model" in str(excinfo.value)
