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
    monkeypatch.setattr(rag, "call_llm", lambda p, m, k, **kw: 'Result: {"supported": 3, "total": 4}')
    assert ev.judge_faithfulness("answer", [{"text": "t", "source": "s", "page": 1}], "m:free", "k") == 0.75

def test_judge_faithfulness_returns_none_on_garbage_or_zero_total(monkeypatch):
    for reply in ["no json here", '{"supported": 0, "total": 0}', '{"supported": "x"}']:
        monkeypatch.setattr(rag, "call_llm", lambda p, m, k, reply=reply, **kw: reply)
        assert ev.judge_faithfulness("a", [], "m:free", "k") is None

def test_judge_faithfulness_returns_none_when_a_count_is_out_of_range(monkeypatch):
    for reply in ['{"supported": 5, "total": 4}', '{"supported": -1, "total": 4}', '{"supported": 0, "total": -2}']:
        monkeypatch.setattr(rag, "call_llm", lambda p, m, k, reply=reply, **kw: reply)
        assert ev.judge_faithfulness("a", [], "m:free", "k") is None

def test_judge_faithfulness_gives_the_judge_a_larger_token_budget(monkeypatch):
    seen = []
    monkeypatch.setattr(rag, "call_llm", lambda p, m, k, **kw: seen.append(kw) or '{"supported": 1, "total": 1}')
    assert ev.judge_faithfulness("a", [], "m:free", "k") == 1.0
    assert seen == [{"max_tokens": 4000}]

def test_no_rag_prompt_says_no_documents_are_given_and_keeps_the_refusal_sentence():
    assert "No documents are given" in ev.NO_RAG_PROMPT and "own medical knowledge" in ev.NO_RAG_PROMPT
    assert rag.REFUSAL in ev.NO_RAG_PROMPT

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

SHEET_QUESTIONS = [{"id": "A01", "ground_truth": "1 g."}, {"id": "A02", "ground_truth": "2 g."},
             {"id": "A03", "ground_truth": "3 g."}, {"id": "O01", "ground_truth": None}]

def test_rating_sheet_keeps_ratings_of_unchanged_answers(tmp_path):
    path = str(tmp_path / "human_rating_sheet.csv")
    ev.write_rating_sheet(path, [sheet_record("A01", "Take 1 g [W.pdf, p. 2]"), sheet_record("A02", "Old")],
                          SHEET_QUESTIONS)
    rows = read_sheet(path)
    assert list(rows[0]) == ["system", "model", "id", "question", "expected_answer", "answer",
                             "correct_0_1", "safety_1_5", "clarity_1_5"]
    for row in rows:
        row["correct_0_1"], row["safety_1_5"], row["clarity_1_5"] = "1", "4", "5"
    ev.write_csv(path, rows)
    ev.write_rating_sheet(path, [sheet_record("A01", "Take 1 g [W.pdf, p. 2]"), sheet_record("A02", "New"),
                                 sheet_record("A03", "Other")], SHEET_QUESTIONS)
    assert [(r["id"], r["correct_0_1"], r["safety_1_5"], r["clarity_1_5"]) for r in read_sheet(path)] == [
        ("A01", "1", "4", "5"), ("A02", "", "", ""), ("A03", "", "", "")]

def test_rating_sheet_shows_the_expected_answer_and_the_refusal_for_out_of_corpus(tmp_path):
    path = str(tmp_path / "human_rating_sheet.csv")
    ev.write_rating_sheet(path, [sheet_record("A02", "2 g [W.pdf, p. 2]"), sheet_record("O01", rag.REFUSAL)],
                          SHEET_QUESTIONS)
    assert [r["expected_answer"] for r in read_sheet(path)] == ["2 g.", rag.REFUSAL]

def test_run_and_sweep_exit_with_the_load_context_message(tmp_path, monkeypatch):
    def missing():
        """Fail as load_context does when the index is missing."""
        raise SystemExit("The index is empty: run .venv/bin/python src/index.py first.")
    monkeypatch.setattr(rag, "load_context", missing)
    monkeypatch.setattr(rag, "load_env", lambda: {})
    monkeypatch.setattr(ev, "RESULTS", str(tmp_path / "results"))
    for call in [lambda: ev.run(["rag_dense"], ["m:free"], False, None), ev.sweep,
                 lambda: ev.retrieval(["medcpt"])]:
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

def test_separation_of_fully_separated_scores():
    points = [(0.9, True), (0.7, True), (0.5, False), (0.3, False)]
    assert ev.separation(points) == {"auroc": 1.0, "margin": 0.2, "gap_share": 0.333}

def test_separation_counts_ties_as_half_and_gives_a_negative_margin_on_overlap():
    points = [(0.6, True), (0.4, True), (0.5, False), (0.4, False)]
    assert ev.separation(points) == {"auroc": 0.625, "margin": -0.1, "gap_share": -0.5}

RETRIEVAL_QUESTIONS = [
    {"id": "A01", "question": "q1", "answerable": True, "source": "W.pdf", "page": 3},
    {"id": "A02", "question": "q2", "answerable": True, "source": "Title.", "page": None},
    {"id": "O01", "question": "q3", "answerable": False, "source": None, "page": None}]

def test_evaluate_retrieval_scores_each_retriever_on_hits_pages_and_separation():
    answers = {"q1": ([{"source": "X.pdf", "page": 1}, {"source": "W.pdf", "page": 3}], 0.8),
               "q2": ([{"source": "Title.", "page": None}], 0.7),
               "q3": ([{"source": "W.pdf", "page": 9}], 0.4)}
    wrong_page = {"q1": ([{"source": "W.pdf", "page": 4}], 0.5), "q2": ([], 0.0), "q3": ([], 0.6)}
    rows, per_question = ev.evaluate_retrieval(RETRIEVAL_QUESTIONS, {"good": answers.get, "bad": wrong_page.get})
    good, bad = ({k: v for k, v in row.items() if k != "mean_latency_ms"} for row in rows)
    assert good == {"retriever": "good", "hit_rate": 1.0, "page_hit_rate": 1.0, "context_precision": 0.75,
                    "auroc": 1.0, "margin": 0.3, "gap_share": 0.75}
    assert (bad["hit_rate"], bad["page_hit_rate"], bad["context_precision"], bad["auroc"]) == (0.5, 0.0, 0.5, 0.0)
    assert per_question[0] == {"id": "A01", "answerable": True, "good_best": 0.8, "good_page_hit": True,
                               "bad_best": 0.5, "bad_page_hit": False}
    assert per_question[2]["good_page_hit"] == "" and per_question[2]["good_best"] == 0.4

def test_retrieval_command_rejects_an_unknown_reranker(monkeypatch, capsys):
    monkeypatch.setattr(ev, "retrieval", lambda names: 1 / 0)
    monkeypatch.setattr("sys.argv", ["evaluate.py", "retrieval", "--rerankers", "medcpt,nope"])
    with pytest.raises(SystemExit):
        ev.main()
    assert "unknown reranker(s) nope" in capsys.readouterr().err

def test_run_system_rag_rerank_uses_bm25_and_the_reranker(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag, "answer_question", lambda *a, **k: seen.update(k) or {"refused": True})
    ev.run_system("rag_rerank", "q", {"embedder": None, "collection": None, "bm25": "B", "reranker": "R"},
                  "m:free", "k", 0.6)
    assert (seen["bm25"], seen["reranker"]) == ("B", "R") and "rag_rerank" in ev.SYSTEMS


class FakeLLM:
    """Stands in for rag.call_llm: records each request and replies with its number."""

    def __init__(self):
        """Start with no request."""
        self.requests = []

    def __call__(self, prompt, model, api_key, **kwargs):
        """Record the request and return 'reply 1', 'reply 2', ..."""
        self.requests.append((prompt, model, kwargs))
        return f"reply {len(self.requests)}"


def test_reply_cache_sends_an_identical_request_once(tmp_path):
    llm = FakeLLM()
    cache = ev.ReplyCache(str(tmp_path / "cache.jsonl"), llm)
    assert [cache("Dose?", "m:free", "k"), cache("Dose?", "m:free", "k")] == ["reply 1", "reply 1"]
    assert len(llm.requests) == 1


def test_reply_cache_sends_a_request_that_differs_in_prompt_model_or_token_budget(tmp_path):
    llm = FakeLLM()
    cache = ev.ReplyCache(str(tmp_path / "cache.jsonl"), llm)
    replies = [cache("Dose?", "m:free", "k"), cache("Course?", "m:free", "k"), cache("Dose?", "n:free", "k"),
               cache("Dose?", "m:free", "k", max_tokens=4000)]
    assert replies == ["reply 1", "reply 2", "reply 3", "reply 4"]
    assert llm.requests[3] == ("Dose?", "m:free", {"max_tokens": 4000})


def test_reply_cache_reuses_the_replies_of_an_earlier_run(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    ev.ReplyCache(path, FakeLLM())("Dose?", "m:free", "k")
    later = FakeLLM()
    assert ev.ReplyCache(path, later)("Dose?", "m:free", "k") == "reply 1" and later.requests == []


def test_reply_cache_file_holds_neither_the_api_key_nor_the_prompt(tmp_path):
    path = tmp_path / "cache.jsonl"
    ev.ReplyCache(str(path), FakeLLM())("What is the dose of paracetamol?", "m:free", "secret-key")
    stored = path.read_text(encoding="utf-8")
    assert "reply 1" in stored and "secret-key" not in stored and "paracetamol" not in stored


def test_reply_cache_does_not_store_a_failed_call(tmp_path):
    path = tmp_path / "cache.jsonl"
    with pytest.raises(ZeroDivisionError):
        ev.ReplyCache(str(path), lambda *a, **k: 1 / 0)("Dose?", "m:free", "k")
    llm = FakeLLM()
    assert ev.ReplyCache(str(path), llm)("Dose?", "m:free", "k") == "reply 1" and len(llm.requests) == 1


def test_reply_cache_adds_up_the_seconds_the_reused_replies_took(tmp_path):
    clock = iter([10.0, 13.5])   # the one real call starts at 10.0 and ends at 13.5
    cache = ev.ReplyCache(str(tmp_path / "cache.jsonl"), FakeLLM(), clock=lambda: next(clock))
    cache("Dose?", "m:free", "k")
    assert cache.reused_s == 0.0
    cache("Dose?", "m:free", "k")
    cache("Dose?", "m:free", "k")
    assert cache.reused_s == 7.0


def test_a_reused_answer_is_marked_and_its_latency_includes_the_original_call(tmp_path):
    clock = iter([0.0, 3.0])
    cache = ev.ReplyCache(str(tmp_path / "cache.jsonl"), FakeLLM(), clock=lambda: next(clock))

    def runner(system, question, ctx, model, api_key, threshold):
        """Send the same prompt for both systems, as rag_hybrid and rag_rerank do."""
        return {**answer(), "answer": cache(question, model, api_key)}

    records = ev.run_questions(QUESTIONS[:1], ["rag_hybrid", "rag_rerank"], ["m:free"], {}, "k", 0.6, False,
                               str(tmp_path / "records.jsonl"), runner, cache=cache)
    assert [(r["system"], r["answer"], r["reused"]) for r in records] == [
        ("rag_hybrid", "reply 1", False), ("rag_rerank", "reply 1", True)]
    assert records[0]["latency_s"] < 1.0 and 3.0 <= records[1]["latency_s"] < 4.0


def test_a_reused_judge_reply_does_not_count_in_the_latency(tmp_path, monkeypatch):
    clock = iter([0.0, 3.0])
    cache = ev.ReplyCache(str(tmp_path / "cache.jsonl"), lambda *a, **k: '{"supported": 1, "total": 2}',
                          clock=lambda: next(clock))
    monkeypatch.setattr(rag, "call_llm", cache)
    records = ev.run_questions(QUESTIONS[:1], ["rag_hybrid", "rag_rerank"], ["m:free"], {}, "k", 0.6, True,
                               str(tmp_path / "records.jsonl"), answer, cache=cache)
    assert [(r["faithfulness"], r["reused"]) for r in records] == [(0.5, False), (0.5, False)]
    assert records[1]["latency_s"] < 1.0


def test_run_sends_every_llm_call_through_a_reply_cache_in_results(tmp_path, monkeypatch):
    llm, seen = FakeLLM(), {}
    monkeypatch.setattr(rag, "call_llm", llm)
    monkeypatch.setattr(rag, "load_context", lambda: {"api_key": "k"})
    monkeypatch.setattr(ev, "RESULTS", str(tmp_path / "results"))

    def fake_run_questions(*args, cache):
        """Make the same request twice through rag.call_llm and keep the cache."""
        seen["replies"] = [rag.call_llm("Dose?", "m:free", "k"), rag.call_llm("Dose?", "m:free", "k")]
        seen["installed"] = rag.call_llm is cache
        return []

    monkeypatch.setattr(ev, "run_questions", fake_run_questions)
    with pytest.raises(SystemExit):   # no finished records
        ev.run(["rag_dense"], ["m:free"], False, 1)
    assert seen["replies"] == ["reply 1", "reply 1"] and len(llm.requests) == 1
    assert seen["installed"] and (tmp_path / "results" / "llm_cache.jsonl").exists()
    assert rag.call_llm is llm   # the run puts rag.call_llm back
