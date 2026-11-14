"""Schema checks for the evaluation question set in data/eval/questions.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "data", "eval", "questions.json"), encoding="utf-8") as fh:
    qs = json.load(fh)


def test_eval_set_has_20_answerable_and_10_out_of_corpus():
    assert [q["id"] for q in qs] == [f"A{i:02d}" for i in range(1, 21)] + [f"O{i:02d}" for i in range(1, 11)]
    assert all(set(q) == {"id", "question", "answerable", "ground_truth", "source", "page", "evidence"} for q in qs)
    assert all(q["answerable"] is (q["id"][0] == "A") for q in qs)
    assert all(q["question"].strip().endswith("?") for q in qs)
    assert len({q["question"] for q in qs}) == 30


def test_answerable_questions_carry_evidence_and_the_others_do_not():
    for q in qs:
        if q["answerable"]:
            assert q["ground_truth"] and q["source"] and 40 <= len(q["evidence"]) <= 300
        else:
            assert q["ground_truth"] is None and q["source"] is None and q["page"] is None and q["evidence"] is None
