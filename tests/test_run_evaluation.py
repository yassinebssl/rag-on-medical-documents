"""Tests for scripts/run_evaluation.py: which answers of the plan are still missing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_evaluation  # noqa: E402


def test_missing_lists_the_planned_answers_that_have_no_record():
    steps = [("rag_rerank,no_rag", "m1"), ("rag_rerank", "m2")]
    records = [{"system": "rag_rerank", "model": "m1", "id": "A01"},
               {"system": "rag_rerank", "model": "m2", "id": "A02"},
               {"system": "rag_dense", "model": "m1", "id": "A01"}]  # not in the plan
    assert run_evaluation.missing(records, ["A01", "A02"], steps) == [
        ("rag_rerank", "m1", "A02"), ("no_rag", "m1", "A01"), ("no_rag", "m1", "A02"), ("rag_rerank", "m2", "A01")]
    assert len(run_evaluation.missing([], ["A01", "A02"], steps)) == 6
