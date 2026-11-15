"""Tests for cross-encoder reranking (a fake cross-encoder, no model download)."""
import math

import rerank
from conftest import FakeReranker


class FakeCrossEncoder:
    """Score a (question, passage) pair by the number of question words in the passage."""

    def __init__(self):
        """Record every batch of pairs predict is given."""
        self.calls = []

    def predict(self, pairs):
        """Return one score per pair."""
        self.calls.append(pairs)
        return [float(sum(w in passage.split() for w in question.split())) for question, passage in pairs]


HITS = [{"id": "a", "text": "bed nets prevent malaria", "similarity": 0.8},
        {"id": "b", "text": "artesunate treats severe malaria", "similarity": 0.7},
        {"id": "c", "text": "salt raises blood pressure", "similarity": 0.6}]


def test_rerank_orders_by_cross_encoder_score_and_keeps_k():
    out = rerank.rerank("artesunate severe malaria", HITS, FakeCrossEncoder(), k=2)
    assert [(h["id"], h["rerank"]) for h in out] == [("b", 3.0), ("a", 1.0)]
    assert out[0]["similarity"] == 0.7 and "rerank" not in HITS[1]


def test_rerank_scores_all_hits_in_one_batch():
    ce = FakeCrossEncoder()
    rerank.rerank("malaria", HITS, ce)
    assert ce.calls == [[("malaria", h["text"]) for h in HITS]]


def test_rerank_of_no_hits_does_not_call_the_model():
    ce = FakeCrossEncoder()
    assert rerank.rerank("q", [], ce) == [] and ce.calls == []


def test_every_reranker_name_maps_to_a_model_id():
    assert set(rerank.RERANKERS) == {"medcpt", "bge", "minilm"}
    assert rerank.RERANKERS["medcpt"] == "ncbi/MedCPT-Cross-Encoder"


def sigmoid(logit):
    """Map a raw score to 0..1."""
    return 1 / (1 + math.exp(-logit))


ABC = [{"text": "a"}, {"text": "b"}, {"text": "c"}]


def test_gate_score_is_the_sigmoid_of_the_best_logit():
    assert abs(rerank.gate_score("q", ABC, FakeReranker([-1.0, 3.0, 0.5])) - sigmoid(3.0)) < 1e-9
    assert rerank.gate_score("q", [], FakeReranker([])) == 0.0


def test_gate_score_scores_one_passage_at_a_time_in_retrieval_order():
    ce = FakeReranker([-1.0, 3.0, 0.5])
    rerank.gate_score("q", ABC, ce)
    assert ce.calls == [[("q", "a")], [("q", "b")], [("q", "c")]]


def test_gate_score_stops_at_the_first_passage_that_reaches_stop_at():
    ce = FakeReranker([-1.0, 0.5, 3.0])
    assert abs(rerank.gate_score("q", ABC, ce, stop_at=0.5) - sigmoid(0.5)) < 1e-9
    assert ce.calls == [[("q", "a")], [("q", "b")]]   # c, the best passage, is never scored


def test_gate_score_with_stop_at_is_the_best_score_when_no_passage_reaches_it():
    ce = FakeReranker([-1.0, -3.0, -0.5])
    assert abs(rerank.gate_score("q", ABC, ce, stop_at=0.5) - sigmoid(-0.5)) < 1e-9
    assert len(ce.calls) == 3


def test_stopping_early_never_changes_the_gate_decision():
    for logits in ([-1.0, 3.0, 0.5], [-4.0, -2.0, -0.1], [0.0, -1.0, -1.0], [-1.0, -1.0, 0.0], [2.0, 9.0, 9.0]):
        for threshold in (0.1, 0.5, 0.9):
            full = rerank.gate_score("q", ABC, FakeReranker(logits))
            early = rerank.gate_score("q", ABC, FakeReranker(logits), stop_at=threshold)
            assert (early < threshold) == (full < threshold)
