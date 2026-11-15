"""Tests for cross-encoder reranking (a fake cross-encoder, no model download)."""
import math

import rerank


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


def test_gate_score_is_the_sigmoid_of_the_best_logit():
    from conftest import FakeReranker
    hits = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert abs(rerank.gate_score("q", hits, FakeReranker([-1.0, 3.0, 0.5])) - 1 / (1 + math.exp(-3.0))) < 1e-9
    assert rerank.gate_score("q", [], FakeReranker([])) == 0.0
