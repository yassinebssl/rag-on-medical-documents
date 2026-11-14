"""Tests for the from-scratch Okapi BM25: tokenizer, score formula, ranking."""
from bm25 import BM25, tokenize


def chunk(i, text):
    """Return a chunk dict with id str(i), source f"s{i}" and page i."""
    return {"id": str(i), "text": text, "source": f"s{i}", "page": i}


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("HIV-1 Post-exposure, 500mg!") == ["hiv", "1", "post", "exposure", "500mg"]


def test_score_matches_the_okapi_formula():
    # N=3, df(c)=1, idf=ln(1+2.5/1.5)=0.98083; tf=2, dl=3, avgdl=2 -> 0.98083*5/4.0625
    hits = BM25([chunk(0, "a b"), chunk(1, "a c c"), chunk(2, "d")]).search("c")
    assert [h["id"] for h in hits] == ["1"]
    assert abs(hits[0]["bm25"] - 1.2072) < 1e-3


def test_search_ranks_the_chunk_with_all_terms_first_and_limits_k():
    index = BM25([chunk(0, "malaria is treated with artemether"),
                  chunk(1, "malaria prevention with bed nets"),
                  chunk(2, "hypertension and sodium intake")])
    hits = index.search("artemether malaria", k=2)
    assert [h["id"] for h in hits] == ["0", "1"]
    assert hits[0] == {"id": "0", "text": "malaria is treated with artemether", "source": "s0",
                       "page": 0, "similarity": None, "bm25": hits[0]["bm25"]}


def test_unknown_terms_and_empty_corpus_return_nothing():
    assert BM25([chunk(0, "a b")]).search("zzz") == []
    assert BM25([]).search("a") == []
