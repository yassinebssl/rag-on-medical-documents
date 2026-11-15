"""Rerank retrieved chunks with a cross-encoder.

Dense retrieval embeds the question and each chunk separately and compares the two
vectors. A cross-encoder reads the question and one passage together and returns one
relevance score, which is slower but more precise, so it is run only on the
candidates that retrieval already found.

Three cross-encoders are compared by `evaluate.py retrieval`:
- medcpt: NCBI's MedCPT cross-encoder, trained on PubMed search logs (Jin et al.,
  "MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs
  for zero-shot biomedical information retrieval", Bioinformatics, 2023,
  https://doi.org/10.1093/bioinformatics/btad651)
- bge: BAAI's multilingual bge-reranker-v2-m3, the reranker paired with BGE-M3
  (https://huggingface.co/BAAI/bge-reranker-v2-m3)
- minilm: a small general-domain reranker trained on MS MARCO
  (https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
"""

import math

GATE_RERANKER = "minilm"  # the cross-encoder of the refusal gate, chosen by evaluate.py retrieval
RERANKERS = {"medcpt": "ncbi/MedCPT-Cross-Encoder",
             "bge": "BAAI/bge-reranker-v2-m3",
             "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2"}


def load_reranker(name):
    """Load one cross-encoder by its short name. The first call downloads it from HuggingFace."""
    from sentence_transformers import CrossEncoder  # imported here so that importing rerank stays light

    return CrossEncoder(RERANKERS[name], max_length=512)


def rerank(question, hits, reranker, k=5):
    """Return the k hits the cross-encoder scores highest, each a copy with its score under "rerank"."""
    if not hits:
        return []
    scores = reranker.predict([(question, h["text"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: -pair[1])[:k]
    return [{**h, "rerank": float(s)} for h, s in ranked]


def gate_score(question, hits, reranker):
    """Return the sigmoid of the cross-encoder's best raw score (logit) over the hits, 0.0 for no hits.

    The sigmoid maps the logit to 0..1, and 0.5 (logit 0) is the model's own boundary
    between relevant and not relevant. It expects a model whose predict returns logits,
    as the MiniLM gate reranker does.
    """
    if not hits:
        return 0.0
    best = max(reranker.predict([(question, h["text"]) for h in hits]))
    return 1 / (1 + math.exp(-best))
