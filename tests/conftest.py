"""Test fakes for the embedder and the Chroma collection: no model, no database."""
import numpy as np


class FakeEmbedder:
    """Stands in for BGE-M3: returns one zero vector per text."""

    def encode(self, texts, normalize_embeddings=True):
        """Return a (len(texts), 3) array of zeros."""
        return np.zeros((len(texts), 3))


class FakeCollection:
    """Stands in for a Chroma collection. rows: list of (id, text, metadata, distance), best first."""

    def __init__(self, rows):
        """Keep the rows in the order given."""
        self.rows = rows

    def count(self):
        """Return the number of rows."""
        return len(self.rows)

    def query(self, query_embeddings, n_results):
        """Return the first n_results rows in Chroma's nested query shape."""
        top = self.rows[:n_results]
        return {"ids": [[r[0] for r in top]], "documents": [[r[1] for r in top]],
                "metadatas": [[r[2] for r in top]], "distances": [[r[3] for r in top]]}

    def get(self, include=None):
        """Return every row in Chroma's flat get shape."""
        return {"ids": [r[0] for r in self.rows], "documents": [r[1] for r in self.rows],
                "metadatas": [r[2] for r in self.rows]}


class FakeReranker:
    """Stands in for a cross-encoder: returns the given raw scores (logits), one per pair, in order."""

    def __init__(self, scores):
        """Keep the scores and record each batch of pairs."""
        self.scores, self.calls = scores, []

    def predict(self, pairs):
        """Return the next len(pairs) scores: a later call continues where the last one stopped."""
        start = sum(len(batch) for batch in self.calls)
        self.calls.append(pairs)
        return self.scores[start:start + len(pairs)]
