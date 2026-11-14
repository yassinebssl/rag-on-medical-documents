"""Okapi BM25 keyword search over the chunks, written from scratch (standard library only).

It is the sparse baseline for the dense BGE-M3 retrieval, and the second ranking
fused with it in hybrid mode (rag.rrf).

Built following two build-your-own-x tutorials (listed at
https://github.com/codecrafters-io/build-your-own-x):
- https://stevenloria.com/tf-idf/ (term and document frequencies, idf)
- https://boyter.org/2010/08/build-vector-space-search-engine-python/ (tokenize, index, rank)
The scoring formula is Okapi BM25 from Robertson and Zaragoza, "The Probabilistic
Relevance Framework: BM25 and Beyond", 2009:
  idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
  score(q, d) = sum over t in q of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
"""
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase the text and split it on anything that is not a letter or a digit."""
    return TOKEN.findall(text.lower())


class BM25:
    """An inverted index over chunk dicts (id, text, source, page), searched with Okapi BM25."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        """Index the chunks: term -> {chunk position: term frequency}, plus chunk lengths."""
        self.chunks, self.k1, self.b = chunks, k1, b
        self.lengths = []
        self.postings = {}
        for i, chunk in enumerate(chunks):
            terms = tokenize(chunk["text"])
            self.lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, {})[i] = tf
        self.avgdl = sum(self.lengths) / len(chunks) if chunks else 0.0

    def idf(self, term):
        """Return ln(1 + (N - df + 0.5) / (df + 0.5)) for a term in the index."""
        df = len(self.postings[term])
        return math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))

    def search(self, question, k=5):
        """Return the k best chunks with a score above zero, as hits with similarity None and bm25."""
        scores = Counter()
        for term in set(tokenize(question)):
            if term not in self.postings:
                continue
            idf = self.idf(term)
            for i, tf in self.postings[term].items():
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        best = sorted((i for i in scores if scores[i] > 0), key=lambda i: -scores[i])[:k]
        return [{"id": self.chunks[i]["id"], "text": self.chunks[i]["text"],
                 "source": self.chunks[i]["source"], "page": self.chunks[i]["page"],
                 "similarity": None, "bm25": scores[i]} for i in best]
