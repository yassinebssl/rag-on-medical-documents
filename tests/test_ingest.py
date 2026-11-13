"""Characterization tests that pin the chunking the 3-hour index was built with."""
from ingest import chunk_document


class FakeTokenizer:
    """One token per word: token ids are 0..n-1, decoded back as numbers."""

    def encode(self, text, add_special_tokens=False):
        """Return list(range(n)) for a text of n words."""
        return list(range(len(text.split())))

    def decode(self, ids):
        """Return the ids joined by spaces."""
        return " ".join(map(str, ids))


def test_chunks_are_512_tokens_with_102_overlap():
    doc = {"text": " ".join(["w"] * 1000), "source": "a.pdf", "page": 3, "kind": "WHO"}
    chunks = chunk_document(doc, FakeTokenizer())
    assert [c["start_token"] for c in chunks] == [0, 410, 820]
    assert len(chunks[0]["text"].split()) == 512
    assert chunks[1]["text"].split()[0] == "410"
    assert all(c["source"] == "a.pdf" and c["page"] == 3 for c in chunks)


def test_short_document_is_one_chunk():
    doc = {"text": "w w w", "source": "t", "page": None, "kind": "PubMed"}
    assert len(chunk_document(doc, FakeTokenizer())) == 1
