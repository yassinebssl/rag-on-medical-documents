"""Embed the chunks with BGE-M3 and store them in a persistent Chroma collection.

Cosine space is set on the collection, so Chroma returns a cosine distance.
Similarity is 1 - distance, which is what the refusal threshold is compared with.
"""
import os
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "chroma_db")
COLLECTION = "medical_docs"
EMBED_MODEL = "BAAI/bge-m3"


def load_embedder():
    """Load BGE-M3 once. The first call downloads about 2 GB from HuggingFace."""
    return SentenceTransformer(EMBED_MODEL)


def build_index(chunks, embedder, batch_size=16, rebuild=False, save_every=96):
    """Embed the chunks into the collection and return it.

    BGE-M3 on CPU takes about 3 s per chunk, so the build is resumable: chunks
    are saved every `save_every`, and a rerun only embeds ids that are missing.
    Ids that are no longer in the corpus are removed. `rebuild` starts from zero
    (needed when the chunking settings change, because ids would be reused).
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # nothing to delete
            pass
    coll = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    done = set(coll.get(include=[])["ids"])
    stale = sorted(done - {c["id"] for c in chunks})
    for start in range(0, len(stale), 500):
        coll.delete(ids=stale[start:start + 500])
    todo = [c for c in chunks if c["id"] not in done]
    print(f"{len(done) - len(stale)} chunks already indexed, {len(stale)} stale removed, "
          f"{len(todo)} to embed", flush=True)
    t0 = time.time()
    for start in range(0, len(todo), save_every):
        part = todo[start:start + save_every]
        texts = [c["text"] for c in part]
        vectors = embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        coll.add(ids=[c["id"] for c in part], embeddings=vectors.tolist(), documents=texts,
                 metadatas=[{"source": c["source"], "kind": c["kind"],
                             "page": c["page"] if c["page"] is not None else -1} for c in part])
        n = start + len(part)
        rate = (time.time() - t0) / n
        print(f"{n}/{len(todo)} embedded, {rate:.1f} s/chunk, "
              f"about {(len(todo) - n) * rate / 60:.0f} min left", flush=True)
    return coll


def open_index():
    """Open the existing collection without rebuilding it."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION)


def all_chunks(collection):
    """Return every stored chunk as a dict with id, text, source and page (None for no page)."""
    res = collection.get(include=["documents", "metadatas"])
    return [{"id": cid, "text": doc, "source": meta["source"],
             "page": meta["page"] if meta["page"] != -1 else None}
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]


if __name__ == "__main__":  # .venv/bin/python src/index.py [--rebuild]
    from ingest import build_chunks

    embedder = load_embedder()
    chunks = build_chunks(embedder.tokenizer)
    print(f"{len(chunks)} chunks from {len({c['source'] for c in chunks})} documents", flush=True)
    if len({c["id"] for c in chunks}) != len(chunks):
        raise SystemExit("duplicate chunk ids: two documents share a source name")
    coll = build_index(chunks, embedder, rebuild="--rebuild" in sys.argv)
    print(f"indexed {coll.count()} chunks into {DB_PATH}")
