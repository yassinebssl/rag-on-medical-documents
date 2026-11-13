"""Load the raw corpus and split it into overlapping chunks.

WHO guidelines are PDFs, read page by page with pypdf, so every chunk keeps its
page number for citations. PubMed articles are plain-text files (full text
from Europe PMC), so their citation is the article title.
"""
import json
import os

from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 102  # 20% overlap, as in the brief


def load_documents(raw_dir=RAW):
    """Return a list of page-level records: text, source name, page number, kind."""
    docs = []
    who_dir = os.path.join(raw_dir, "who")
    for name in sorted(os.listdir(who_dir)):
        reader = PdfReader(os.path.join(who_dir, name))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if len(text) > 50:  # skip blank or scanned pages with no text layer
                docs.append({"text": text, "source": name, "page": i + 1, "kind": "WHO"})
    pub_dir = os.path.join(raw_dir, "pubmed")
    manifest = {}
    manifest_path = os.path.join(raw_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = {os.path.basename(m["file"]): m["title"] for m in json.load(fh)}
    for name in sorted(os.listdir(pub_dir)):
        with open(os.path.join(pub_dir, name), encoding="utf-8") as fh:
            text = fh.read().strip()
        title = manifest.get(name, name)
        docs.append({"text": text, "source": title, "page": None, "kind": "PubMed"})
    return docs


def chunk_document(doc, tokenizer):
    """Split one document into overlapping token windows, keeping its citation fields."""
    ids = tokenizer.encode(doc["text"], add_special_tokens=False)
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    chunks = []
    for start in range(0, max(len(ids), 1), step):
        window = ids[start:start + CHUNK_TOKENS]
        if not window:
            break
        chunks.append({"text": tokenizer.decode(window), "source": doc["source"],
                       "page": doc["page"], "kind": doc["kind"], "start_token": start})
        if start + CHUNK_TOKENS >= len(ids):
            break
    return chunks


def build_chunks(tokenizer, raw_dir=RAW):
    """Load every document and return all chunks, with a stable id per chunk."""
    chunks = []
    for doc in load_documents(raw_dir):
        for c in chunk_document(doc, tokenizer):
            c["id"] = f"{c['source']}|{c['page']}|{c['start_token']}"
            chunks.append(c)
    return chunks
