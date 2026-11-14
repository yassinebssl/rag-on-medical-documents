"""Check that every answerable eval question quotes its source verbatim.

Run from the repo root: .venv/bin/python scripts/check_eval_set.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ingest import load_documents  # noqa: E402


def normalize(text):
    """Collapse every run of whitespace to one space and lowercase."""
    return " ".join(text.split()).lower()


def main():
    """Print one line per missing quote and exit 1 unless all 20 are found."""
    with open(os.path.join(ROOT, "data", "eval", "questions.json"), encoding="utf-8") as fh:
        answerable = [q for q in json.load(fh) if q["answerable"]]
    texts = {}
    for d in load_documents():
        texts.setdefault(d["source"], {})[d["page"]] = normalize(d["text"])
    found = 0
    for q in answerable:
        if q["source"] not in texts:
            print(f"{q['id']}: unknown source {q['source']!r}")
        elif normalize(q["evidence"]) not in texts[q["source"]].get(q["page"], ""):
            print(f"{q['id']}: evidence not found in {q['source']!r} page {q['page']}")
        else:
            found += 1
    print(f"{found}/{len(answerable)} evidence quotes found")
    sys.exit(0 if found == 20 else 1)


if __name__ == "__main__":
    main()
