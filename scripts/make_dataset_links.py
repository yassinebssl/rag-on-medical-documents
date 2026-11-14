"""Write docs/dataset_links.md, the list of source links of the corpus, from data/raw/manifest.json.

Run from the repo root: .venv/bin/python scripts/make_dataset_links.py
"""
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(manifest):
    """Return the Markdown document: a heading, the count per source and one table row per document."""
    counts = Counter(m["source"] for m in manifest)
    per_source = ", ".join(f"{n} from {source}" for source, n in counts.items())
    lines = ["# Dataset links", "", f"The corpus has {len(manifest)} documents: {per_source}.", "",
             "| # | Source | Title | File | Size | Link |", "|---|---|---|---|---|---|"]
    for i, m in enumerate(manifest, 1):
        title = m["title"].replace("|", "\\|")
        lines.append(f"| {i} | {m['source']} | {title} | {os.path.basename(m['file'])} "
                     f"| {m['bytes'] // 1024} KB | {m['url']} |")
    return "\n".join(lines) + "\n"


def main():
    """Read the manifest and write docs/dataset_links.md."""
    with open(os.path.join(ROOT, "data", "raw", "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(ROOT, "docs", "dataset_links.md"), "w", encoding="utf-8") as fh:
        fh.write(render(manifest))


if __name__ == "__main__":
    main()
