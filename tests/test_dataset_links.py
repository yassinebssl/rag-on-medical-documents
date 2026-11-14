"""Check the Markdown table that scripts/make_dataset_links.py builds from the manifest."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from make_dataset_links import render  # noqa: E402


def test_render_lists_every_document_with_its_link():
    md = render([{"source": "WHO IRIS", "file": "data/raw/who/WHO_01_x.pdf", "title": "Hypertension | guide", "url": "https://iris.who.int/x", "bytes": 2048},
                 {"source": "PubMed Central OA (full text XML)", "file": "data/raw/pubmed/PMC_1.txt", "title": "Paracetamol.", "url": "https://www.ebi.ac.uk/y", "bytes": 1024}])
    assert "| 1 | WHO IRIS | Hypertension \\| guide | WHO_01_x.pdf | 2 KB | https://iris.who.int/x |" in md
    assert "| 2 | PubMed Central OA (full text XML) | Paracetamol. | PMC_1.txt | 1 KB | https://www.ebi.ac.uk/y |" in md
    assert md.startswith("# Dataset links\n")
