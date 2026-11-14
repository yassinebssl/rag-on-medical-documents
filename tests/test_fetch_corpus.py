"""Tests for fetch_corpus.py --from-manifest: fakes for the downloads, files in tmp_path only."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import fetch_corpus as fc  # noqa: E402

PDF_URL = "https://iris.who.int/server/api/core/bitstreams/1234/content"
TXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13687808/fullTextXML"


def test_from_manifest_writes_the_recorded_files_and_skips_failures(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    manifest = [{"source": "WHO IRIS", "file": "data/raw/who/WHO_07_x.pdf", "title": "W", "url": PDF_URL, "bytes": 1},
                {"source": "PubMed Central OA (full text XML)", "file": "data/raw/pubmed/PMC_PMC13687808_y.txt",
                 "title": "P", "url": TXT_URL, "bytes": 1}]
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = (raw / "manifest.json").read_text(encoding="utf-8")
    monkeypatch.setattr(fc, "ROOT", str(tmp_path))
    monkeypatch.setattr(fc, "RAW", str(raw))
    calls = []

    def download_pdf(url, path):
        """Record the call and fail like a dropped connection."""
        calls.append(("pdf", url, path))
        raise OSError("connection reset")

    def pubmed_fulltext(pmcid):
        """Record the call and return the article text."""
        calls.append(("txt", pmcid))
        return "article text"

    def no_search(*args):
        """Fail if main() searches instead of reading the manifest."""
        raise AssertionError("searched by topic")

    monkeypatch.setattr(fc, "download_pdf", download_pdf)
    monkeypatch.setattr(fc, "pubmed_fulltext", pubmed_fulltext)
    monkeypatch.setattr(fc, "collect_who", no_search)
    monkeypatch.setattr(fc, "collect_pubmed", no_search)
    monkeypatch.setattr(sys, "argv", ["fetch_corpus.py", "--from-manifest"])
    fc.main()
    assert calls == [("pdf", PDF_URL, str(tmp_path / "data/raw/who/WHO_07_x.pdf")), ("txt", "PMC13687808")]
    assert (tmp_path / "data/raw/pubmed/PMC_PMC13687808_y.txt").read_text(encoding="utf-8") == "article text"
    assert not (tmp_path / "data/raw/who/WHO_07_x.pdf").exists()
    assert (raw / "manifest.json").read_text(encoding="utf-8") == before
    out = capsys.readouterr().out
    assert "data/raw/who/WHO_07_x.pdf" in out and "connection reset" in out
    assert "1/2" in out
