"""Download the open medical corpus for the RAG project.

Sources (see vault/01-Project-Brief.md):
- WHO guidelines and publications, via the WHO IRIS public REST API.
- PubMed Central open-access articles, via the Europe PMC REST API.

The script works in two passes. First it collects candidate PDF links from
many topics (cheap JSON calls). Then it downloads candidates in parallel and
keeps the first successful files. Each download has a short timeout, so one
slow file cannot stall the run.

Files land in data/raw/who/ and data/raw/pubmed/. Only real PDFs under 15 MB
are kept. data/raw/manifest.json records source URL, size and title per file.

Usage:
    python3 scripts/fetch_corpus.py --who 20 --pubmed 30
"""
import argparse
import concurrent.futures as futures
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
UA = {"User-Agent": "rag-medical-project/1.0 (university student project)"}
MAX_BYTES = 15 * 1024 * 1024
DOWNLOAD_TIMEOUT = 60
WORKERS = 8

WHO_QUERIES = [
    "hypertension", "diabetes", "malaria", "tuberculosis", "HIV", "antenatal care",
    "sodium intake", "sugars intake", "physical activity", "depression",
    "antibiotic", "pneumonia children", "hepatitis", "cancer screening", "stroke",
    "immunization", "mental health", "breastfeeding", "cardiovascular risk",
    "snakebite", "cholera", "measles", "influenza", "nutrition adolescents",
    "road traffic injuries", "alcohol", "tobacco cessation", "obesity", "dengue",
    "rabies", "maternal health", "neonatal care",
]

PUBMED_QUERIES = [
    "hypertension treatment", "type 2 diabetes management", "paracetamol dosing",
    "antibiotic stewardship", "community acquired pneumonia", "sepsis management",
    "asthma management", "stroke rehabilitation", "major depressive disorder treatment",
    "chronic kidney disease", "heart failure treatment", "tuberculosis treatment",
    "anemia in pregnancy", "vaccination schedule", "hepatitis B treatment",
    "malaria treatment", "COPD exacerbation", "atrial fibrillation anticoagulation",
    "osteoarthritis management", "migraine treatment",
]


def get_json(url):
    """Fetch and decode a JSON document."""
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def download_pdf(url, path):
    """Download a PDF. Return its size in bytes, or None if it is not a usable PDF."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or not data.startswith(b"%PDF"):
        return None
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


def slug(text, limit=70):
    """Make a filesystem-safe name from a title."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:limit]


def norm_title(title):
    """Normalised title used to detect duplicates."""
    return re.sub(r"\W+", " ", title.lower()).strip()


def collect_who(needed):
    """Return up to `needed` unique WHO candidates as (title, pdf_url) pairs."""
    candidates, seen_titles, seen_uuids = [], set(), set()
    for query in WHO_QUERIES:
        if len(candidates) >= needed:
            break
        url = ("https://iris.who.int/server/api/discover/search/objects?"
               + urllib.parse.urlencode({"query": f"guideline {query}", "size": 5}))
        try:
            hits = get_json(url)["_embedded"]["searchResult"]["_embedded"]["objects"]
        except Exception as exc:
            print(f"[who] search failed for {query!r}: {exc}", flush=True)
            continue
        for hit in hits:
            item = hit["_embedded"]["indexableObject"]
            title = item.get("name", "untitled")
            if item["uuid"] in seen_uuids or norm_title(title) in seen_titles:
                continue
            seen_uuids.add(item["uuid"])
            seen_titles.add(norm_title(title))
            try:
                bundles = get_json(f"https://iris.who.int/server/api/core/items/{item['uuid']}/bundles")
                original = next(b for b in bundles["_embedded"]["bundles"] if b["name"] == "ORIGINAL")
                bits = get_json(original["_links"]["bitstreams"]["href"])["_embedded"]["bitstreams"]
                pdfs = [b for b in bits if b["name"].lower().endswith(".pdf")]
                if not pdfs:
                    continue
                candidates.append((title, pdfs[0]["_links"]["content"]["href"]))
            except Exception as exc:
                print(f"[who] skip {title[:60]!r}: {exc}", flush=True)
    print(f"[who] {len(candidates)} candidates collected", flush=True)
    return candidates


def collect_pubmed(needed):
    """Return up to `needed` unique open-access PMC candidates as (title, pmcid) pairs."""
    candidates, seen = [], set()
    for topic in PUBMED_QUERIES:
        if len(candidates) >= needed:
            break
        query = f"({topic}) AND OPEN_ACCESS:y AND HAS_PDF:y"
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
               + urllib.parse.urlencode({"query": query, "format": "json",
                                         "pageSize": 15, "resultType": "lite"}))
        try:
            results = get_json(url)["resultList"]["result"]
        except Exception as exc:
            print(f"[pubmed] search failed for {topic!r}: {exc}", flush=True)
            continue
        for rec in results:
            pmcid = rec.get("pmcid")
            if not pmcid or pmcid in seen:
                continue
            seen.add(pmcid)
            candidates.append((rec.get("title", "untitled"), pmcid))
    print(f"[pubmed] {len(candidates)} candidates collected", flush=True)
    return candidates


def fetch_parallel(jobs, out_dir, target, source, manifest):
    """Download `jobs` (title, url, stem) in parallel until `target` files are kept."""
    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {}
        queue = list(jobs)
        while queue and len(pending) < WORKERS:
            job = queue.pop(0)
            pending[pool.submit(download_pdf, job[1], os.path.join(out_dir, job[2]))] = job
        while pending:
            done, _ = futures.wait(list(pending), return_when=futures.FIRST_COMPLETED)
            for fut in done:
                title, url, name = pending.pop(fut)
                try:
                    size = fut.result()
                except Exception as exc:
                    size = None
                    print(f"[{source}] failed {name[:50]}: {exc}", flush=True)
                if size is not None and kept < target:
                    kept += 1
                    manifest.append({"source": source, "file": f"data/raw/{out_dir.split(os.sep)[-1]}/{name}",
                                     "title": title, "url": url, "bytes": size})
                    print(f"[{source}] {kept}/{target} {name} ({size // 1024} KB)", flush=True)
                elif size is None:
                    print(f"[{source}] not a usable PDF: {name[:50]}", flush=True)
                else:  # finished after the target was already met: drop the extra file
                    os.remove(os.path.join(out_dir, name))
                if kept < target and queue:
                    job = queue.pop(0)
                    pending[pool.submit(download_pdf, job[1], os.path.join(out_dir, job[2]))] = job
                elif kept >= target:
                    for other in pending:
                        other.cancel()
                    queue.clear()
    return kept


def pubmed_fulltext(pmcid):
    """Return the plain-text body of an open-access article from Europe PMC XML, or None."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        root = ET.fromstring(resp.read())
    paras = ["".join(p.itertext()).strip() for p in root.iter("p")]
    text = "\n\n".join(p for p in paras if p)
    return text if len(text) >= 2000 else None


def fetch_pubmed_text(pubs, out_dir, target, manifest):
    """Save open-access article full text as .txt files, in candidate order.

    Europe PMC serves article PDFs with 403 to scripts, and the PMC PDF pages
    return HTML. The article XML is reliable, so the plain text is stored instead.
    Returns the number of articles kept.
    """
    os.makedirs(out_dir, exist_ok=True)
    texts = {}
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(pubmed_fulltext, pmcid): (title, pmcid) for title, pmcid in pubs}
        for fut in futures.as_completed(futs):
            title, pmcid = futs[fut]
            try:
                text = fut.result()
            except Exception as exc:
                print(f"[pubmed] failed {pmcid}: {exc}", flush=True)
                continue
            if text:
                texts[pmcid] = (title, text)
    kept = 0
    for title, pmcid in pubs:
        if kept >= target:
            break
        if pmcid not in texts:
            continue
        title, text = texts[pmcid]
        name = f"PMC_{pmcid}_{slug(title, 50)}.txt"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
        kept += 1
        manifest.append({"source": "PubMed Central OA (full text XML)", "file": f"data/raw/pubmed/{name}",
                         "title": title, "url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                         "bytes": len(text.encode("utf-8"))})
        print(f"[pubmed] {kept}/{target} {name} ({len(text) // 1024} KB text)", flush=True)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--who", type=int, default=20, help="number of WHO PDFs")
    parser.add_argument("--pubmed", type=int, default=30, help="number of PubMed OA PDFs")
    args = parser.parse_args()

    os.makedirs(RAW, exist_ok=True)
    manifest = []

    who = collect_who(args.who * 2)
    who_jobs = [(t, u, f"WHO_{i + 1:02d}_{slug(t)}.pdf") for i, (t, u) in enumerate(who)]
    who_kept = fetch_parallel(who_jobs, os.path.join(RAW, "who"), args.who, "who", manifest)

    pubs = collect_pubmed(args.pubmed * 2)
    pub_kept = fetch_pubmed_text(pubs, os.path.join(RAW, "pubmed"), args.pubmed, manifest)

    with open(os.path.join(RAW, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    total_mb = sum(m["bytes"] for m in manifest) / (1024 * 1024)
    print(f"done: {who_kept} WHO + {pub_kept} PubMed PDFs, {total_mb:.1f} MB total", flush=True)


if __name__ == "__main__":
    main()
