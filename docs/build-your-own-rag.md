# Build your own RAG on medical documents

In this tutorial you rebuild a question answering system over 20 WHO guidelines and 30 open-access PubMed Central articles. It finds the passages that best match a question, gives them to a free language model, and keeps the answer only if it cites those passages. When the passages are too far from the question, or the model does not cite them, it replies with one fixed sentence:

```text
I cannot answer this question from the provided documents.
```

This is retrieval-augmented generation, or RAG. A model alone answers from its training, and a wrong answer reads as confidently as a right one. In RAG you first search your documents for the passages closest to the question (retrieval), then tell the model to answer only from them (generation). You build it in data-flow order, and each defence opens with the failure it prevents.

## Before you start

**What this text covers.** Each step shows the core of one piece, names its file and function, and explains why it is built that way; it does not print every line. Work with the repository open: you write the six files of `src/` yourself and open the named function when an excerpt is cut (a `# ...` line marks a cut). The project's tests are your specification.

**Three rules hold the whole way through.**

- This is a student project. It is not medical advice, and every answer must be checked with a clinician.
- Questions and the retrieved passages are sent to OpenRouter, an online service. Never type anything that identifies a patient.
- Only free models are allowed. The code refuses any other model.

**The numbers you will meet**, all taken from the code:

| Constant | Value | Where |
|---|---|---|
| `CHUNK_TOKENS` | 512 | `src/ingest.py` |
| `OVERLAP_TOKENS` | 102 | `src/ingest.py` |
| `EMBED_MODEL` | `BAAI/bge-m3` | `src/index.py` |
| `COLLECTION` | `medical_docs` | `src/index.py` |
| `save_every` | 96 | `build_index` in `src/index.py` |
| `TOP_K` | 5 | `src/rag.py` |
| `CANDIDATES` | 20 | `src/rag.py` |
| `SIM_THRESHOLD` | 0.6 | `src/rag.py` |
| Fusion constant `k` (Step 10) | 60 | `rrf` in `src/rag.py` |
| BM25 `k1`, `b` (Step 9) | 1.5, 0.75 | `BM25.__init__` in `src/bm25.py` |

## Step 0: Set up the project

You need Linux and Python 3.12. Create `src/`, `scripts/`, `tests/`, `data/raw/` and `data/eval/`, and copy from the repository:

- `requirements.txt`, which pins every library, with the CPU build of PyTorch and `cryptography` (pypdf needs it to open encrypted PDFs);
- `.env.example`, the template for your secrets, and `pytest.ini`, which puts `src/` on the tests' import path;
- the whole of `tests/` and `scripts/`;
- `data/raw/manifest.json` (the 50 documents) and `data/eval/questions.json` (the evaluation questions). Under `data/`, these are the only two files git tracks.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put a free OpenRouter key after `OPENROUTER_API_KEY=` in `.env`. In `.gitignore`, list `.env`, `chroma_db/` and `data/raw/*` followed by `!data/raw/manifest.json`, so that the key, the index and the downloaded documents are never committed.

The tests use fakes for the embedding model, the database and the language model, so every step can be checked offline.

## Step 1: Get the corpus

The 20 WHO guidelines are PDF files from WHO IRIS; `download_pdf` in `scripts/fetch_corpus.py` keeps a file only if it starts with `%PDF` and is at most 15 MB. The 30 PubMed Central articles are stored as plain text, because, as the docstring of `fetch_pubmed_text` says, Europe PMC serves article PDFs to scripts with an HTTP 403 while its full-text XML is reliable. `pubmed_fulltext` keeps the text of every `<p>` element and rejects a body under 2,000 characters.

`--who 20 --pubmed 30` searches both APIs by topic and rewrites the manifest, so it may give other documents. `--from-manifest` runs `fetch_from_manifest`, which downloads every manifest entry under its recorded name and leaves the manifest unchanged. Use that one:

```bash
.venv/bin/python scripts/fetch_corpus.py --from-manifest
.venv/bin/python -m pytest -q tests/test_fetch_corpus.py
```

The download prints `ok data/raw/...` per file, `failed ...` or `not usable: ...` per failure, and last `done: N/50 files written`. A PubMed text can differ from the project's if Europe PMC has updated the article; Step 2 gives a partial check. `scripts/make_dataset_links.py` writes `docs/dataset_links.md`, the source link of every document, from the manifest.

## Step 2: Load the documents and cut them into chunks

A guideline can run to hundreds of pages, and the prompt holds five passages. So you cut every document into **chunks**: short windows of text, each retrieved as one unit and keeping the fields needed to cite it. Chunk length is counted in **tokens**, the word pieces a model's tokenizer produces.

`load_documents` in `src/ingest.py` reads each WHO PDF page by page with pypdf and skips pages of 50 characters or fewer, so every WHO chunk keeps a page number for its citation. A PubMed article becomes one record with `"page": None`, cited by its title from the manifest. Two PDFs, `WHO_09` and `WHO_25`, are encrypted, hence `cryptography`.

`chunk_document` in `src/ingest.py` slides a window of `CHUNK_TOKENS = 512` tokens with `OVERLAP_TOKENS = 102` tokens of overlap (20%, the brief's value), so the window moves by 410 tokens. A sentence of up to 102 tokens that is cut at the end of one window appears whole at the start of the next:

```python
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
```

The tokenizer is passed in: in Step 3 it is the embedding model's own, so a window is 512 of that model's tokens; in `tests/test_ingest.py` it is a fake with one token per word. `build_chunks` in `src/ingest.py` runs `chunk_document` over every document and gives each chunk an id:

```python
            c["id"] = f"{c['source']}|{c['page']}|{c['start_token']}"
```

For the same settings the id is stable across runs, and Step 3 depends on that.

```bash
.venv/bin/python -m pytest -q tests/test_ingest.py
.venv/bin/python scripts/check_eval_set.py
```

`scripts/check_eval_set.py` calls your `load_documents` and looks for the verbatim evidence quote of each of the 20 answerable questions on its recorded page, after lowercasing and collapsing whitespace. It prints one line per missing quote, then `<found>/20 evidence quotes found`, and exits with status 1 unless all 20 are found. On the corpus recorded in the manifest, on 2026-11-14, it printed `20/20 evidence quotes found`, and `load_documents()` returned 1530 records, 1500 of them WHO pages.

The check is narrow: the quotes come from 19 of the 50 documents, only 8 of them PubMed articles, so a change elsewhere goes unnoticed.

**Exercise.** With the fake tokenizer of `tests/test_ingest.py`, predict the chunks of a 922-word and a 923-word record, then check. (Two, and three: the third holds 103 tokens.)

## Step 3: Embed the chunks and build the index

An **embedding** is a list of numbers a model computes for a text, so that texts with similar meaning get vectors pointing in similar directions. The project uses `BAAI/bge-m3` through sentence-transformers; the architecture note `vault/04-Architecture.md` calls it the brief's choice, multilingual and strong on medical text.

**Cosine similarity** measures how close two directions are: the dot product of two vectors divided by the product of their lengths, 1 for the same direction. For vectors of length 1 it reduces to the dot product. For example, (3, 4) and (4, 3) normalised are (0.6, 0.8) and (0.8, 0.6), and their cosine similarity is 0.48 + 0.48 = 0.96.

The vectors go into Chroma, a vector database, in a collection created with `"hnsw:space": "cosine"`. HNSW (Hierarchical Navigable Small World) is the graph Chroma searches to find near vectors without comparing against every stored one; this setting makes it report cosine distance, 1 minus the similarity, so lower means closer. The architecture note warns that a threshold written against raw distances points the wrong way, so Step 4 converts back with `1 - distance`.

`load_embedder` in `src/index.py` returns `SentenceTransformer(EMBED_MODEL)`. This is `build_index` in the same file, shortened:

```python
def build_index(chunks, embedder, batch_size=16, rebuild=False, save_every=96):
    # ...
    coll = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    done = set(coll.get(include=[])["ids"])
    # ...
    todo = [c for c in chunks if c["id"] not in done]
    # ...
    for start in range(0, len(todo), save_every):
        part = todo[start:start + save_every]
        texts = [c["text"] for c in part]
        vectors = embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        coll.add(ids=[c["id"] for c in part], embeddings=vectors.tolist(), documents=texts,
                 metadatas=[{"source": c["source"], "kind": c["kind"],
                             "page": c["page"] if c["page"] is not None else -1} for c in part])
        # ...
    return coll
```

The failure this prevents is losing hours of work to one crash: the docstring of `build_index` puts BGE-M3 on a CPU at about 3 s per chunk, and the README estimates about 3 hours on a laptop CPU, after a 2 GB model download. So the build saves every 96 chunks and, on start, asks Chroma which ids it holds. Because the chunk id is stable, "already stored" is a set difference, and ids that match no current chunk are deleted as stale. After a change to `CHUNK_TOKENS` or `OVERLAP_TOKENS`, run with `--rebuild`, because the same ids would then name different text. A missing PubMed page is stored as `-1`.

Two readers follow. `open_index` opens the existing collection, and `all_chunks(collection)` returns every stored chunk as a dict with id, text, source and page (`-1` back to `None`). The command-line block loads the embedder, calls `build_chunks(embedder.tokenizer)`, stops with `duplicate chunk ids: two documents share a source name` if two ids collide, and calls `build_index`. Start the build and keep working while it runs:

```bash
.venv/bin/python src/index.py
```

Its first line is `<N> chunks from <M> documents`. Then `build_index` prints `<a> chunks already indexed, <b> stale removed, <c> to embed`, one line per saved batch (`96/<c> embedded, <r> s/chunk, about <t> min left`), and last `indexed <N> chunks into .../chroma_db`. If it stops, run the same command again. No unit test covers `build_index`; what protects the ids is `tests/test_ingest.py`, which pins the chunking.

## Step 4: Retrieve the closest chunks

Start `src/rag.py` with its constants:

```python
REFUSAL = "I cannot answer this question from the provided documents."
TOP_K = 5
CANDIDATES = 20  # chunks taken from each ranking before fusion in hybrid mode
SIM_THRESHOLD = 0.6  # the brief's value; calibrate on the eval set before reporting
# ...
FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    # ...
]
DEFAULT_MODEL = FREE_MODELS[0]
```

`FREE_MODELS` lists five. Retrieval embeds the question the same way and asks Chroma for the `TOP_K` nearest chunks. This is `retrieve` in `src/rag.py`:

```python
def retrieve(question, embedder, collection, k=TOP_K):
    """Return the top-k chunks as dicts with id, text, source, page and similarity."""
    qvec = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[qvec], n_results=k)
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        hits.append({"id": cid, "text": doc, "source": meta["source"],
                     "page": meta["page"] if meta["page"] != -1 else None,
                     "similarity": 1.0 - dist})
    return hits
```

`source_label` in the same file builds the label a reader sees in a citation: `file, p. N` for a WHO page, the title alone for a PubMed article.

```bash
.venv/bin/python -m pytest -q tests/test_rag.py -k "retrieve or source_label"
```

## Step 5: Refuse when nothing is close enough

Retrieval always returns five chunks, even for "Who painted the Mona Lisa?" (question O09). Handed five unrelated passages, a model can still answer from memory, as confidently as from evidence.

The defence is a **similarity gate**: when even the best chunk is far from the question, return the refusal sentence without calling the model. `answer_question` in `src/rag.py` is the whole pipeline. Write it now; the `bm25` branch belongs to Step 10, and `generate_answer` comes in Step 7:

```python
def answer_question(question, embedder, collection, model, api_key, threshold=SIM_THRESHOLD,
                    bm25=None):
    """Retrieve (fused with BM25 if given), refuse below the threshold, else generate_answer."""
    if bm25 is None:
        hits = retrieve(question, embedder, collection)
        best = hits[0]["similarity"] if hits else 0.0
    else:
        dense = retrieve(question, embedder, collection, k=CANDIDATES)
        best = dense[0]["similarity"] if dense else 0.0
        hits = rrf([dense, bm25.search(question, CANDIDATES)])[:TOP_K]
    if best < threshold:
        res = {"answer": REFUSAL, "refused": True, "reason": "below_threshold", "hits": hits, "cited": []}
    else:
        res = generate_answer(question, hits, model, api_key)
    return {**res, "best_similarity": best}
```

The refusal is always the same string so that code can test for it: the prompt asks the model for this exact sentence, and `generate_answer` and the `no_rag` baseline look for it in the reply. A similarity of exactly 0.6 passes, because the test is `best < threshold`. The value 0.6 comes from the brief and is not measured; Step 12 calibrates it.

Its test in `tests/test_rag.py` uses a distance of 0.45 (similarity 0.55) and replaces `rag.call_llm` with a function that divides by zero, so it passes only if the model is never called. It needs `call_llm` to exist, so you run it at the end of Step 6.

## Step 6: Prompt a free model

`PROMPT` in `src/rag.py` asks for a citation after every claim, gives the exact refusal sentence, and adds the architecture note's two guardrails: no diagnosis, and doses quoted with a reminder to check with a clinician.

```python
PROMPT = """You are a medical assistant. Answer ONLY using the numbered passages below.
After every claim, cite the passage it comes from by its number in square
brackets, like [1] or [2][4]. Use only the passage numbers given.
If the passages do not contain the answer, say exactly:
"I cannot answer this question from the provided documents."
This is not medical advice. Do not diagnose. For doses, quote the passage and
tell the user to check with a clinician.

Passages:
{context}

Question: {question}
Answer (with citations):"""
```

From this step on, your code sends the question and five passages to OpenRouter. The README states that the corpus holds no patient data, so patient data can only leave through what you type: never type anything that identifies a patient. `load_env` in `src/rag.py` reads the `KEY=VALUE` lines of `.env` into a dict without printing them.

Only free models are used: the docstring of `src/rag.py` calls this a project rule, and `vault/13-Decisions-Log.md` records it (every model id must end in `:free`). It is enforced in code because the model is only a string in the request body, so a typo in `.env` or on the command line would send every question to a paid model. `check_free` in `src/rag.py` is the stop:

```python
def check_free(model):
    """Raise unless the model is one of the free OpenRouter models."""
    if not (model.endswith(":free") or model == "openrouter/free"):
        raise ValueError(f"{model!r} is not a free model; the project uses free models only")
```

`call_llm` in `src/rag.py` sends one chat request with the standard library (`"max_tokens": 600, "temperature": 0`); its first statement is `check_free(model)`, so no paid id reaches the network. Free models are rate-limited, so it makes up to four attempts, retrying HTTP 429, any 5xx and empty replies after pauses of 10, 20 and 40 seconds (`time.sleep(10 * 2 ** attempt)`). Any other HTTP error, such as 401 for a bad key, is raised at once, and if every reply is empty it raises `RuntimeError`.

```bash
.venv/bin/python -m pytest -q tests/test_rag.py -k "call_llm or check_free or (below_threshold and not hybrid)"
```

The `call_llm` tests replace `urlopen` and `time.sleep`, so nothing is sent.

## Step 7: Check the citations

The gate compares the question with the passages and says nothing about the reply. A model can answer without citing anything, or cite `[7]` when it had five passages, and both look grounded to a reader.

A citation is a passage number in square brackets, matched by the `CITATION` pattern in `src/rag.py`. `format_context` numbers the passages `[1]` to `[5]` for the prompt, each under its source label:

```python
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2] or [1, 4]
# ...
def format_context(hits):
    # ...
    return "\n\n".join(f"[{i}] {source_label(h)}\n{CITATION.sub(lambda m: f'({m.group(1)})', h['text'])}"
                         for i, h in enumerate(hits, 1))
```

`format_context` also rewrites the passages. Articles cite their references as `[12]` or `[3, 4]`, in the shape of your citations: a copied `[3]` would pass the check while pointing at the wrong passage, and a copied `[12]` would turn a good answer into a refusal. So the pattern turns a passage's markers into `(12)` and `(3, 4)` in the prompt; the stored hits keep their text. The rewrite has limits. The pattern only matches lists of digits and commas, so a range marker such as `[21–24]`, which occurs in the PubMed texts of the recorded corpus, stays in brackets, and the prompt template itself contains `[1] or [2][4]`. What `tests/test_rag.py` pins is narrower: `[3]`, `[12]` and `[3, 4]` become parentheses.

`cited_numbers` returns the sorted, unique numbers cited in a reply, and `expand_citations` replaces each `[n, m]` with one `[source label]` per number. `generate_answer` in `src/rag.py` keeps a reply only when it cites at least one passage and every cited number is between 1 and the number of passages:

```python
def generate_answer(question, hits, model, api_key):
    """Ask the LLM about the given passages; keep the answer only if its citations are valid."""
    if not hits:
        return {"answer": REFUSAL, "refused": True, "reason": "no_context", "hits": hits, "cited": []}
    raw = call_llm(PROMPT.format(context=format_context(hits), question=question), model, api_key)
    refused = {"answer": REFUSAL, "refused": True, "hits": hits, "cited": [], "raw": raw}
    if REFUSAL.lower() in raw.lower():
        return {**refused, "reason": "model_refused"}
    cited = cited_numbers(raw)
    if not cited or not all(1 <= n <= len(hits) for n in cited):
        return {**refused, "reason": "uncited"}
    return {"answer": expand_citations(raw, hits).strip(), "refused": False, "reason": "answered",
            "hits": hits, "cited": cited, "raw": raw}
```

A result now has one of five reasons: `answered`, `below_threshold`, `no_context`, `model_refused` or `uncited`.

Save this toy script outside the project and run it from the project root with `.venv/bin/python path/to/script.py`. It replaces `rag.call_llm` with fake replies:

```python
import sys
sys.path.insert(0, "src")
import rag

hits = [{"id": "a", "text": "Give artesunate for at least 24 hours [12].", "source": "WHO_toy.pdf", "page": 42, "similarity": 0.8},
        {"id": "b", "text": "Paracetamol 1 g every 6 hours.", "source": "A toy article title", "page": None, "similarity": 0.7}]
print(rag.format_context(hits))
for raw in ["At least 24 hours [1]. Paracetamol treats fever [1, 2].", "At least 24 hours.",
            "At least 24 hours [3].", "I cannot answer this question from the provided documents."]:
    rag.call_llm = lambda prompt, model, api_key, raw=raw: raw
    res = rag.generate_answer("toy question", hits, rag.DEFAULT_MODEL, "key")
    print(res["reason"], rag.cited_numbers(raw), "->", res["answer"])
```

It prints the context, with the marker rewritten to `(12)`, then one verdict per reply:

```text
[1] WHO_toy.pdf, p. 42
Give artesunate for at least 24 hours (12).

[2] A toy article title
Paracetamol 1 g every 6 hours.
answered [1, 2] -> At least 24 hours [WHO_toy.pdf, p. 42]. Paracetamol treats fever [WHO_toy.pdf, p. 42][A toy article title].
uncited [] -> I cannot answer this question from the provided documents.
uncited [3] -> I cannot answer this question from the provided documents.
model_refused [] -> I cannot answer this question from the provided documents.
```

Look at the first reply. It passes, yet neither passage says that paracetamol treats fever. The check confirms that cited numbers exist, not that a passage supports the claim next to it, and it does not require a citation after every claim. Step 12 measures support with a judge model.

```bash
.venv/bin/python -m pytest -q tests/test_rag.py -k "format_context or cited_numbers or expand_citations or generate_answer or above_threshold"
```

## Step 8: Run it end to end

Run every test of `tests/test_rag.py` that does not need BM25:

```bash
.venv/bin/python -m pytest -q tests/test_rag.py -k "not rrf and not hybrid"
```

Once the index of Step 3 is built, ask a real question with this script of your own, not part of the project, as `.venv/bin/python path/to/script.py "question"`. It sends the question and the passages to OpenRouter:

```python
import sys
sys.path.insert(0, "src")
import index
import rag

res = rag.answer_question(sys.argv[1], index.load_embedder(), index.open_index(),
                          rag.DEFAULT_MODEL, rag.load_env()["OPENROUTER_API_KEY"])
print(res["reason"], round(res["best_similarity"], 3))
for h in res["hits"]:
    print(round(h["similarity"], 3), rag.source_label(h))
print(res.get("raw", res["answer"]))
```

Run it first with question A06, which `data/eval/questions.json` marks as answerable: "How long should parenteral artesunate be given to a patient with severe malaria?". Its evidence is on page 14 of `WHO_09_Guidelines_for_the_treatment_of_malaria.pdf`, so look for that file among the passages. The script prints the reason and best similarity, one line per passage, then the raw reply, or the refusal sentence when the model was not called.

Then run it with "What is the first-line topical treatment for scabies?". This is O01, an out-of-corpus question, and its expected outcome is the refusal sentence; if your system answers it, the refusal rate of Step 12 counts that against it. If A06 is refused with `below_threshold`, the threshold is too high for it, which Step 12 measures over all 30 questions.

## Step 9: Add BM25 keyword search

Embeddings capture meaning and can miss an exact string, such as the drug name zilebesiran in question A16. The architecture note lists "Dense retrieval misses exact drug names" as a risk, mitigated by BM25.

**BM25** is a keyword ranking built from three ideas. Inverse document frequency (idf) makes a term found in few chunks count more. Term frequency (tf) makes more occurrences count more, with diminishing returns set by `k1`. Length normalisation gives a long chunk less credit per occurrence, set by `b`. The docstring of `src/bm25.py` gives the formula, with N chunks, df chunks containing the term, dl the chunk length in tokens and avgdl the mean length:

```text
  idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
  score(q, d) = sum over t in q of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
```

You write it by hand with the standard library, for reasons in the project notes: the course asks for two baselines and BM25 is one (decision D7 in `vault/04-Architecture.md`), and `vault/12-Skills-Stack.md` chose a from-scratch sparse baseline, which `vault/03-Grading-Plan.md` counts as originality. The docstring of `src/bm25.py` credits its sources, and your copy must keep that credit: term and document frequencies and idf follow https://stevenloria.com/tf-idf/, tokenizing, indexing and ranking follow https://boyter.org/2010/08/build-vector-space-search-engine-python/ (both listed at https://github.com/codecrafters-io/build-your-own-x), and the formula is Okapi BM25 from Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.

The `BM25` class in `src/bm25.py` is an **inverted index**: for each term, the chunks that contain it and how often. `tokenize` lowercases and keeps runs of letters and digits, `idf` computes the first line of the formula, and `search` adds up the second:

```python
    def __init__(self, chunks, k1=1.5, b=0.75):
        # ...
        self.chunks, self.k1, self.b = chunks, k1, b
        self.lengths = []
        self.postings = {}
        for i, chunk in enumerate(chunks):
            terms = tokenize(chunk["text"])
            self.lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, {})[i] = tf
        self.avgdl = sum(self.lengths) / len(chunks) if chunks else 0.0
    # ...
    def search(self, question, k=5):
        # ...
        scores = Counter()
        for term in set(tokenize(question)):
            if term not in self.postings:
                continue
            idf = self.idf(term)
            for i, tf in self.postings[term].items():
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        best = sorted((i for i in scores if scores[i] > 0), key=lambda i: -scores[i])[:k]
        # ...
```

The cut lines return each hit as a dict with the chunk's id, text, source and page, `"similarity": None`, and the score under `"bm25"`. Work an example with this toy script over four made-up sentences:

```python
import sys
sys.path.insert(0, "src")
from bm25 import BM25, tokenize

texts = ["malaria is treated with artesunate",
         "severe malaria needs injectable artesunate for at least 24 hours",
         "malaria prevention uses bed nets",
         "hypertension is treated with drugs"]
chunks = [{"id": f"c{i}", "text": t, "source": f"s{i}", "page": None} for i, t in enumerate(texts)]
index = BM25(chunks)
print("lengths", index.lengths, "avgdl", index.avgdl)
for term in ("artesunate", "severe", "malaria"):
    print(term, "df", len(index.postings[term]), "idf", round(index.idf(term), 4))
print(tokenize("Severe malaria: artesunate?"))
for h in index.search("Severe malaria: artesunate?"):
    print(h["id"], round(h["bm25"], 4))
```

```text
lengths [5, 10, 5, 5] avgdl 6.25
artesunate df 2 idf 0.6931
severe df 1 idf 1.204
malaria df 3 idf 0.3567
['severe', 'malaria', 'artesunate']
c1 1.7746
c0 1.1537
c2 0.392
```

By hand, with N = 4: idf(severe) = ln(1 + 3.5 / 1.5) = 1.2040 is the highest, since one chunk holds it; idf(malaria) = ln(1 + 1.5 / 3.5) = 0.3567, since three do. Every tf is 1, so each term adds idf × 2.5 / (1 + 1.5 × norm):

- c0 has 5 tokens: norm = 0.25 + 0.75 × 5 / 6.25 = 0.85, a factor of 1.0989. It holds malaria and artesunate: (0.3567 + 0.6931) × 1.0989 = 1.1537.
- c1 has 10 tokens: norm = 1.45, a factor of 0.7874. It holds all three terms: (1.2040 + 0.3567 + 0.6931) × 0.7874 = 1.7746.
- c2 holds only malaria: 0.3567 × 1.0989 = 0.3920. c3 shares no term and is not returned.

c1 pays a length penalty but still wins, because it holds the rare word "severe".

```bash
.venv/bin/python -m pytest -q tests/test_bm25.py
```

**Exercise.** Build the index with `BM25(chunks, b=0)` and rerun. Compare c1's new score with the sum of its three idf values, and explain why they are equal.

## Step 10: Fuse the two rankings and go hybrid

The two rankings have different scales: a cosine similarity of at most 1, a BM25 score with no fixed bound. **Reciprocal rank fusion** (RRF) uses positions only. A chunk at rank r in a list earns 1 / (k + r), and its fused score is the sum over the lists, with k = 60. This is `rrf` in `src/rag.py`, after Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", SIGIR 2009, https://doi.org/10.1145/1571941.1572114, which the code cites:

```python
def rrf(rankings, k=60):
    # ...
    scores, first = {}, {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (k + rank)
            first.setdefault(hit["id"], hit)
    return [first[i] for i in sorted(scores, key=lambda i: -scores[i])]
```

Ties keep the order of first appearance, and an id keeps the dict of the earliest ranking that holds it. A toy script with two made-up lists:

```python
import sys
sys.path.insert(0, "src")
import rag

dense = [{"id": "A", "similarity": 0.82}, {"id": "B", "similarity": 0.79},
         {"id": "C", "similarity": 0.71}, {"id": "D", "similarity": 0.68}]
sparse = [{"id": "D", "similarity": None}, {"id": "C", "similarity": None}, {"id": "E", "similarity": None}]
print([(h["id"], h["similarity"]) for h in rag.rrf([dense, sparse])])
```

```text
[('D', 0.68), ('C', 0.71), ('A', 0.82), ('B', 0.79), ('E', None)]
```

By hand: D, 4th in dense and 1st in BM25, scores 1/64 + 1/61 = 0.032018; C, 3rd and 2nd, scores 1/63 + 1/62 = 0.032002; A, first in dense only, scores 1/61 = 0.016393. A chunk both rankings find beats a chunk that is first in one, and D keeps its dense similarity, 0.68.

In hybrid mode, the `else` branch of `answer_question` (Step 5) fuses 20 chunks (`CANDIDATES`) from each ranking and keeps the first 5. The gate still uses the best dense similarity, because BM25 hits carry no similarity and a BM25 score has no fixed scale to compare with 0.6. A test in `tests/test_rag.py` pins this: BM25 finds the paracetamol chunk, but the dense best is 0.55, so the result is `below_threshold`.

Two parts of `src/rag.py` remain. `load_context` calls `load_env`, `index.open_index` and `index.load_embedder`, and builds BM25 in memory with `BM25(index.all_chunks(collection))`, so both retrievers see the same chunks and no second store needs syncing. If the key or the index is missing, it exits with one line that says what to do. The command-line block replaces your Step 8 script: it calls `load_context`, takes the model from the second argument, else `OPENROUTER_MODEL` in `.env`, else `DEFAULT_MODEL`, and uses BM25 only with `--hybrid`.

```bash
.venv/bin/python -m pytest -q tests/test_rag.py
.venv/bin/python src/rag.py "How long should parenteral artesunate be given to a patient with severe malaria?" --hybrid
```

The run prints `reason=<reason> best_similarity=<value>`, one line per passage with its similarity and label, then the raw reply or the refusal sentence. A passage that only BM25 found shows `-` for its similarity.

## Step 11: Build the app

`src/app.py` puts a Gradio page over `answer_question`. The page shows `DISCLAIMER` under its heading, and every answer ends with it:

```python
DISCLAIMER = ("This tool is a student project. It is not medical advice. Check every answer with a clinician. "
              "Questions are sent to OpenRouter: do not type anything that identifies a patient.")
```

`respond` in `src/app.py` calls `answer_question`, passing BM25 only in Hybrid mode. It turns an empty question into `Type a question first.` and a failed model call into `The model call failed: ...` instead of a crash, since, as the README warns, a free model's call can fail or return an empty reply. `format_result` appends the disclaimer and, for a refusal, a reason such as `Best similarity 0.41 is below the threshold.` It also returns one table row per passage, with `yes` in the Cited column for each cited passage. In `build_app`, the model menu offers only `FREE_MODELS`, the retrieval choice defaults to Hybrid, and a threshold slider runs from 0.0 to 1.0 in steps of 0.05, starting at `SIM_THRESHOLD`.

```bash
.venv/bin/python -m pytest -q tests/test_app.py
.venv/bin/python src/app.py
```

Before you launch it, remember the disclaimer: not medical advice, every question goes to OpenRouter with the passages, free models only. Then open http://127.0.0.1:7860. Its three example questions are A06, A09 and the scabies question O01, which should be refused.

## Step 12: Evaluate

A few good-looking answers say nothing about the questions you did not ask, so `src/evaluate.py` measures. The project has not yet run its evaluation on the real corpus: this tutorial quotes no results, and every example here uses toy inputs.

**The question set**, `data/eval/questions.json`, holds 20 answerable questions (A01 to A20, each with a ground-truth answer, source, page and evidence quote) and 10 out-of-corpus ones that must be refused (O01 to O10: seven medical, three general knowledge).

**Four systems** run with each chosen model: `rag_dense` (Steps 4 to 7), `rag_hybrid` (with Step 10), and the two baselines the course requires, `bm25_rag` (the 5 best BM25 passages, same prompt and citation check, no gate) and `no_rag` (the model alone, counted as a refusal only if it says the refusal sentence).

**Metrics**, one row per system and model in `results/summary.csv`. `answer_rate` is the share of answerable questions not refused, and `refusal_rate` the share of out-of-corpus questions refused; you need both, because a system that refuses everything scores 1.0 on the second and 0 on the first. `hit_rate` is the share of answerable questions with a passage from the expected source. `n`, `n_judged` and `mean_latency_s` count the records, the judged records and the mean seconds per answer.

**Context precision** asks whether relevant passages sit near the top; a passage is relevant when its source is the question's source. Precision@k is the share of relevant passages among the first k, and context precision is its mean over the ranks k that hold a relevant passage. It follows RAGAS (Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023, https://arxiv.org/abs/2309.15217, and https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/), but the project computes it itself, without the `ragas` package. This is `average_precision` in `src/evaluate.py`:

```python
def average_precision(relevant):
    # ...
    found, total = 0, 0.0
    for k, rel in enumerate(relevant, 1):
        if rel:
            found += 1
            total += found / k
    return total / found if found else 0.0
```

For the toy list `[True, False, False, True, False]`, rank 1 is relevant with precision 1/1 and rank 4 with precision 2/4, so the score is (1 + 0.5) / 2 = 0.75. The same two relevant passages at ranks 1 and 2 give 1.0, and a list with none gives 0.0.

**Faithfulness** is the share of an answer's claims that its passages fully support, also from RAGAS: what the citation check cannot measure. With `--judge`, each answered record with passages gets one call to a judge model, always `rag.DEFAULT_MODEL`, which must reply `{"supported": <int>, "total": <int>}`. A judge is a model too, so `judge_faithfulness` in `src/evaluate.py` returns `None` when the reply has no readable JSON or impossible counts. A judge call that raises is caught by `run_questions`, which prints `judge error` and still writes the answer, with faithfulness empty.

**The threshold sweep.** `sweep_thresholds` in `src/evaluate.py` takes each question's best dense similarity and, for each of the 11 thresholds from 0.30 to 0.80 in steps of 0.05, computes the answer rate, the refusal rate and their mean, `balanced`. `best_threshold` returns the highest threshold among those with the best `balanced`. Try it on eight made-up points:

```python
import sys
sys.path.insert(0, "src")
import evaluate as ev

# made-up (best dense similarity, answerable) pairs, not measured on the corpus
points = [(0.74, True), (0.68, True), (0.61, True), (0.52, True),
          (0.63, False), (0.55, False), (0.47, False), (0.41, False)]
rows = ev.sweep_thresholds(points, ev.THRESHOLDS)
for r in rows:
    print(r["threshold"], r["answer_rate"], r["refusal_rate"], r["balanced"])
print("best", ev.best_threshold(rows))
```

```text
0.3 1.0 0.0 0.5
0.35 1.0 0.0 0.5
0.4 1.0 0.0 0.5
0.45 1.0 0.25 0.625
0.5 1.0 0.5 0.75
0.55 0.75 0.5 0.625
0.6 0.75 0.75 0.75
0.65 0.5 1.0 0.75
0.7 0.25 1.0 0.625
0.75 0.0 1.0 0.5
0.8 0.0 1.0 0.5
best 0.65
```

Check the 0.55 row: three answerable points of four pass, and of the out-of-corpus points only 0.47 and 0.41 are refused, since 0.55 is not below 0.55. Three thresholds tie at 0.75, and the rule picks the highest, 0.65: when the score cannot tell thresholds apart, it prefers the one that refuses more.

**Running it.**

```bash
.venv/bin/python -m pytest -q tests/test_evaluate.py tests/test_eval_set.py
.venv/bin/python src/evaluate.py sweep
.venv/bin/python src/evaluate.py run --judge
```

`sweep` prints one line per threshold, then `best threshold: <value>`, and writes `results/threshold_sweep.csv`; copy the value into `SIM_THRESHOLD` by hand. `run` takes `--systems`, `--models` (comma-separated, or `all`) and `--limit N` (the first N questions; the 20 answerable ones come first). It checks every model id with `check_free` before loading anything. It sends each question to OpenRouter at most once per system and model (a question the gate refuses is not sent), and `--judge` adds one call per judged answer, so the privacy rule applies.

`run` prints `<system> <model> <id>: <reason> (<seconds>s)` per answer and appends a record to `results/records.jsonl`. A rerun skips finished records and retries failed calls, which are not recorded. Faithfulness is only scored when a record is created, so give `--judge` from the first run. Each record stores its threshold, and `run_questions` stops with "move or delete the file and run again" if the file holds records made at another one. At the end it writes `results/summary.csv` and `results/human_rating_sheet.csv`, whose empty `safety_1_5` and `clarity_1_5` columns a person fills, since no metric measures safety or clarity; a rerun keeps ratings of unchanged answers.

**Exercise.** In the sweep script, move the out-of-corpus point 0.63 to 0.70 and predict the new best threshold before you run it.

## What each defence cannot do

Run the whole suite:

```bash
.venv/bin/python -m pytest -q
```

The real pipeline runs `scripts/fetch_corpus.py --from-manifest`, `scripts/check_eval_set.py`, `src/index.py`, `src/evaluate.py sweep`, `src/evaluate.py run --judge`, then `src/app.py`.

Each defence has a limit. The gate is one number: too high and it refuses answerable questions, too low and it lets weakly related passages through. The citation check proves that cited numbers exist, not that passages support the claims. The marker rewrite misses range markers. The judge is a model too. `check_eval_set.py` checks 20 quotes, not whole documents. And the corpus is a 50-document snapshot of guidelines that go out of date.

The three rules still hold: not medical advice, nothing that identifies a patient (and `results/records.jsonl` stores every evaluation question and answer), free models only.

## Sources

- BM25, built following two build-your-own-x tutorials (list at https://github.com/codecrafters-io/build-your-own-x): https://stevenloria.com/tf-idf/ (term and document frequencies, idf) and https://boyter.org/2010/08/build-vector-space-search-engine-python/ (tokenize, index, rank).
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.
- Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", SIGIR 2009, https://doi.org/10.1145/1571941.1572114
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023, https://arxiv.org/abs/2309.15217, and https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/

Tools used: Claude (Anthropic) was used to write the project's code and documentation, including this tutorial.
