# RAG on Medical Documents

This project answers medical questions from a small set of guidelines and articles. It finds the passages that best match a question, gives them to a free language model, and returns an answer in which every claim points to one of those passages by source and page. When the retrieved passages are too far from the question, or the model does not cite them, it replies "I cannot answer this question from the provided documents." instead of guessing.

This is a student project. It is not medical advice, and its answers must be checked with a clinician.

## Team and course

Team: Boussoualim Yassine, Benahcene Mohamed Wassim, Aissaoui Anes Safouane, Djerdali Mohamed Achraf.

Course: LLM Final Projects 2026-2027, Prof. A. Moussaoui, Project 2.

## Quick start

The commands are for a clean Linux machine with Python 3 (the project was built with Python 3.12), run from the repository root.

1. Create the virtual environment: `python3 -m venv .venv`
2. Install the dependencies: `.venv/bin/pip install -r requirements.txt` (this installs the CPU build of PyTorch, so it needs network access).
3. Copy `.env.example` to `.env` and put a free OpenRouter key after `OPENROUTER_API_KEY=`. `OPENROUTER_MODEL` is the default model for the command line and the evaluation. Every model must be a free one. Never commit `.env`.
4. Get the corpus into `data/raw/who/` (PDF files) and `data/raw/pubmed/` (text files), as described in the Data section.
5. Build the index: `.venv/bin/python src/index.py`. The first run downloads the BGE-M3 model (about 2 GB), and the embedding takes about 3 hours on a laptop CPU. The build is resumable: after an interruption, run the same command again and it only embeds the chunks that are missing. `--rebuild` starts from zero. The index is stored in `chroma_db/`.
6. Launch the app: `.venv/bin/python src/app.py`, then open http://127.0.0.1:7860.

## Other commands

Ask one question from the command line:

```
.venv/bin/python src/rag.py "question" [model] [--hybrid]
```

It prints the reason (`answered`, `below_threshold`, `uncited`, `model_refused` or `no_context`), the best similarity, the retrieved passages with their similarity, and the model's answer. Without `model` it uses `OPENROUTER_MODEL` from `.env`, then the first model of `FREE_MODELS` in `src/rag.py`. Without `--hybrid` it uses dense retrieval only.

Run the tests (they use fakes: no model, no database, no network):

```
.venv/bin/pytest
```

Check that every evidence quote of the evaluation questions appears in the corpus (needs `data/raw/`):

```
.venv/bin/python scripts/check_eval_set.py
```

Calibrate the similarity threshold:

```
.venv/bin/python src/evaluate.py sweep
```

For each threshold from 0.30 to 0.80 in steps of 0.05, it takes the best dense similarity of every question and prints the answer rate, the refusal rate and their mean (called `balanced`), then the best threshold. It writes `results/threshold_sweep.csv`. It does not change the threshold: the value used by the app, the command line and the evaluation is `SIM_THRESHOLD` in `src/rag.py` (0.6), which the app also lets you change with a slider.

Evaluate the systems:

```
.venv/bin/python src/evaluate.py run [--systems a,b] [--models m1,m2|all] [--judge] [--limit N]
```

- `--systems`: comma-separated list taken from `rag_dense`, `rag_hybrid`, `bm25_rag`, `no_rag`. The default is all four.
- `--models`: comma-separated model ids, or `all` for the five models of `FREE_MODELS`. The default is the single model `OPENROUTER_MODEL` from `.env`, or the first free model. A model that is not free stops the run.
- `--judge`: also scores faithfulness, with one extra LLM call per answer.
- `--limit N`: keeps only the first N questions of the set. The set starts with the 20 answerable questions, so a small N contains no out-of-corpus question.

The run is resumable: every answer is appended to `results/records.jsonl`, and a rerun skips what is already there. A call that fails is printed and not recorded, so a rerun retries it. After the run, these files are written in `results/`:

- `records.jsonl`: one line per answer.
- `summary.csv`: one row of metrics per system and model, computed from all the records in `records.jsonl`.
- `human_rating_sheet.csv`: the answers with empty `safety_1_5` and `clarity_1_5` columns, to be filled by hand.
- `threshold_sweep.csv`: written by the `sweep` command.

## Data

The corpus has 50 documents: 20 WHO guidelines (PDF files from WHO IRIS) and 30 open-access articles from PubMed Central (full text from the Europe PMC API, stored as text files). The documents are not committed: `data/raw/` is ignored by git. `docs/dataset_links.md` lists the source URL of every document.

`scripts/fetch_corpus.py` downloads a corpus of this kind:

```
.venv/bin/python scripts/fetch_corpus.py --who 20 --pubmed 30
```

It searches WHO IRIS and Europe PMC by topic, so a new download may not give exactly the same documents as the ones used for the results. `data/raw/manifest.json` records what was actually used (source, file, title, URL and size of each document). `docs/dataset_links.md` is generated from it with `.venv/bin/python scripts/make_dataset_links.py`.

## How it works

1. Loading (`src/ingest.py`): each WHO PDF is read page by page with pypdf, and pages with 50 characters or fewer are skipped. Each PubMed article is one text record, cited by its title.
2. Chunking: each record is cut into windows of 512 tokens (BGE-M3 tokenizer) that overlap by 102 tokens. A chunk id is `source|page|start_token`.
3. Indexing (`src/index.py`): each chunk is embedded with `BAAI/bge-m3` and stored in a persistent Chroma collection named `medical_docs`, with cosine distance. The similarity of a hit is 1 minus the distance.
4. Retrieval (`src/rag.py`): the question is embedded and the 5 closest chunks are the passages.
5. Similarity gate: if the best similarity is below the threshold (0.6 by default), the answer is the refusal sentence and the model is not called.
6. Generation: the passages are numbered [1] to [5] in the prompt, and a free model on OpenRouter is asked to answer only from them and to cite them by number. The call is refused by `check_free` unless the model id ends in `:free` or is `openrouter/free`.
7. Citation check: the answer is kept only if it cites at least one passage and every cited number is one of the given passages. Then each number is replaced by the label of its source, `[file, p. N]` for a WHO guideline or `[article title]` for a PubMed article. If the model itself says the refusal sentence, or the check fails, the answer is replaced by the refusal sentence. The check confirms that the numbers exist. It does not confirm that a passage supports the claim next to it.
8. The refusal sentence is always `I cannot answer this question from the provided documents.`
9. Hybrid retrieval (`--hybrid`, or the Hybrid choice in the app, which is the app's default): the 20 closest chunks by BGE-M3 and the 20 best chunks by BM25 are fused by reciprocal rank fusion (constant 60), and the first 5 are the passages. The similarity gate still uses the best dense similarity.

Files of `src/`:

| File | Role |
|---|---|
| `ingest.py` | Reads the PDFs and text files and cuts them into chunks. |
| `index.py` | Embeds the chunks with BGE-M3 into Chroma, resumable, and reads them back. |
| `bm25.py` | Okapi BM25 search over the chunks, written from scratch with the standard library. |
| `rag.py` | Retrieval, similarity gate, prompt, OpenRouter call, citation check, reciprocal rank fusion, and the command line. |
| `evaluate.py` | Evaluation of four systems on the question set, and the threshold sweep. |
| `app.py` | Gradio app: question box, model, retrieval mode and threshold controls, answer, and a table of the retrieved sources. |

## Evaluation

The question set is `data/eval/questions.json`: 30 questions, 20 answerable from the corpus (each with a ground-truth answer, its source, its page and a verbatim evidence quote) and 10 out of the corpus, which must be refused.

Four systems are compared, each with each chosen model:

- `rag_dense`: dense retrieval, similarity gate, cited answer.
- `rag_hybrid`: the same with BM25 fused in by reciprocal rank fusion.
- `bm25_rag`: the 5 best BM25 passages, then the same prompt and citation check, with no similarity gate.
- `no_rag`: the model alone, with no passages. It counts as a refusal only if it says the refusal sentence.

Metrics, per system and model (columns of `summary.csv`):

- `n`: number of recorded answers.
- `answer_rate`: share of answerable questions that were not refused.
- `refusal_rate`: share of out-of-corpus questions that were refused.
- `hit_rate`: share of answerable questions for which at least one retrieved passage comes from the expected source. Empty for `no_rag`.
- `context_precision`: for each answerable question, the mean of precision@k over the ranks k that hold a relevant passage (0 if none), averaged over the questions. A passage is relevant when its source is the question's source. Empty for `no_rag`.
- `faithfulness`: with `--judge`, for each answered question that has passages, one judge call (always the first model of `FREE_MODELS`) counts the claims of the answer and how many the passages fully support. The score is supported claims divided by all claims, averaged over the answers where the judge replied in a readable form. Empty without `--judge`.
- `mean_latency_s`: mean time per answer in seconds.

Context precision and faithfulness follow the RAGAS definitions, but they are computed by `src/evaluate.py`, not by the `ragas` package. Context precision uses the relevance labels of the question set instead of an LLM judge.

### Results

Results are written to results/summary.csv by the evaluation run.

## Limits and ethics

- This is not medical advice. A wrong or outdated answer can hurt a patient, so every answer must be checked with a clinician.
- Guidelines go out of date. The corpus is a snapshot of the documents downloaded for this project.
- The corpus is small (50 documents), so many real questions are outside it and are refused.
- The free models are rate-limited and slow, and a call can fail or return an empty reply. The code retries and pauses, and the app shows the error instead of crashing.
- The similarity threshold is a single number. A value that is too high refuses answerable questions, and one that is too low lets weakly related passages through.
- No patient data is stored or processed. The questions are sent to OpenRouter with the retrieved passages.

## Tools and sources

Libraries: PyTorch (CPU), sentence-transformers, chromadb, pypdf, cryptography (to read encrypted PDFs), gradio and pytest. The embedding model is `BAAI/bge-m3`. The language models are free models served by OpenRouter.

References cited in the code:

- BM25 is built following two build-your-own-x tutorials (list at https://github.com/codecrafters-io/build-your-own-x): https://stevenloria.com/tf-idf/ and https://boyter.org/2010/08/build-vector-space-search-engine-python/. The scoring formula is Okapi BM25 from Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.
- Reciprocal rank fusion: Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", SIGIR 2009, https://doi.org/10.1145/1571941.1572114
- RAGAS: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023, https://arxiv.org/abs/2309.15217, and https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/

Tools used: Claude (Anthropic) was used to write code and documentation.
