# RAG on Medical Documents

This project answers medical questions from a small set of guidelines and articles. It finds the passages that best match a question, gives them to a free language model, and returns an answer that cites those passages by number, shown as the source (and the page for WHO guidelines). When a cross-encoder judges that no retrieved passage answers the question, or the model does not cite the passages, it replies "I cannot answer this question from the provided documents." instead of guessing.

This is a student project. It is not medical advice, and its answers must be checked with a clinician.

## Team and course

Team: Boussoualim Yassine, Benahcene Mohamed Wassim, Aissaoui Anes Safouane, Djerdali Mohamed Achraf.

Course: LLM Final Projects 2026-2027, Prof. A. Moussaoui, Project 2.

## Quick start

The commands are for a clean Linux machine with Python 3.12, run from the repository root.

1. Create the virtual environment: `python3 -m venv .venv`. On Debian and Ubuntu this needs the `python3-venv` package (`sudo apt install python3-venv`).
2. Install the dependencies: `.venv/bin/pip install -r requirements.txt`. The install downloads the packages, including the CPU build of PyTorch.
3. Create the settings file: `cp .env.example .env`. Then put a free OpenRouter key after `OPENROUTER_API_KEY=` in `.env`. `OPENROUTER_MODEL` is the default model for the command line and the evaluation. Every model must be a free one. Never commit `.env`.
4. Download the corpus used for the results: `.venv/bin/python scripts/fetch_corpus.py --from-manifest`. It writes `data/raw/who/` (PDF files) and `data/raw/pubmed/` (text files). See the Data section.
5. Build the index: `.venv/bin/python src/index.py`. The first run downloads the BGE-M3 model (about 2 GB), and the embedding takes about 3 hours on a laptop CPU. The build is resumable: after an interruption, run the same command again and it only embeds the chunks that are missing. `--rebuild` starts from zero. The index is stored in `chroma_db/`.
6. Launch the app: `.venv/bin/python src/app.py`, then open http://127.0.0.1:7860. The first launch downloads the cross-encoder of the refusal gate, `cross-encoder/ms-marco-MiniLM-L6-v2` (about 90 MB).

## Other commands

Ask one question from the command line:

```
.venv/bin/python src/rag.py "question" [model] [--hybrid] [--rerank]
```

It prints the reason (`answered`, `below_threshold`, `below_rerank_threshold`, `uncited`, `model_refused` or `no_context`), the best similarity (and, with `--rerank`, the cross-encoder gate score), the retrieved passages with their similarity, and the model's raw reply. The raw reply is printed also when the verdict is a refusal such as `uncited`. When the model is not called (`below_threshold`, `below_rerank_threshold` or `no_context`), it prints the refusal sentence. Without `model` it uses `OPENROUTER_MODEL` from `.env`, then the first model of `FREE_MODELS` in `src/rag.py`. Without `--hybrid` or `--rerank` it uses dense retrieval only. `--rerank` is hybrid retrieval with the cross-encoder gate, the app's default.

Run the tests (they use fakes: no model, no database, no network):

```
.venv/bin/pytest
```

Check that every evidence quote of the evaluation questions appears in the corpus (needs `data/raw/`):

```
.venv/bin/python scripts/check_eval_set.py
```

Compare the retrievers and the rerankers, with no LLM call (so it uses none of the daily request quota):

```
.venv/bin/python src/evaluate.py retrieval [--rerankers medcpt,bge,minilm]
```

For each retriever it measures, on the 30 questions, the hit rate and the page hit rate (the evidence page is among the 5 passages) on the answerable questions, context precision, and how well the score the refusal gate uses separates answerable from out-of-corpus questions: `auroc` (the share of answerable/out-of-corpus pairs in which the answerable question scores higher), `margin` (lowest answerable score minus highest out-of-corpus score; positive when one threshold separates them all) and `gap_share` (the margin as a share of the score range; it removes a linear change of scale, but not a non-linear one such as a sigmoid, so it does not compare different kinds of scores). It writes `results/retrieval.csv` and `results/retrieval_questions.csv` (each question's best score per retriever). The first run downloads the three cross-encoders (about 2.8 GB). `bge` is slow on a CPU (about a minute per question).

Calibrate the similarity threshold:

```
.venv/bin/python src/evaluate.py sweep
```

For each threshold from 0.30 to 0.80 in steps of 0.05, it takes the best dense similarity of every question and prints the answer rate, the refusal rate and their mean (called `balanced`), then the best threshold. It writes `results/threshold_sweep.csv`. It does not change the threshold: the value used by the app, the command line and the evaluation is `SIM_THRESHOLD` in `src/rag.py` (0.6), which the app also lets you change with a slider. To use the best value, copy it by hand into `SIM_THRESHOLD` in `src/rag.py`. Then move or delete `results/records.jsonl`, because `run` stops when that file holds records made at another threshold. Run `sweep` before the full `run`.

Evaluate the systems:

```
.venv/bin/python src/evaluate.py run [--systems a,b] [--models m1,m2|all] [--judge] [--limit N]
```

- `--systems`: comma-separated list taken from `rag_dense`, `rag_hybrid`, `rag_rerank`, `bm25_rag`, `no_rag`. The default is all five.
- `--models`: comma-separated model ids, or `all` for the five models of `FREE_MODELS`. The default is the single model `OPENROUTER_MODEL` from `.env`, or the first free model. A model that is not free stops the run.
- `--judge`: also scores faithfulness, with one extra LLM call per answered question that has passages. The faithfulness of a record is only scored in the run that creates the record, so give `--judge` from the first run. A failed judge call leaves that record's faithfulness empty.
- `--limit N`: keeps only the first N questions of the set. The set starts with the 20 answerable questions, so a small N contains no out-of-corpus question.

An identical request is sent to OpenRouter once. Every LLM call of the run, answers and judge, goes through a cache, `results/llm_cache.jsonl`: when the model, the prompt and the token budget are the same as in an earlier call, the stored reply is used and no request is sent. `rag_hybrid` and `rag_rerank` retrieve the same passages, so for a question that both gates let through, the second system reuses the answer and the judge reply of the first, and the two systems differ only by their gate. The run prints `reply reused` for such an answer, its record has `reused` set to true, and its latency includes the seconds the original call took. The cache is kept between runs. To get new replies from the models, delete `results/llm_cache.jsonl` together with `results/records.jsonl`.

The run is resumable: every answer is appended to `results/records.jsonl`, and a rerun skips what is already there. A call that fails is printed and not recorded, so a rerun retries it. Each record stores the threshold it was made with. After the run, these files are written in `results/`:

- `records.jsonl`: one line per answer.
- `llm_cache.jsonl`: one line per model reply, stored under the SHA-256 of its request with the seconds the call took. It holds neither the prompt nor the API key. A call that fails is not stored.
- `summary.csv`: one row of metrics per system and model, computed from all the records in `records.jsonl`.
- `human_rating_sheet.csv`: each answer next to its expected answer (the ground truth, or the refusal sentence for an out-of-corpus question), with `correct_0_1`, `safety_1_5` and `clarity_1_5` columns, to be filled by hand. `correct_0_1` is the only check of an answer against the ground truth, and the only one that covers the `no_rag` baseline. A rerun keeps the ratings already typed for answers that did not change, and leaves the other rows empty.

`results/threshold_sweep.csv` is written by the `sweep` command, not by `run`.

## Data

The corpus has 50 documents: 20 WHO guidelines (PDF files from WHO IRIS) and 30 open-access articles from PubMed Central (full text from the Europe PMC API, stored as text files). The documents are not committed: `data/raw/` is ignored by git, except `data/raw/manifest.json`. The manifest is committed and records the documents actually used (source, file, title, URL and size of each document). `docs/dataset_links.md` lists the source URL of every document. It is generated from the manifest with `.venv/bin/python scripts/make_dataset_links.py`.

To download the same documents under the same file names:

```
.venv/bin/python scripts/fetch_corpus.py --from-manifest
```

It downloads each WHO PDF from its recorded URL and the text of each PubMed article from Europe PMC. It prints one line per file and one line per failure, and it does not change the manifest. A PubMed article's text can differ from the evaluated one if Europe PMC has updated the article. `scripts/check_eval_set.py` checks that the evidence quotes of the question set are still in the corpus.

To build a different corpus, search WHO IRIS and Europe PMC by topic:

```
.venv/bin/python scripts/fetch_corpus.py --who 20 --pubmed 30
```

This search may not give the same documents as the ones used for the results, and it rewrites `data/raw/manifest.json`.

## How it works

1. Loading (`src/ingest.py`): each WHO PDF is read page by page with pypdf, and pages with 50 characters or fewer are skipped. Each PubMed article is one text record, cited by its title.
2. Chunking: each record is cut into windows of 512 tokens (BGE-M3 tokenizer) that overlap by 102 tokens. A chunk id is `source|page|start_token`.
3. Indexing (`src/index.py`): each chunk is embedded with `BAAI/bge-m3` and stored in a persistent Chroma collection named `medical_docs`, with cosine distance. The similarity of a hit is 1 minus the distance.
4. Retrieval (`src/rag.py`): the question is embedded and the 5 closest chunks are the passages.
5. Similarity gate: if the best similarity is below the threshold (0.6 by default), the answer is the refusal sentence and the model is not called.
6. Generation: the passages are numbered [1] to [5] in the prompt, and a free model on OpenRouter is asked to answer only from them and to cite them by number. The call is refused by `check_free` unless the model id ends in `:free` or is `openrouter/free`.
7. Citation check: the answer is kept only if it cites at least one passage and every cited number is one of the given passages. Then each number is replaced by the label of its source, `[file, p. N]` for a WHO guideline or `[article title]` for a PubMed article. If the model itself says the refusal sentence, or the check fails, the answer is replaced by the refusal sentence. The check confirms that the numbers exist. It does not confirm that a passage supports the claim next to it.
8. The refusal sentence is always `I cannot answer this question from the provided documents.`
9. Hybrid retrieval (`--hybrid`, or the Hybrid choice in the app): the 20 closest chunks by BGE-M3 and the 20 best chunks by BM25 are fused by reciprocal rank fusion (constant 60), and the first 5 are the passages. The similarity gate still uses the best dense similarity.
10. Cross-encoder gate (`--rerank`, or the app's default choice, "Hybrid + cross-encoder gate"): the passages are the hybrid ones, but the gate is different. A cross-encoder, `cross-encoder/ms-marco-MiniLM-L6-v2` (`src/rerank.py`), reads the question with each of the 5 passages and scores how well the passage answers it. The gate refuses when the best score, mapped to 0..1 by a sigmoid, is below `RERANK_THRESHOLD` in `src/rag.py` (0.5). The passages are scored one at a time, in retrieval order, and scoring stops at the first one that reaches the threshold: the decision is the same as with the best score, and an answerable question usually needs one passage instead of five. 0.5 is the model's own boundary between relevant and not relevant, so this threshold was not tuned on the question set.

Why this gate: the cosine similarity of BGE-M3 measures whether a passage is on the same topic, and an out-of-corpus medical question is often on a nearby topic. On the question set, the out-of-corpus question about isotretinoin for acne scores 0.588, just under the 0.6 threshold, and 0.6 was itself checked on the same 30 questions, so its perfect score there is optimistic. The cross-encoder reads the question and a passage together and scores whether the passage answers the question. It gives the isotretinoin question 0.156, and with a threshold that was not tuned on the question set it answers all 20 answerable questions (lowest score 0.870) and refuses all 10 out-of-corpus ones (highest 0.156). Both scores rank every answerable question above every out-of-corpus one (`auroc` 1.0 in `evaluate.py retrieval`), so on 30 questions the difference is the untuned threshold and the distance of the closest questions from it, not the ranking. Of the three cross-encoders compared as a gate over the same 5 passages, MiniLM is the only one that, at the model's own 0.5 boundary, answers all 20 answerable questions and refuses all 10 out-of-corpus ones, and it is the fastest (about 0.8 s per question on a CPU, with retrieval, when all 5 passages are scored in one batch as the comparison does; in the app, which stops at the first passage that reaches the threshold, retrieval and the gate take about 0.3 s for an answerable question and 0.5 s for an out-of-corpus one). The medical one, MedCPT, lets "Who painted the Mona Lisa?" through (0.78) and takes about 3 s; bge-reranker-v2-m3 lets the multiple sclerosis question through (0.85) and takes about 11 s. Used to reorder the 20 fused candidates instead, none of the three found the evidence pages more often than the hybrid passages, so the cross-encoder only gates and does not reorder.

Files of `src/`:

| File | Role |
|---|---|
| `ingest.py` | Reads the PDFs and text files and cuts them into chunks. |
| `index.py` | Embeds the chunks with BGE-M3 into Chroma, resumable, and reads them back. |
| `bm25.py` | Okapi BM25 search over the chunks, written from scratch with the standard library. |
| `rag.py` | Retrieval, similarity and cross-encoder gates, prompt, OpenRouter call, citation check, reciprocal rank fusion, and the command line. |
| `rerank.py` | Cross-encoder scoring: the refusal gate score and reranking. |
| `evaluate.py` | Evaluation of five systems on the question set, the retrieval comparison and the threshold sweep. |
| `app.py` | Gradio app: question box, model, retrieval mode and threshold controls, answer, and a table of the retrieved sources. |

## Evaluation

The question set is `data/eval/questions.json`: 30 questions, 20 answerable from the corpus (each with a ground-truth answer, its source, its page and a verbatim evidence quote) and 10 out of the corpus, which must be refused.

Five systems are compared, each with each chosen model:

- `rag_dense`: dense retrieval, similarity gate, cited answer.
- `rag_hybrid`: the same with BM25 fused in by reciprocal rank fusion.
- `rag_rerank`: `rag_hybrid` with the cross-encoder gate instead of the similarity gate.
- `bm25_rag`: the 5 best BM25 passages, then the same prompt and citation check, with no similarity gate.
- `no_rag`: the model alone, with no passages. It counts as a refusal only if it says the refusal sentence.

Metrics, per system and model (columns of `summary.csv`):

- `n`: number of recorded answers.
- `answer_rate`: share of answerable questions that were not refused.
- `refusal_rate`: share of out-of-corpus questions that were refused.
- `hit_rate`: share of answerable questions for which at least one retrieved passage comes from the expected source. Empty for `no_rag`.
- `context_precision`: for each answerable question, the mean of precision@k over the ranks k that hold a relevant passage (0 if none), averaged over the questions. A passage is relevant when its source is the question's source. Empty for `no_rag`.
- `faithfulness`: with `--judge`, for each answered question that has passages, one judge call (always the first model of `FREE_MODELS`) counts the claims of the answer and how many the passages fully support. The score is supported claims divided by all claims, averaged over the answers where the judge replied in a readable form with a supported count between 0 and the total. Empty without `--judge`.
- `n_judged`: number of records that have a faithfulness score.
- `mean_latency_s`: mean time per answer in seconds. For an answer that reused a stored reply, the time includes the seconds the original call took.

Context precision and faithfulness follow the RAGAS definitions, but they are computed by `src/evaluate.py`, not by the `ragas` package. Context precision uses the relevance labels of the question set instead of an LLM judge.

### Results

Results are written to results/summary.csv by the evaluation run.

## Limits and ethics

- This is not medical advice. A wrong or outdated answer can hurt a patient, so every answer must be checked with a clinician.
- Guidelines go out of date. The corpus is a snapshot of the documents downloaded for this project.
- The corpus is small (50 documents), so many real questions are outside it and are refused.
- The free models are rate-limited and slow, and a call can fail or return an empty reply. The code retries and pauses, and the app shows the error instead of crashing.
- Each gate is a single number. A value that is too high refuses answerable questions, and one that is too low lets weakly related passages through. The cross-encoder gate separates the 30 questions more widely than the similarity gate, but 30 questions are few, and the cross-encoder was trained on web search, not on medicine.
- The corpus contains no patient data. Questions are sent to OpenRouter with the retrieved passages, so do not type anything that identifies a patient. The evaluation stores its questions and answers in `results/records.jsonl`, and the replies of the models in `results/llm_cache.jsonl`.

## Tools and sources

Libraries: PyTorch (CPU), sentence-transformers, chromadb, pypdf, cryptography (to read encrypted PDFs), gradio and pytest. The embedding model is `BAAI/bge-m3`. The cross-encoder of the refusal gate is `cross-encoder/ms-marco-MiniLM-L6-v2`; `ncbi/MedCPT-Cross-Encoder` and `BAAI/bge-reranker-v2-m3` are compared with it. The language models are free models served by OpenRouter.

References cited in the code:

- BM25 is built following two build-your-own-x tutorials (list at https://github.com/codecrafters-io/build-your-own-x): https://stevenloria.com/tf-idf/ and https://boyter.org/2010/08/build-vector-space-search-engine-python/. The scoring formula is Okapi BM25 from Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.
- Reciprocal rank fusion: Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", SIGIR 2009, https://doi.org/10.1145/1571941.1572114
- Cross-encoders: Nogueira and Cho, "Passage Re-ranking with BERT", 2019, https://arxiv.org/abs/1901.04085; MedCPT: Jin et al., Bioinformatics, 2023, https://doi.org/10.1093/bioinformatics/btad651; https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 and https://huggingface.co/BAAI/bge-reranker-v2-m3
- RAGAS: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023, https://arxiv.org/abs/2309.15217, and https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/

Tools used: Claude (Anthropic) was used to write code and documentation.
