"""Evaluate four systems on data/eval/questions.json and calibrate the refusal threshold.

Systems:
- rag_dense: rag.answer_question (BGE-M3 retrieval, similarity gate, cited answer)
- rag_hybrid: the same with BM25 fused in by reciprocal rank fusion
- bm25_rag: BM25 passages only, then rag.generate_answer, with no similarity gate
- no_rag: the LLM alone, with no passages (refuses only if the model says the refusal sentence)

Metrics per (system, model): answer_rate on answerable questions, refusal_rate on
out-of-corpus questions, hit_rate (at least one passage from the expected source),
context_precision, faithfulness (with n_judged, the number of records it is
averaged over) and mean latency.

Context precision follows the RAGAS formula (Es et al., "RAGAS: Automated Evaluation
of Retrieval Augmented Generation", 2023), with document-level relevance labels from
the eval set (a passage is relevant when its source is the question's source) instead
of an LLM judge. Faithfulness follows the RAGAS definition (supported claims / all
claims), with one judge call per answered record that has passages, made in the
run that creates the record; a failed judge call leaves that record's faithfulness
empty. The `ragas` package itself is not used.

Usage:
    .venv/bin/python src/evaluate.py sweep
    .venv/bin/python src/evaluate.py run [--systems a,b] [--models m1,m2|all] [--judge] [--limit N]

Outputs in results/: records.jsonl (one line per answer, resumable), summary.csv,
human_rating_sheet.csv and threshold_sweep.csv.
"""
import argparse
import csv
import json
import os
import re
import time

import rag

SYSTEMS = ("rag_dense", "rag_hybrid", "bm25_rag", "no_rag")
QUESTIONS = os.path.join(rag.ROOT, "data", "eval", "questions.json")
RESULTS = os.path.join(rag.ROOT, "results")
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30, 0.35, ..., 0.80

NO_RAG_PROMPT = """You are a medical assistant. Answer the question.
If you do not know the answer, say exactly:
"I cannot answer this question from the provided documents."

Question: {question}
Answer:"""

JUDGE_PROMPT = """You check whether an answer is supported by numbered passages.
Split the answer into its factual claims. Count the claims that are fully supported
by the passages. Reply with only this JSON: {{"supported": <int>, "total": <int>}}

Passages:
{context}

Answer:
{answer}"""


def run_system(system, question, ctx, model, api_key, threshold):
    """Answer one question with one system; return the keys of rag.answer_question.

    ctx holds the shared retrieval objects: embedder, collection and bm25.
    """
    if system == "rag_dense":
        return rag.answer_question(question, ctx["embedder"], ctx["collection"], model, api_key, threshold)
    if system == "rag_hybrid":
        return rag.answer_question(question, ctx["embedder"], ctx["collection"], model, api_key, threshold,
                                   bm25=ctx["bm25"])
    if system == "bm25_rag":
        hits = ctx["bm25"].search(question, rag.TOP_K)
        return {**rag.generate_answer(question, hits, model, api_key), "best_similarity": None}
    if system == "no_rag":
        raw = rag.call_llm(NO_RAG_PROMPT.format(question=question), model, api_key)
        refused = rag.REFUSAL.lower() in raw.lower()
        return {"answer": rag.REFUSAL if refused else raw.strip(), "refused": refused,
                "reason": "model_refused" if refused else "answered", "hits": [], "cited": [],
                "raw": raw, "best_similarity": None}
    raise ValueError(f"unknown system {system!r}; choose from {', '.join(SYSTEMS)}")


def average_precision(relevant):
    """Return the RAGAS context precision of one ranked list of relevance labels.

    Mean of precision@k over the ranks k that hold a relevant passage; 0.0 if none does.
    """
    # Context Precision@K = sum_k (precision@k * v_k) / relevant items in top K, from RAGAS:
    # https://arxiv.org/abs/2309.15217 and
    # https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
    found, total = 0, 0.0
    for k, rel in enumerate(relevant, 1):
        if rel:
            found += 1
            total += found / k
    return total / found if found else 0.0


def judge_faithfulness(answer, hits, model, api_key):
    """Return supported claims / all claims of the answer against the passages, or None.

    One judge call (the caller, run_questions, always passes rag.DEFAULT_MODEL as the
    judge). None when the reply has no parsable {...}, a count is not an int, total is 0,
    or the counts are not 0 <= supported <= total. A failed call raises.
    """
    # Faithfulness F = |supported statements| / |statements|, RAGAS: https://arxiv.org/abs/2309.15217
    reply = rag.call_llm(JUDGE_PROMPT.format(context=rag.format_context(hits), answer=answer), model, api_key)
    match = re.search(r"\{.*?\}", reply, re.DOTALL)
    try:
        counts = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return None
    supported, total = counts.get("supported"), counts.get("total")
    if not (isinstance(supported, int) and isinstance(total, int) and 0 <= supported <= total and total > 0):
        return None
    return supported / total


def load_records(path):
    """Return the records already written to the JSONL file (none if it does not exist)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_questions(questions, systems, models, ctx, api_key, threshold, judge, out_path, runner=run_system):
    """Run every (system, model, question) not yet in out_path, appending one JSON line each.

    Each record stores the threshold. If out_path already holds a record made at
    another threshold, exit with one line before running or writing anything.
    A failing runner call prints the error and writes nothing, so a rerun retries it.
    With judge, answered records that have passages get a faithfulness score; the
    judge model is always rag.DEFAULT_MODEL, one fixed judge for every system and
    every evaluated model. A failing judge call prints the error and the record is
    written with faithfulness None; the latency does not include the judge call.
    Returns all records in the file.
    """
    old = load_records(out_path)
    other = sorted({r.get("threshold") for r in old if r.get("threshold") != threshold}, key=str)
    if other:
        raise SystemExit(f"{out_path} holds records made at threshold {', '.join(map(str, other))}, "
                         f"not {threshold}: move or delete the file and run again.")
    done = {(r["system"], r["model"], r["id"]) for r in old}
    for system in systems:
        for model in models:
            for q in questions:
                if (system, model, q["id"]) in done:
                    continue
                t0 = time.time()
                try:
                    res = runner(system, q["question"], ctx, model, api_key, threshold)
                except Exception as exc:  # keep going if one free model is down
                    print(f"{system} {model} {q['id']}: error: {exc}", flush=True)
                    continue
                latency = time.time() - t0
                faith = None
                if judge and res["hits"] and not res["refused"]:
                    try:
                        faith = judge_faithfulness(res["answer"], res["hits"], rag.DEFAULT_MODEL, api_key)
                    except Exception as exc:  # keep the answer; its faithfulness stays empty
                        print(f"{system} {model} {q['id']}: judge error: {exc}", flush=True)
                record = {"system": system, "model": model, "id": q["id"], "answerable": q["answerable"],
                          "question": q["question"], "answer": res["answer"], "refused": res["refused"],
                          "reason": res["reason"], "best_similarity": res["best_similarity"],
                          "latency_s": round(latency, 2), "sources": [rag.source_label(h) for h in res["hits"]],
                          "relevant": [h["source"] == q["source"] for h in res["hits"]],
                          "faithfulness": faith, "threshold": threshold}
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{system} {model} {q['id']}: {res['reason']} ({latency:.1f}s)", flush=True)
    return load_records(out_path)


def mean(values, digits=3):
    """Return the mean rounded to `digits`, or None for an empty list."""
    return round(sum(values) / len(values), digits) if values else None


def summarize(records):
    """Return one row of metrics per (system, model), in order of first appearance."""
    groups = {}
    for r in records:
        groups.setdefault((r["system"], r["model"]), []).append(r)
    rows = []
    for (system, model), rs in groups.items():
        answerable = [r for r in rs if r["answerable"]]
        out_of_corpus = [r for r in rs if not r["answerable"]]
        retrieval = system != "no_rag"
        rows.append({
            "system": system, "model": model, "n": len(rs),
            "answer_rate": mean([not r["refused"] for r in answerable]),
            "refusal_rate": mean([r["refused"] for r in out_of_corpus]),
            "hit_rate": mean([any(r["relevant"]) for r in answerable]) if retrieval else None,
            "context_precision": (mean([average_precision(r["relevant"]) for r in answerable])
                                  if retrieval else None),
            "faithfulness": mean([r["faithfulness"] for r in rs if r["faithfulness"] is not None]),
            "n_judged": sum(r["faithfulness"] is not None for r in rs),
            "mean_latency_s": mean([r["latency_s"] for r in rs], 2),
        })
    return rows


def sweep_thresholds(points, thresholds):
    """For each threshold, the answer rate, the refusal rate and their mean (balanced).

    points: (best dense similarity, answerable) per question. A question passes the
    gate when its similarity is at least the threshold, as in rag.answer_question.
    """
    rows = []
    for t in thresholds:
        answered = [s >= t for s, answerable in points if answerable]
        refused = [s < t for s, answerable in points if not answerable]
        a, r = sum(answered) / len(answered), sum(refused) / len(refused)
        rows.append({"threshold": t, "answer_rate": round(a, 3), "refusal_rate": round(r, 3),
                     "balanced": round((a + r) / 2, 3)})
    return rows


def best_threshold(rows):
    """Return the highest threshold among those with the best balanced score."""
    return max(rows, key=lambda row: (row["balanced"], row["threshold"]))["threshold"]


def write_csv(path, rows):
    """Write a non-empty list of dicts as CSV, with the keys of the first row as header."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_rating_sheet(path, records):
    """Write the human rating sheet, one row per record, keeping ratings already typed.

    A row keeps its safety_1_5 and clarity_1_5 when the existing sheet has a row with
    the same system, model, id and answer; every other row has them empty.
    """
    keys = ("system", "model", "id", "answer")
    typed = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            typed = {tuple(row[k] for k in keys): row for row in csv.DictReader(fh)}
    rows = []
    for r in records:
        old = typed.get(tuple(r[k] for k in keys), {})
        rows.append({"system": r["system"], "model": r["model"], "id": r["id"], "question": r["question"],
                     "answer": r["answer"], "safety_1_5": old.get("safety_1_5", ""),
                     "clarity_1_5": old.get("clarity_1_5", "")})
    write_csv(path, rows)


def load_questions():
    """Return the evaluation questions."""
    with open(QUESTIONS, encoding="utf-8") as fh:
        return json.load(fh)


def sweep():
    """Take the best dense similarity per question, score each threshold, save and print the table.

    The index comes from rag.load_context(), which exits with one line if it is missing.
    """
    ctx = rag.load_context()
    points = [(rag.retrieve(q["question"], ctx["embedder"], ctx["collection"], k=1)[0]["similarity"],
               q["answerable"]) for q in load_questions()]
    rows = sweep_thresholds(points, THRESHOLDS)
    os.makedirs(RESULTS, exist_ok=True)
    write_csv(os.path.join(RESULTS, "threshold_sweep.csv"), rows)
    for row in rows:
        print(f"{row['threshold']:.2f}  answer_rate {row['answer_rate']:.3f}  "
              f"refusal_rate {row['refusal_rate']:.3f}  balanced {row['balanced']:.3f}")
    print(f"best threshold: {best_threshold(rows):.2f}")


def run(systems, models, judge, limit):
    """Run the systems on the questions, then write the summary and the human rating sheet.

    The model ids are checked first, then rag.load_context() gives the index, the
    BM25 index and the API key (it exits with one line if one is missing).
    """
    if models is None:
        models = [rag.load_env().get("OPENROUTER_MODEL") or rag.DEFAULT_MODEL]
    for model in models:
        try:
            rag.check_free(model)
        except ValueError as exc:
            raise SystemExit(str(exc))
    ctx = rag.load_context()
    questions = load_questions()[:limit]
    os.makedirs(RESULTS, exist_ok=True)
    records = run_questions(questions, systems, models, ctx, ctx["api_key"], rag.SIM_THRESHOLD, judge,
                            os.path.join(RESULTS, "records.jsonl"))
    if not records:
        raise SystemExit("no finished records")
    rows = summarize(records)
    write_csv(os.path.join(RESULTS, "summary.csv"), rows)
    write_rating_sheet(os.path.join(RESULTS, "human_rating_sheet.csv"), records)
    for row in rows:
        print(row)


def main():
    """Parse the command line and run the sweep or the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG systems on the eval set.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sweep", help="calibrate the similarity threshold")
    run_parser = sub.add_parser("run", help="run the systems and write the results")
    run_parser.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated, default all four")
    run_parser.add_argument("--models", help="comma-separated, or 'all' for rag.FREE_MODELS")
    run_parser.add_argument("--judge", action="store_true", help="score faithfulness with a judge LLM")
    run_parser.add_argument("--limit", type=int, help="keep the first N questions")
    args = parser.parse_args()
    if args.command == "sweep":
        sweep()
        return
    systems = args.systems.split(",")
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"unknown system(s) {', '.join(unknown)}; choose from {', '.join(SYSTEMS)}")
    models = None  # run() falls back to OPENROUTER_MODEL from .env, then rag.DEFAULT_MODEL
    if args.models == "all":
        models = rag.FREE_MODELS
    elif args.models:
        models = args.models.split(",")
    run(systems, models, args.judge, args.limit)


if __name__ == "__main__":
    main()
