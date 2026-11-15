"""Write the LaTeX tables and the sweep figure of the report from the results files.

Run from the repo root: .venv/bin/python scripts/make_report_assets.py

Reads results/summary.csv, results/retrieval.csv, results/threshold_sweep.csv and
results/records.jsonl and writes docs/report/generated/summary_table.tex, retrieval_table.tex,
sweep_table.tex, sweep_plot.tex and reasons_table.tex. Values are copied from the CSV files as they are; an empty cell prints as
"--". A missing or empty input file gives a table (or figure) that says the results are not
yet available. Standard library only.
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "report", "generated")

BASELINES = ("no_rag", "bm25_rag")
SYSTEM_ORDER = ["no_rag", "bm25_rag", "rag_dense", "rag_hybrid", "rag_rerank"]
# which evaluated system uses each row of results/retrieval.csv
USED_BY = {"dense": "rag_dense", "bm25": "bm25_rag (baseline)", "hybrid": "rag_hybrid, rag_rerank",
           "hybrid+minilm_gate": "rag_rerank (gate)"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

SUMMARY_CAPTION = ("Evaluation of each system and model (from \\texttt{results/summary.csv}). "
                   "\\texttt{no\\_rag} and \\texttt{bm25\\_rag} are the baselines. Answer: share of the answerable "
                   "questions A01--A20 not refused. Ref.~O: share of the out-of-corpus questions O01--O10 refused. "
                   "Ref.~N: share of the held-out questions N01--N10 refused. Hit: share of answerable questions with "
                   "a passage from the expected source. CP: context precision. Faith.: faithfulness, with the number "
                   "of judged answers in brackets. Lat.: mean seconds per answer. -- means no value.")
RETRIEVAL_CAPTION = ("Retrieval and gate comparison, with no LLM call (from \\texttt{results/retrieval.csv}). "
                     "Hit rates and context precision are on the 20 answerable questions; auroc, margin and gap\\_share "
                     "separate them from the 10 out-of-corpus ones, the heldout columns from the 10 held-out ones. "
                     "BM25 is the keyword baseline. \\emph{+name} rows reorder the 20 "
                     "fused candidates with a cross-encoder; \\emph{\\_gate} rows keep the hybrid passages and use the "
                     "cross-encoder score only for the refusal gate. The margin is in the units of each score "
                     "(cosine, BM25 score, cross-encoder logit or probability); gap\\_share divides it by the score range.")
SWEEP_CAPTION = ("Threshold sweep of the dense similarity gate on the original 30 questions (from "
                 "\\texttt{results/threshold\\_sweep.csv}). Balanced is the mean of the answer rate on A01--A20 and the "
                 "refusal rate on O01--O10. The lowest threshold, which refuses almost nothing, plays the role of a "
                 "system without a gate.")
REASONS_CAPTION = ("Outcome of each recorded answer, by system, model and question group (counted from "
                   "\\texttt{results/records.jsonl}). \\texttt{no\\_rag} and \\texttt{bm25\\_rag} are the baselines. "
                   "Gate: refused before any model call. By model: the model wrote the refusal sentence. Uncited: the reply "
                   "had no valid citation. Other: any other reason, such as no passage.")
GROUPS = {"A": "A01--A20", "O": "O01--O10", "N": "N01--N10"}
PLOT_CAPTION = ("Answer rate, refusal rate and their mean (balanced) against the similarity threshold, on the "
                "original 30 questions (from \\texttt{results/threshold\\_sweep.csv}).")


def escape(text):
    """Return text with the LaTeX special characters escaped."""
    return "".join(SPECIAL.get(ch, ch) for ch in str(text))


def read_rows(path):
    """Return the rows of a CSV file as dicts, or None when the file is missing or has no data row."""
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def cell(value):
    """Return a value escaped for LaTeX, or "--" when it is empty."""
    return escape(value) if value not in (None, "") else "--"


def table(caption, label, header, rows, align):
    """Return a LaTeX table: caption, booktabs rules, one line per row; a table of more than 5 columns is
    scaled to the text width."""
    wide = len(header) > 5
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%" if wide else r"{%", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def unavailable(caption, label, name):
    """Return a small table that says the results of the named file are not yet available."""
    return table(caption, label, ["Status"],
                 [[rf"Results not yet available: \texttt{{results/{escape(name)}}} is missing or empty."]], "l")


def summary_table(rows):
    """Return the table of results/summary.csv: one row per system and model, baselines first."""
    if rows is None:
        return unavailable(SUMMARY_CAPTION, "tab:summary", "summary.csv")
    order = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    rows = sorted(rows, key=lambda r: order.get(r["system"], len(order)))   # stable: models keep their order
    body = []
    for r in rows:
        faith = "--" if r["faithfulness"] in ("", None) else f"{escape(r['faithfulness'])} ({escape(r['n_judged'])})"
        body.append(["baseline" if r["system"] in BASELINES else "RAG", rf"\texttt{{{escape(r['system'])}}}",
                     escape(r["model"]), cell(r["n"]), cell(r["answer_rate"]), cell(r["refusal_rate"]),
                     cell(r["heldout_refusal_rate"]), cell(r["hit_rate"]), cell(r["context_precision"]), faith,
                     cell(r["mean_latency_s"])])
    header = ["Role", "System", "Model", "n", "Answer", "Ref. O", "Ref. N", "Hit", "CP", "Faith.", "Lat. (s)"]
    return table(SUMMARY_CAPTION, "tab:summary", header, body, "lllrrrrrrrr")


def retrieval_table(rows):
    """Return the table of results/retrieval.csv: every column of the file, plus the system that uses the row."""
    if rows is None:
        return unavailable(RETRIEVAL_CAPTION, "tab:retrieval", "retrieval.csv")
    columns = [c for c in rows[0] if c != "retriever"]
    body = [[escape(r["retriever"]), escape(USED_BY.get(r["retriever"], "--"))] + [cell(r[c]) for c in columns]
            for r in rows]
    header = ["Retriever", "Used by"] + [escape(c) for c in columns]
    return table(RETRIEVAL_CAPTION, "tab:retrieval", header, body, "ll" + "r" * len(columns))


def best_threshold(rows):
    """Return the threshold string of the best balanced row; on a tie, the highest threshold (as src/evaluate.py)."""
    return max(rows, key=lambda r: (float(r["balanced"]), float(r["threshold"])))["threshold"]


def sweep_table(rows):
    """Return the table of results/threshold_sweep.csv, with the best balanced threshold marked."""
    if rows is None:
        return unavailable(SWEEP_CAPTION, "tab:sweep", "threshold_sweep.csv")
    best = best_threshold(rows)
    body = [[cell(r["threshold"]), cell(r["answer_rate"]), cell(r["refusal_rate"]), cell(r["balanced"]),
             "best balanced" if r["threshold"] == best else ""] for r in rows]
    return table(SWEEP_CAPTION, "tab:sweep", ["Threshold", "Answer rate", "Refusal rate", "Balanced", "Note"],
                 body, "rrrrl")


def sweep_plot(rows):
    """Return a pgfplots figure of the sweep: one line per rate against the threshold."""
    head = [r"\begin{figure}[htbp]", r"\centering"]
    tail = [rf"\caption{{{PLOT_CAPTION}}}", r"\label{fig:sweep}", r"\end{figure}"]
    if rows is None:
        note = r"\fbox{\parbox{0.8\linewidth}{\centering Results not yet available: \texttt{results/threshold\_sweep.csv} is missing or empty.}}"
        return "\n".join(head + [note] + tail) + "\n"
    lines = head + [r"\begin{tikzpicture}",
                    r"\begin{axis}[width=0.8\linewidth, height=5.2cm, xlabel={Similarity threshold}, "
                    r"ylabel={Rate}, ymin=0, ymax=1.05, grid=major, legend columns=3, "
                    r"legend style={at={(0.5,-0.32)}, anchor=north, font=\small}]"]
    for key, style, name in (("answer_rate", "blue, mark=*", "Answer rate (A01--A20)"),
                             ("refusal_rate", "red, mark=square*", "Refusal rate (O01--O10)"),
                             ("balanced", "black, dashed, mark=triangle*", "Balanced")):
        points = " ".join(f"({r['threshold']},{r[key]})" for r in rows)
        lines += [rf"\addplot[{style}] coordinates {{{points}}};", rf"\addlegendentry{{{name}}}"]
    lines += [r"\end{axis}", r"\end{tikzpicture}"] + tail
    return "\n".join(lines) + "\n"


def read_records(path):
    """Return the records of a JSON-lines file, skipping lines that do not parse; None if there are none."""
    if not os.path.exists(path):
        return None
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records or None


def reasons_table(records):
    """Return the table of outcomes (answered, refused by the gate or the model, uncited, other) per system,
    model and question group, baselines first."""
    if records is None:
        return unavailable(REASONS_CAPTION, "tab:reasons", "records.jsonl")
    order = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    models = list(dict.fromkeys(r["model"] for r in records))
    counts = {}
    for r in records:
        reason = r["reason"]
        kind = ("answered" if reason == "answered" else "gate" if reason.startswith("below_") else
                "model" if reason == "model_refused" else "uncited" if reason == "uncited" else "other")
        key = (order.get(r["system"], len(order)), r["system"], models.index(r["model"]), r["id"][0])
        counts.setdefault(key, dict.fromkeys(["answered", "gate", "model", "uncited", "other"], 0))[kind] += 1
    body = []
    for key in sorted(counts, key=lambda k: (k[0], k[1], k[2], "AON".find(k[3]))):
        c = counts[key]
        body.append([rf"\texttt{{{escape(key[1])}}}", escape(models[key[2]]), GROUPS.get(key[3], escape(key[3])),
                     str(sum(c.values()))] + [str(c[k]) for k in ("answered", "gate", "model", "uncited", "other")])
    header = ["System", "Model", "Questions", "n", "Answered", "Gate", "By model", "Uncited", "Other"]
    return table(REASONS_CAPTION, "tab:reasons", header, body, "lllrrrrrr")


def main(results_dir=RESULTS, out_dir=OUT):
    """Read the results files and write the LaTeX files into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    sweep = read_rows(os.path.join(results_dir, "threshold_sweep.csv"))
    outputs = {"summary_table.tex": summary_table(read_rows(os.path.join(results_dir, "summary.csv"))),
               "retrieval_table.tex": retrieval_table(read_rows(os.path.join(results_dir, "retrieval.csv"))),
               "sweep_table.tex": sweep_table(sweep), "sweep_plot.tex": sweep_plot(sweep),
               "reasons_table.tex": reasons_table(read_records(os.path.join(results_dir, "records.jsonl")))}
    for name, tex in outputs.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(tex)
        print("wrote", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
