"""Tests for scripts/make_report_assets.py: LaTeX tables and the sweep figure built from results files (no network)."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import make_report_assets as mra  # noqa: E402

SUMMARY_HEADER = ["system", "model", "n", "answer_rate", "refusal_rate", "heldout_refusal_rate", "hit_rate",
                  "context_precision", "faithfulness", "n_judged", "mean_latency_s"]


def write_csv(path, header, rows):
    """Write a CSV file with the given header and rows (lists of values)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_summary_table_has_one_row_per_system_and_model_with_baselines_first(tmp_path):
    path = tmp_path / "summary.csv"
    write_csv(path, SUMMARY_HEADER, [
        ["rag_rerank", "m/a:free", "39", "1.0", "1.0", "0.9", "1.0", "0.906", "0.97", "19", "6.03"],
        ["no_rag", "m/a:free", "6", "1.0", "", "", "", "", "", "0", "9.31"],
        ["rag_rerank", "m/b:free", "40", "0.95", "1.0", "0.8", "1.0", "0.911", "0.9", "18", "5.18"]])
    tex = mra.summary_table(mra.read_rows(str(path)))
    body = tex[tex.index(r"\midrule"):tex.index(r"\bottomrule")]
    assert sum(line.rstrip().endswith("\\\\") for line in body.splitlines()) == 3   # one row each
    assert body.index("no\\_rag") < body.index("rag\\_rerank")         # baseline first
    assert "baseline" in body and "0.97 (19)" in body and "0.9 (18)" in body
    assert "m/b:free" in body and "0.911" in body and "6.03" in body
    assert r"\caption{" in tex and r"\label{tab:summary}" in tex
    no_rag_line = next(line for line in body.splitlines() if "no\\_rag" in line)
    assert no_rag_line.count("--") >= 5                                 # empty cells stay empty, never invented


def test_a_missing_or_header_only_file_gives_a_not_available_table(tmp_path):
    empty = tmp_path / "summary.csv"
    write_csv(empty, SUMMARY_HEADER, [])
    for rows in (mra.read_rows(str(tmp_path / "absent.csv")), mra.read_rows(str(empty))):
        assert rows is None
        tex = mra.summary_table(rows)
        assert "not yet available" in tex and r"\label{tab:summary}" in tex
        assert "rag\\_rerank" not in tex


def test_retrieval_table_names_the_system_that_uses_each_retriever(tmp_path):
    path = tmp_path / "retrieval.csv"
    write_csv(path, ["retriever", "hit_rate", "page_hit_rate", "context_precision", "auroc", "margin", "gap_share",
                     "mean_latency_ms"],
              [["dense", "0.95", "0.8", "0.857", "1.0", "0.0357", "0.074", "185"],
               ["bm25", "1.0", "0.95", "0.932", "0.945", "-5.8951", "-0.157", "7"],
               ["hybrid+minilm_gate", "1.0", "0.9", "0.911", "1.0", "3.5874", "0.187", "837"]])
    tex = mra.retrieval_table(mra.read_rows(str(path)))
    bm25_line = next(line for line in tex.splitlines() if line.startswith("bm25"))
    assert "baseline" in bm25_line and "-5.8951" in bm25_line
    assert "hybrid+minilm\\_gate" in tex and "rag\\_rerank" in tex and r"\label{tab:retrieval}" in tex


def test_sweep_table_and_plot_use_every_threshold_and_mark_the_best(tmp_path):
    path = tmp_path / "threshold_sweep.csv"
    write_csv(path, ["threshold", "answer_rate", "refusal_rate", "balanced"],
              [["0.5", "1.0", "0.3", "0.65"], ["0.6", "1.0", "1.0", "1.0"], ["0.65", "0.95", "1.0", "0.975"]])
    rows = mra.read_rows(str(path))
    table = mra.sweep_table(rows)
    best_line = next(line for line in table.splitlines() if line.startswith("0.6 "))
    assert "best" in best_line and r"\label{tab:sweep}" in table
    plot = mra.sweep_plot(rows)
    assert "(0.5,1.0)" in plot and "(0.65,0.975)" in plot and plot.count(r"\addplot") >= 3
    assert r"\label{fig:sweep}" in plot
    assert "not yet available" in mra.sweep_plot(None) and r"\label{fig:sweep}" in mra.sweep_plot(None)


def test_main_writes_every_file_and_says_which_inputs_are_missing(tmp_path):
    results, out = tmp_path / "results", tmp_path / "generated"
    results.mkdir()
    write_csv(results / "summary.csv", SUMMARY_HEADER,
              [["bm25_rag", "m/a:free", "40", "0.9", "0.8", "0.5", "1.0", "0.93", "0.95", "18", "4.2"]])
    mra.main(str(results), str(out))
    names = sorted(os.listdir(out))
    assert names == ["reasons_table.tex", "retrieval_table.tex", "summary_table.tex", "sweep_plot.tex",
                     "sweep_table.tex"]
    assert "bm25\\_rag" in (out / "summary_table.tex").read_text(encoding="utf-8")
    for name in ("reasons_table.tex", "retrieval_table.tex", "sweep_table.tex", "sweep_plot.tex"):
        assert "not yet available" in (out / name).read_text(encoding="utf-8")


def test_reasons_table_counts_each_outcome_per_system_model_and_question_group(tmp_path):
    path = tmp_path / "records.jsonl"
    lines = [{"system": "rag_rerank", "model": "m:free", "id": "N01", "reason": "below_rerank_threshold"},
             {"system": "rag_rerank", "model": "m:free", "id": "N02", "reason": "model_refused"},
             {"system": "rag_rerank", "model": "m:free", "id": "N03", "reason": "model_refused"},
             {"system": "rag_rerank", "model": "m:free", "id": "A01", "reason": "answered"},
             {"system": "no_rag", "model": "m:free", "id": "A01", "reason": "answered"}]
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n{torn line\n", encoding="utf-8")
    records = mra.read_records(str(path))
    assert len(records) == 5                                            # the torn line is skipped
    tex = mra.reasons_table(records)
    held_out = next(line for line in tex.splitlines() if "rag\\_rerank" in line and "N01--N10" in line)
    assert held_out.split(" & ")[3:8] == ["3", "0", "1", "2", "0"]      # n, answered, gate, model, uncited
    body = tex[tex.index(r"\midrule"):]
    assert body.index("no\\_rag") < body.index("rag\\_rerank") and r"\label{tab:reasons}" in tex
    assert "not yet available" in mra.reasons_table(mra.read_records(str(tmp_path / "absent.jsonl")))


def test_escape_handles_latex_special_characters():
    assert mra.escape("a_b & 50% #1") == r"a\_b \& 50\% \#1"
