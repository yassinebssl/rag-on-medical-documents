"""Write site/public/index.html: the evaluation's recorded answers as one static page, for a free host.

The page makes no model call, so it spends no OpenRouter quota and needs no key. It shows each question of the
evaluation set with the answer, the verdict and the passages the rag_rerank system recorded in results/records.jsonl.
No passage text is published, only the source names.

Run with .venv/bin/python scripts/build_demo_site.py after the evaluation has recorded new answers.
"""
import json
import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import app  # noqa: E402  (the verdict wording and colours of the web app)

SYSTEM = "rag_rerank"
REPO = "https://github.com/yassinebssl/rag-on-medical-documents"
GROUPS = [("A", "Answerable from the documents",
           "Each answer is in one of the 50 documents. The system should answer and cite where."),
          ("N", "Covered topic, missing fact",
           "The documents cover the topic but not the asked fact. Written before any run, used for no choice."),
          ("O", "Outside the documents",
           "Seven medical topics absent from the corpus and three general-knowledge questions.")]
CHEVRON = ('<svg class="chevron" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">'
           '<path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" stroke-width="1.6" '
           'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def load_records(path=ROOT / "results" / "records.jsonl"):
    """Return the recorded answers of SYSTEM, sorted by question id."""
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return sorted((r for r in records if r["system"] == SYSTEM), key=lambda r: r["id"])


def cited_labels(record):
    """Return the source labels that the answer cites, as [label]."""
    return {label for label in record["sources"] if f"[{label}]" in record["answer"]}


def answer_html(record):
    """Return the answer as HTML: escaped, **bold** kept, paragraphs kept, each citation in a muted span."""
    text = escape(record["answer"])
    for label in set(record["sources"]):
        cite = f"[{escape(label)}]"
        text = text.replace(cite, f'<span class="cite">{cite}</span>')
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return "".join(f"<p>{p.strip().replace(chr(10), '<br>')}</p>" for p in text.split("\n\n") if p.strip())


def verdict_of(record):
    """Return the kind, label and detail of the app's status line for a recorded answer."""
    cited = cited_labels(record)
    res = {**record, "hits": record["sources"],
           "cited": [i for i, label in enumerate(record["sources"], 1) if label in cited]}
    return app.verdict(res)


def scores_html(record):
    """Return the gate scores and latency as one muted line."""
    parts = []
    if record.get("best_similarity") is not None:
        parts.append(f"best similarity {record['best_similarity']:.2f}")
    if record.get("rerank_score") is not None:
        parts.append(f"cross-encoder {record['rerank_score']:.2f}")
    if record.get("latency_s") is not None:
        parts.append(f"{record['latency_s']:.1f} s")
    return f'<p class="scores">{" · ".join(parts)}</p>' if parts else ""


def sources_html(record):
    """Return the retrieved passages as an ordered list, the cited ones marked."""
    cited = cited_labels(record)
    items = "".join(f'<li{" class=cited" if label in cited else ""}><span class="source">{escape(label)}</span>'
                    f'{"<span class=cited-mark>cited</span>" if label in cited else ""}</li>'
                    for label in record["sources"])
    return f'<h4>Passages found, best first</h4><ol class="sources">{items}</ol>' if items else ""


def item_html(record, open_id):
    """Return one question as a disclosure: the question and verdict, then the answer and its passages."""
    kind, label, detail = verdict_of(record)
    return (f'<details class="item" id="{record["id"]}"{" open" if record["id"] == open_id else ""}>'
            f'<summary>{CHEVRON}<span class="qid">{record["id"]}</span>'
            f'<span class="question">{escape(record["question"])}</span>'
            f'<span class="pill pill-{kind}">{escape(label)}</span></summary>'
            f'<div class="body"><p class="detail">{escape(detail)}</p>'
            f'<div class="answer">{answer_html(record)}</div>{scores_html(record)}{sources_html(record)}</div>'
            "</details>")


def group_html(prefix, title, note, records, open_id):
    """Return one question set with its heading, its note and how many the system answered."""
    rows = [r for r in records if r["id"].startswith(prefix)]
    answered = sum(not r["refused"] for r in rows)
    tally = (f"{answered} of {len(rows)} answered" if prefix == "A"
             else f"{len(rows) - answered} of {len(rows)} refused")
    return (f'<section class="group" aria-labelledby="g-{prefix}"><div class="group-head">'
            f'<h2 id="g-{prefix}">{title} <span class="range">{prefix}01–{prefix}{len(rows):02d}</span></h2>'
            f'<p class="tally">{tally}</p></div><p class="note">{note}</p>'
            f'{"".join(item_html(r, open_id) for r in rows)}</section>')


def build(records):
    """Return the whole page."""
    model = records[0]["model"]
    answerable = [r for r in records if r["answerable"]]
    outside = [r for r in records if not r["answerable"]]
    lede = (f"Of the {len(answerable)} questions the documents answer, the system answered "
            f"{sum(not r['refused'] for r in answerable)}. It refused "
            f"{sum(r['refused'] for r in outside)} of the {len(outside)} questions whose answer is not in them.")
    groups = "".join(group_html(p, t, n, records, "A06") for p, t, n in GROUPS)
    return PAGE.format(css=CSS, lede=escape(lede), model=escape(model), groups=groups, repo=REPO,
                       disclaimer=escape(app.DISCLAIMER.split(" Questions are sent")[0]))


CSS = """
:root {
  --ink: #16262B; --deep: #0E3B43; --muted: #5E6E73; --line: #D6DFE0; --mist: #F3F6F6; --paper: #FFFFFF;
  --teal: #0F7C80; --teal-tint: #E4F1F1;
  --green: #1F6A46; --green-tint: #E3F2EA; --red: #A3312A; --red-tint: #FAE6E4;
  --amber: #7A4E0C; --amber-tint: #FBF0DC;
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  color-scheme: light;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink); caret-color: var(--teal); accent-color: var(--teal);
  font: 400 1rem/1.6 "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; }
::selection { background: #BFE0E1; color: var(--ink); }
:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; border-radius: 4px; }
a { color: var(--teal); text-underline-offset: 3px; text-decoration-thickness: 1px; }
a:hover { color: var(--deep); }
main { max-width: 820px; margin: 0 auto; padding: 56px 24px 72px; }

header h1 { font-size: clamp(1.75rem, 1.4rem + 1.6vw, 2.25rem); line-height: 1.15; letter-spacing: -0.015em;
  color: var(--deep); margin: 0 0 16px; text-wrap: balance; }
.lede { font-size: 1.125rem; line-height: 1.55; max-width: 62ch; margin: 0 0 12px; text-wrap: pretty; }
.meta, .disclaimer { color: var(--muted); font-size: 0.875rem; line-height: 1.5; max-width: 72ch; margin: 0 0 6px; }
.meta code { font: 500 0.8125rem/1 "IBM Plex Mono", ui-monospace, monospace; color: var(--ink); }

.group { margin-top: 56px; }
.group-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 16px;
  border-bottom: 1px solid var(--ink); padding-bottom: 8px; }
.group h2 { font-size: 1.25rem; line-height: 1.3; margin: 0; color: var(--deep); }
.range { font-weight: 400; color: var(--muted); font-size: 0.9375rem; margin-left: 6px;
  font-variant-numeric: tabular-nums; }
.tally { margin: 0; font-weight: 600; font-size: 0.9375rem; font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 0.875rem; margin: 8px 0 8px; max-width: 72ch; }

.item { border-bottom: 1px solid var(--line); }
.item summary { display: grid; grid-template-columns: 16px 3rem 1fr auto; align-items: baseline; gap: 4px 12px;
  padding: 14px 4px; cursor: pointer; list-style: none; border-radius: 4px;
  transition: background-color 140ms ease; }
.item summary::-webkit-details-marker { display: none; }
@media (hover: hover) and (pointer: fine) { .item summary:hover { background: var(--mist); } }
.chevron { color: var(--muted); align-self: start; margin-top: 0.2em; transition: transform 160ms var(--ease-out); }
.item[open] .chevron { transform: rotate(90deg); }
.qid { color: var(--muted); font-size: 0.875rem; font-variant-numeric: tabular-nums; }
.question { font-weight: 500; line-height: 1.45; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px 2px 8px; border-radius: 999px;
  font-size: 0.8125rem; font-weight: 600; white-space: nowrap; }
.pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill-answered { background: var(--green-tint); color: var(--green); }
.pill-refused { background: var(--red-tint); color: var(--red); }
.pill-uncited { background: var(--amber-tint); color: var(--amber); }

.body { padding: 0 4px 24px calc(16px + 3rem + 24px); }
.detail { margin: 0 0 12px; color: var(--muted); font-size: 0.9375rem; }
.answer { max-width: 68ch; font-size: 1.0625rem; line-height: 1.65; }
.answer p { margin: 0 0 0.75em; }
.cite { color: var(--muted); font-size: 0.8125em; word-break: break-all; }
.scores { color: var(--muted); font-size: 0.8125rem; margin: 4px 0 16px; font-variant-numeric: tabular-nums; }
.body h4 { font-size: 0.8125rem; font-weight: 600; color: var(--deep); margin: 0 0 6px; }
.sources { margin: 0; padding-left: 1.5em; font-size: 0.875rem; line-height: 1.45; font-variant-numeric: tabular-nums; }
.sources li { padding: 3px 0; color: var(--muted); }
.sources li::marker { color: var(--muted); }
.sources li.cited { color: var(--ink); }
.source { overflow-wrap: anywhere; }
.cited-mark { display: inline-block; margin-left: 8px; padding: 0 7px; border-radius: 999px; background: var(--green-tint);
  color: var(--green); font-size: 0.75rem; font-weight: 600; }

footer { margin-top: 64px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted);
  font-size: 0.875rem; }
footer p { margin: 0 0 6px; max-width: 72ch; }

@media (max-width: 640px) {
  main { padding: 32px 16px 56px; }
  .item summary { grid-template-columns: 16px 1fr; padding: 12px 0; }
  .qid { grid-column: 2; }
  .question, .item summary .pill { grid-column: 2; justify-self: start; }
  .body { padding: 0 0 20px 28px; }
}
@media (prefers-reduced-motion: reduce) {
  .chevron { transition: none; }
}
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medical RAG answers</title>
<meta name="description" content="Recorded answers of a retrieval-augmented system over 20 WHO guidelines and 30 PubMed Central articles: cited answers, and refusals when the documents do not hold the answer.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{css}</style>
</head>
<body>
<main>
<header>
<h1>Questions on medical guidelines</h1>
<p class="lede">{lede}</p>
<p class="meta">These are the answers recorded during the evaluation, with hybrid retrieval and the cross-encoder gate, from <code>{model}</code>. The page calls no model.</p>
<p class="disclaimer">{disclaimer}</p>
</header>
{groups}
<footer>
<p>Student project, ING5 Generative AI. Source code, report and the app that answers live: <a href="{repo}">{repo}</a>.</p>
<p>The documents themselves are not published here, only their names.</p>
</footer>
</main>
</body>
</html>
"""

if __name__ == "__main__":
    out = ROOT / "site" / "public" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = load_records()
    out.write_text(build(records))
    print(f"wrote {out.relative_to(ROOT)}: {len(records)} questions")
