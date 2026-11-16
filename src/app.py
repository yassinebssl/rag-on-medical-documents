"""Gradio web app over the RAG pipeline: ask a question, read the cited answer and its sources.

Launch with .venv/bin/python src/app.py, then open http://127.0.0.1:7860.
"""
from html import escape

import gradio as gr

import rag
from rag import load_context

DISCLAIMER = ("This tool is a student project. It is not medical advice. Check every answer with a clinician. "
              "Questions are sent to OpenRouter: do not type anything that identifies a patient.")
DENSE, HYBRID, RERANKED = "Dense", "Hybrid (dense + BM25)", "Hybrid + cross-encoder gate"
MODES = {DENSE: rag.SIM_THRESHOLD, HYBRID: rag.SIM_THRESHOLD, RERANKED: rag.RERANK_THRESHOLD}  # default threshold
DEFAULT_MODE = RERANKED
NOT_IN_PASSAGES = "The retrieved passages do not contain the answer."
REASONS = {"below_threshold": "Best similarity {best:.2f} is below the threshold.",
           "below_rerank_threshold": "The cross-encoder rates the best passage {rerank:.2f}, below the threshold.",
           "uncited": "The model's answer did not cite the retrieved passages.",
           "model_refused": NOT_IN_PASSAGES,
           "no_context": NOT_IN_PASSAGES}
# reason -> (kind, label) of the status line; the kind picks its colour in CSS
VERDICTS = {"answered": ("answered", "Answered"),
            "below_threshold": ("refused", "Refused by the gate"),
            "below_rerank_threshold": ("refused", "Refused by the gate"),
            "model_refused": ("refused", "Refused by the model"),
            "no_context": ("refused", "Refused"),
            "uncited": ("uncited", "Uncited, refused")}
EXAMPLES = ["How long should parenteral artesunate be given to a patient with severe malaria?",
            "When should antiretroviral therapy be started in a person living with HIV who begins tuberculosis treatment?",
            "What is the first-line topical treatment for scabies?"]
EXAMPLE_LABELS = ["Artesunate in severe malaria", "Starting HIV therapy during tuberculosis treatment",
                  "First-line treatment for scabies"]  # Gradio cuts a label at 60 characters
HEADERS = ["#", "Source", "Page", "Similarity", "Cited"]
INTRO = ("Answers come from 50 documents: 20 WHO guidelines and 30 PubMed Central articles. "
         "Each claim cites its source, and a question the documents do not cover is refused.")

# Colours of the defense slides. Refusal red and uncited amber are darkened so that small text passes 4.5:1.
CSS = """
.gradio-container {
  --ink: #16262B; --deep: #0E3B43; --muted: #5E6E73; --line: #D6DFE0; --edge: #7C8D92; --mist: #F3F6F6;
  --teal: #0F7C80; --teal-tint: #E4F1F1;
  --green: #1F6A46; --green-tint: #E3F2EA; --red: #A3312A; --red-tint: #FAE6E4;
  --amber: #7A4E0C; --amber-tint: #FBF0DC;
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  max-width: 960px !important; padding: 40px clamp(16px, 4vw, 24px) 56px !important;
  caret-color: var(--teal); accent-color: var(--teal);
}
.gradio-container .main.fillable { padding-inline: 0 !important; }  /* the container alone sets the gutter */
.gradio-container ::selection { background: #BFE0E1; color: var(--ink); }
.gradio-container :focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }

.gradio-container .block.page-title, .gradio-container .block.page-note, .gradio-container .block.answer {
  overflow: visible !important; }  /* Gradio sets overflow: auto inline, which draws scroll arrows on the title */
.form { background: transparent; border: 0; box-shadow: none; }
.page-title h1 { font-size: 1.75rem; line-height: 1.2; letter-spacing: -0.01em; margin: 0; color: var(--deep);
  text-wrap: balance; }
.page-note p { color: var(--muted); font-size: 0.875rem; line-height: 1.5; max-width: 72ch; margin: 0; }

.ask-row { align-items: flex-end; gap: 12px; margin-top: 12px; }
.ask-row textarea { font-size: 1.0625rem; }
.ask-button { height: 46px; transition: transform 140ms var(--ease-out), background-color 140ms ease; }
.ask-button:active:not(:disabled) { transform: scale(0.97); }
.ask-button:disabled { opacity: 1; background: var(--teal-tint); color: var(--deep); border-color: var(--teal-tint);
  cursor: progress; }

#examples .label { color: var(--muted); font-size: 0.8125rem; }
#examples button.gallery-item { border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink);
  font-size: 0.875rem; text-align: left; transition: border-color 140ms ease, background-color 140ms ease; }
@media (hover: hover) and (pointer: fine) {
  #examples button.gallery-item:hover { border-color: var(--teal); background: var(--teal-tint); }
}

.settings { background: var(--mist); border-radius: 8px; padding: 16px; gap: 16px; }
.settings .wrap:has(> .wrap-inner) { border-color: var(--edge); }
.settings .form { gap: 16px 24px; }  /* Gradio packs the model and the threshold into one gapless form */

.block.verdict-slot { margin-top: 12px; padding-top: 20px; border-top: 1px solid var(--line); }
.verdict { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; font-size: 0.9375rem;
  line-height: 1.5; color: var(--ink); animation: verdict-in 180ms var(--ease-out); }
@keyframes verdict-in { from { opacity: 0; transform: translateY(4px); } }  /* Gradio drops @starting-style */
@keyframes verdict-fade { from { opacity: 0; } }
.verdict-pill { display: inline-flex; align-items: center; gap: 8px; padding: 3px 12px 3px 10px;
  border-radius: 999px; font-weight: 600; white-space: nowrap; }
.verdict-pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.verdict-answered .verdict-pill { background: var(--green-tint); color: var(--green); }
.verdict-refused .verdict-pill { background: var(--red-tint); color: var(--red); }
.verdict-uncited .verdict-pill { background: var(--amber-tint); color: var(--amber); }
.verdict-error .verdict-pill { background: var(--ink); color: #fff; }
.verdict-pending .verdict-pill { background: var(--teal-tint); color: var(--deep); }
.verdict-pending .verdict-pill::before { animation: pulse 900ms ease-in-out infinite alternate; }
.verdict-intro { color: var(--muted); max-width: 72ch; }
@keyframes pulse { from { opacity: 1; } to { opacity: 0.25; } }

.answer { max-width: 72ch; font-size: 1.0625rem; line-height: 1.6; color: var(--ink); }
.answer p { margin: 0 0 0.75em; }
.answer .cite { color: var(--muted); font-size: 0.875em; word-break: break-all; }  /* a file name has no break point */
.answer p:last-child > em:only-child { font-style: normal; font-size: 0.8125rem; line-height: 1.5; color: var(--muted); }

.sources-wrap { overflow-x: auto; }
.sources-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; line-height: 1.4; color: var(--ink);
  font-variant-numeric: tabular-nums; }
.sources-table caption { text-align: left; font-weight: 600; color: var(--deep); padding: 0 0 8px; }
.gradio-container .prose .sources-table, .gradio-container .prose .sources-table :is(tr, th, td) { border: 0; }
.gradio-container .prose .sources-table th { text-align: left; font-weight: 600; font-size: 0.8125rem;
  color: var(--muted); background: transparent; padding: 8px 12px; border-bottom: 1px solid var(--line);
  white-space: nowrap; }
.gradio-container .prose .sources-table td { padding: 10px 12px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
.sources-table td.source { overflow-wrap: anywhere; }
.gradio-container .prose .sources-table .num { text-align: right; white-space: nowrap; }
.sources-table tr.cited td { background: #F6FBF8; }
.sources-table tr.cited td.source { font-weight: 600; }
.cited-mark { display: inline-block; padding: 1px 8px; border-radius: 999px; background: var(--green-tint);
  color: var(--green); font-weight: 600; font-size: 0.8125rem; }

/* Gradio prefixes the selectors inside @media and splits them at commas: no .gradio-container and no :is() here */
@media (max-width: 640px) {
  .sources-table th, .sources-table td { padding: 8px 6px; }
  .sources-table th:first-child, .sources-table td:first-child { display: none; }  /* the caption gives the order */
  .sources-table td.source { min-width: 9rem; }
}
@media (prefers-reduced-motion: reduce) {
  .verdict { animation-name: verdict-fade; }
  .ask-button:active:not(:disabled) { transform: none; }
  .verdict-pending .verdict-pill::before { animation: none; }
}
"""


def build_theme():
    """Return the page's theme: the slide colours, the font Gradio ships, and no dark variant."""
    teal = gr.themes.Color(c50="#F0F7F7", c100="#E4F1F1", c200="#C3E0E1", c300="#94C6C8", c400="#4FA3A6",
                           c500="#0F7C80", c600="#0C686C", c700="#0E535A", c800="#0E3B43", c900="#0B2E35",
                           c950="#16262B")
    grey = gr.themes.Color(c50="#F7F9F9", c100="#F3F6F6", c200="#E6ECEC", c300="#D6DFE0", c400="#A3B1B4",
                           c500="#5E6E73", c600="#4B5A5F", c700="#37474C", c800="#26363B", c900="#16262B",
                           c950="#0E1B1F")
    theme = gr.themes.Base(primary_hue=teal, secondary_hue=teal, neutral_hue=grey,
                           radius_size=gr.themes.sizes.radius_sm).set(
        body_background_fill="#FFFFFF", body_text_color="#16262B", body_text_color_subdued="#5E6E73",
        block_border_width="0px", block_shadow="none", block_padding="0px", block_background_fill="transparent",
        block_label_text_color="#5E6E73", block_title_text_color="#37474C", block_title_text_weight="600",
        block_info_text_color="#5E6E73", layout_gap="16px",
        input_background_fill="#FFFFFF", input_border_color="#7C8D92", input_border_color_hover="#5E6E73",
        input_border_color_focus="#0F7C80", input_border_width="1px", input_shadow="none",
        input_shadow_focus="0 0 0 3px #E4F1F1", input_placeholder_color="#5E6E73",
        button_primary_background_fill="#0F7C80", button_primary_background_fill_hover="#0C686C",
        button_primary_text_color="#FFFFFF", button_primary_border_color="#0F7C80",
        button_primary_border_color_hover="#0C686C", button_primary_shadow="none",
        button_large_text_weight="600", button_medium_text_weight="600",
        slider_color="#0F7C80", loader_color="#0F7C80", color_accent="#0F7C80", color_accent_soft="#E4F1F1",
        checkbox_label_background_fill="#FFFFFF", checkbox_label_background_fill_hover="#F7F9F9",
        checkbox_label_background_fill_selected="#E4F1F1", checkbox_label_border_color="#D6DFE0",
        checkbox_label_border_color_selected="#0F7C80", checkbox_label_border_width="1px",
        checkbox_label_text_color="#16262B", checkbox_label_text_color_selected="#0E3B43",
        checkbox_border_color="#7C8D92", checkbox_border_color_selected="#0F7C80",
        checkbox_background_color_selected="#0F7C80", checkbox_label_shadow="none")
    for name in dir(theme):  # the page is read on a projector and on paper-white screens: one light look
        if name.endswith("_dark") and hasattr(theme, name[:-5]):
            setattr(theme, name, getattr(theme, name[:-5]))
    return theme


def verdict(res):
    """Return the kind, the label and the detail of the status line for a pipeline result."""
    kind, label = VERDICTS[res["reason"]]
    if res["refused"]:
        return kind, label, REASONS[res["reason"]].format(best=res.get("best_similarity"),
                                                          rerank=res.get("rerank_score"))
    return kind, label, f"{len(res['cited'])} of {len(res['hits'])} passages cited."


def verdict_html(kind, label, detail):
    """Return the status line: a coloured label (none for a plain notice) and one sentence."""
    pill = f'<span class="verdict-pill">{escape(label)}</span>' if label else ""
    return (f'<div class="verdict verdict-{kind}" role="status">{pill}'
            f'<span class="verdict-detail">{escape(detail)}</span></div>')


def sources_html(rows):
    """Return the retrieved passages as an HTML table, or nothing when there is none."""
    if not rows:
        return ""
    head = "".join(f'<th scope="col"{" class=num" if h in ("#", "Page", "Similarity") else ""}>{h}</th>'
                   for h in HEADERS)
    body = ""
    for rank, source, page, similarity, cited in rows:
        mark = '<span class="cited-mark">yes</span>' if cited else ""
        body += (f'<tr{" class=cited" if cited else ""}><td class="num">{rank}</td>'
                 f'<td class="source">{escape(source)}</td><td class="num">{escape(page)}</td>'
                 f'<td class="num">{escape(similarity)}</td><td>{mark}</td></tr>')
    return ('<div class="sources-wrap"><table class="sources-table"><caption>Passages found, best first</caption>'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def format_result(res):
    """Return the answer as Markdown (with the disclaimer) and the source rows."""
    rows = [[rank, hit["source"], "-" if hit["page"] is None else str(hit["page"]),
             "-" if hit["similarity"] is None else f"{hit['similarity']:.3f}",
             "yes" if rank in res["cited"] else ""]
            for rank, hit in enumerate(res["hits"], 1)]
    return f"{res['answer']}\n\n*{DISCLAIMER}*", rows


def mark_citations(answer, hits):
    """Return the answer with each [source, p. N] wrapped in a span, so the page can set it smaller and muted."""
    for label in {rag.source_label(hit) for hit in hits}:
        answer = answer.replace(f"[{label}]", f'<span class="cite">[{escape(label)}]</span>')
    return answer


def respond(question, model, mode, threshold, ctx):
    """Answer one question from the UI: return the status line, the answer and the sources table.

    An empty question or a failed LLM call becomes a status line with no answer.
    """
    if not question.strip():
        return verdict_html("intro", "", "Type a question first."), "", ""
    bm25 = ctx["bm25"] if mode in (HYBRID, RERANKED) else None
    reranker = ctx["reranker"] if mode == RERANKED else None
    try:  # the slider is the threshold of whichever gate the mode uses
        res = rag.answer_question(question, ctx["embedder"], ctx["collection"], model, ctx["api_key"],
                                  threshold=threshold, bm25=bm25, reranker=reranker, rerank_threshold=threshold)
    except Exception as exc:  # network, rate limit or empty reply: show it instead of crashing
        detail = f"The model call failed: {exc}. Ask again, or choose another model."
        return verdict_html("error", "Not answered", detail), "", ""
    answer, rows = format_result(res)
    return verdict_html(*verdict(res)), mark_citations(answer, res["hits"]), sources_html(rows)


def build_app(ctx):
    """Build the Gradio page; the Ask button and Enter in the question box both call respond."""
    with gr.Blocks(title="Medical RAG") as demo:
        gr.Markdown("# Questions on medical guidelines", elem_classes="page-title")
        gr.Markdown(DISCLAIMER, elem_classes="page-note")
        with gr.Row(elem_classes="ask-row"):
            question = gr.Textbox(label="Question", placeholder="Ask in English, as one full sentence",
                                  lines=1, max_lines=4, autofocus=True, scale=1)
            ask = gr.Button("Ask", variant="primary", scale=0, min_width=128, elem_classes="ask-button")
        gr.Examples(EXAMPLES, inputs=question, example_labels=EXAMPLE_LABELS, elem_id="examples")
        with gr.Column(elem_classes="settings"):
            mode = gr.Radio(list(MODES), value=DEFAULT_MODE, label="Retrieval and refusal gate")
            with gr.Row():
                model = gr.Dropdown(rag.FREE_MODELS, value=rag.DEFAULT_MODEL, label="Model")
                threshold = gr.Slider(0.0, 1.0, value=MODES[DEFAULT_MODE], step=0.05, label="Refusal threshold")
        status = gr.HTML(verdict_html("intro", "", INTRO), elem_classes="verdict-slot")
        answer = gr.Markdown(elem_classes="answer")
        sources = gr.HTML(elem_classes="sources")

        def searching(q):
            """Clear the last answer and lock the button while the question runs; each answer costs a request."""
            if not q.strip():
                return gr.update(), gr.update(), "", ""
            return (gr.update(value="Asking…", interactive=False),
                    verdict_html("pending", "Searching", "Reading the 50 documents."), "", "")

        def ask_question(q, m, md, t):
            """Call respond with the loaded context and unlock the button."""
            return (gr.update(value="Ask", interactive=True), *respond(q, m, md, t, ctx))

        outputs = [ask, status, answer, sources]
        mode.change(lambda m: MODES[m], mode, threshold)  # each gate has its own default threshold
        gr.on([ask.click, question.submit], searching, question, outputs, show_progress="hidden", queue=False).then(
            ask_question, [question, model, mode, threshold], outputs, show_progress="hidden")
    return demo


if __name__ == "__main__":
    build_app(load_context()).launch(server_name="127.0.0.1", server_port=7860, theme=build_theme(), css=CSS)
