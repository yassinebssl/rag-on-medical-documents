"""Gradio web app over the RAG pipeline: ask a question, read the cited answer and its sources.

Launch with .venv/bin/python src/app.py, then open http://127.0.0.1:7860.
"""
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
EXAMPLES = ["How long should parenteral artesunate be given to a patient with severe malaria?",
            "When should antiretroviral therapy be started in a person living with HIV who begins tuberculosis treatment?",
            "What is the first-line topical treatment for scabies?"]


def format_result(res):
    """Return the answer as Markdown (with the refusal reason and the disclaimer) and the source rows."""
    rows = [[rank, hit["source"], "-" if hit["page"] is None else str(hit["page"]),
             "-" if hit["similarity"] is None else f"{hit['similarity']:.3f}",
             "yes" if rank in res["cited"] else ""]
            for rank, hit in enumerate(res["hits"], 1)]
    answer = res["answer"]
    if res["refused"]:
        answer += "\n\n" + REASONS[res["reason"]].format(best=res["best_similarity"], rerank=res.get("rerank_score"))
    return f"{answer}\n\n*{DISCLAIMER}*", rows


def respond(question, model, mode, threshold, ctx):
    """Answer one question from the UI; an empty question or a failed LLM call becomes a message."""
    if not question.strip():
        return "Type a question first.", []
    bm25 = ctx["bm25"] if mode in (HYBRID, RERANKED) else None
    reranker = ctx["reranker"] if mode == RERANKED else None
    try:  # the slider is the threshold of whichever gate the mode uses
        res = rag.answer_question(question, ctx["embedder"], ctx["collection"], model, ctx["api_key"],
                                  threshold=threshold, bm25=bm25, reranker=reranker, rerank_threshold=threshold)
    except Exception as exc:  # network, rate limit or empty reply: show it instead of crashing
        return f"The model call failed: {exc}", []
    return format_result(res)


def build_app(ctx):
    """Build the Gradio page; the Ask button and Enter in the question box both call respond."""
    with gr.Blocks(title="Medical RAG") as demo:
        gr.Markdown("# Questions on medical guidelines")
        gr.Markdown(DISCLAIMER)
        question = gr.Textbox(label="Question")
        with gr.Row():
            model = gr.Dropdown(rag.FREE_MODELS, value=rag.DEFAULT_MODEL, label="Model")
            mode = gr.Radio(list(MODES), value=DEFAULT_MODE, label="Retrieval and refusal gate")
            threshold = gr.Slider(0.0, 1.0, value=MODES[DEFAULT_MODE], step=0.05, label="Refusal threshold")
        ask = gr.Button("Ask", variant="primary")
        answer = gr.Markdown()
        sources = gr.Dataframe(headers=["#", "Source", "Page", "Similarity", "Cited"], interactive=False)
        gr.Examples(EXAMPLES, inputs=question)

        def ask_question(q, m, md, t):
            """Call respond with the loaded context."""
            return respond(q, m, md, t, ctx)

        mode.change(lambda m: MODES[m], mode, threshold)  # each gate has its own default threshold
        gr.on([ask.click, question.submit], ask_question, [question, model, mode, threshold], [answer, sources])
    return demo


if __name__ == "__main__":
    build_app(load_context()).launch(server_name="127.0.0.1", server_port=7860)
