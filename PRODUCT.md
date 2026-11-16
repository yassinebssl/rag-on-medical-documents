# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The project team, presenting the app live on a projector at the course defense, with the jury reading the screen from across the room. The team types questions, including follow-up questions, and the jury judges whether the answers are grounded, cited and fast. The app also stays online after the defense for anyone with the link.

## Product Purpose

The app answers medical questions in English from 50 open documents: 20 WHO guidelines and 30 PubMed Central articles. Every claim in an answer cites the passage it comes from, as a document name and, for WHO guidelines, a page. When the documents do not hold the answer, the app refuses with one fixed sentence instead of guessing. Success at the defense: the jury sees a cited answer, a refusal for an out-of-scope question, and how long each step took.

## Positioning

A refusal gate decides before the model writes: a cross-encoder (MiniLM, ms-marco) scores whether a retrieved passage answers the question, and below 0.5 the app refuses without calling the model. Retrieval fuses dense search (BGE-M3, ChromaDB) with a BM25 index written from scratch. The answer model is a free OpenRouter model, and an answer that cites no valid passage is refused too.

## Operating Context

- A projector at the defense: large type, high contrast, readable from the back of a room.
- Conversation memory: a follow-up question uses the previous question for retrieval and the earlier turns for context. The evaluation measured single questions; memory exists only in the app.
- Response time is shown for each answer, split into the local search and gate, and the model's reply. All local code is Python; there is no Rust in the project.
- Deployed on a free Hugging Face Space (Docker), with the corpus fetched from its original URLs at build time and the OpenRouter key stored as a Space secret. No rate limit on visitors (the team's decision); the free daily quota of the account is shared with the evaluation runs.

## Capabilities and Constraints

- Three retrieval modes: dense, hybrid (dense + BM25), hybrid + cross-encoder gate (default, threshold 0.5).
- A choice of free OpenRouter models (rag.FREE_MODELS).
- Refusal sentence, verbatim: "I cannot answer this question from the provided documents."
- Required disclaimer: a student project, not medical advice; check every answer with a clinician; questions go to OpenRouter, so no patient-identifying details.
- Failure states: model call failed (rate limit, empty reply, daily quota used up), empty question, uncited answer.
- Licences: 6 of the 30 PubMed articles are CC BY-NC-ND; the app shows document names and short generated answers, never republishes the documents.

## Brand Commitments

None binding. The project has no product name beyond "RAG on Medical Documents". The defense decks use an ink and teal palette; the app is free to differ.

## Evidence on Hand

- results/records.jsonl: recorded answers of the evaluation (132 of 240 at the time of writing).
- The static recorded-answers page: https://medical-rag-answers.fikraplus.workers.dev
- Source: https://github.com/yassinebssl/rag-on-medical-documents
- No testimonials, users, or benchmarks beyond the evaluation; none may be invented.

## Product Principles

1. Every claim shows where it comes from; a citation is part of the answer, not decoration.
2. A refusal is a correct outcome and gets the same weight on screen as an answer.
3. Show the mechanism: which passages were found, what the gate scored, how long each step took.
4. Readable from the back of the room before anything else.

## Accessibility & Inclusion

Projector use sets the floor: body text large, contrast at least 4.5:1 for all text, colour never the only carrier of a verdict.
