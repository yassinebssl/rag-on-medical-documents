---
title: RAG on Medical Documents
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers from 20 WHO guidelines and 30 PMC articles
---

Ask questions of 20 WHO guidelines and 30 PubMed Central open-access articles. Each answer cites the passages it comes from, and a cross-encoder gate refuses a question the documents do not cover before any model is called. Follow-up questions use the earlier turns.

A student project, not medical advice. Questions are sent to OpenRouter's free models: do not type anything that identifies a patient.

Source code, report and evaluation: https://github.com/yassinebssl/rag-on-medical-documents
