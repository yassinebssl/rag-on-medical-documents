# The question app on a free Hugging Face Space (Docker SDK, CPU basic).
# The image holds the code and the two public models. The index holds the documents' text, so it is not
# in the image: start.py downloads it at each start from a private dataset repository (INDEX_REPO),
# with the HF_TOKEN secret. The Space also needs the OPENROUTER_API_KEY secret.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
ENV HOME=/home/user PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/app/.hf PYTHONUNBUFFERED=1

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

USER user
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

COPY --chown=user src/ src/
COPY --chown=user start.py .

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "start.py"]
