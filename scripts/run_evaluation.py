"""Run one day's share of the full evaluation, then print how much is left.

Run from the repo root, once a day: .venv/bin/python scripts/run_evaluation.py

OpenRouter gives this account 50 free-model requests per UTC day, so the full run takes
several days. src/evaluate.py resumes from results/records.jsonl, and this script sends
nothing once the day's requests are used up. The output is also appended to
results/logs/<UTC date>.log.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import evaluate  # noqa: E402
import rag  # noqa: E402

# (systems, model) in the order they are run: every system with the default model, the
# main system first, then the main system alone with a second model.
STEPS = [("rag_rerank,rag_hybrid,no_rag,bm25_rag,rag_dense", rag.FREE_MODELS[0]),
         ("rag_rerank", rag.FREE_MODELS[1])]


def missing(records, question_ids, steps=STEPS):
    """Return the (system, model, question id) triples of the steps that have no record yet."""
    done = {(r["system"], r["model"], r["id"]) for r in records}
    return [(system, model, qid) for systems, model in steps for system in systems.split(",")
            for qid in question_ids if (system, model, qid) not in done]


def requests_left(api_key):
    """Return the free-model requests left today; asking OpenRouter for it uses none of them."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["data"]["free_model_daily_requests"]["remaining"]


def main():
    """Run each unfinished step while requests are left, then print what is still missing."""
    os.makedirs(os.path.join(evaluate.RESULTS, "logs"), exist_ok=True)
    lock = open(os.path.join(evaluate.RESULTS, ".run.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("an evaluation run is already in progress")
    api_key = rag.load_env().get("OPENROUTER_API_KEY")
    ids = [q["id"] for q in evaluate.load_questions()]
    records_path = os.path.join(evaluate.RESULTS, "records.jsonl")
    with open(os.path.join(evaluate.RESULTS, "logs", time.strftime("%Y-%m-%d.log", time.gmtime())), "a",
              encoding="utf-8") as log:

        def say(line):
            """Print the line and append it to the day's log."""
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        for systems, model in STEPS:
            if not missing(evaluate.load_records(records_path), ids, [(systems, model)]):
                continue
            left = requests_left(api_key)
            say(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {left} free requests left today")
            if left <= 0:
                break
            cmd = [sys.executable, os.path.join(ROOT, "src", "evaluate.py"), "run", "--judge",
                   "--systems", systems, "--models", model]
            say(" ".join(cmd[1:]))
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                say(line.rstrip())
            if proc.wait():
                raise SystemExit(f"evaluate.py stopped with exit code {proc.returncode}")
        todo = missing(evaluate.load_records(records_path), ids)
        total = len(missing([], ids))
        say("evaluation complete" if not todo else f"{len(todo)} of {total} answers still to record")


if __name__ == "__main__":
    main()
