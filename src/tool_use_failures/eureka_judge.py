# eureka_judge.py
#
# Scores the Eureca Chat runs with the same LLM judges used for the nine
# agents (DeepEval TaskCompletion, see judge_service.py). The agent is NOT
# executed again: each judge sees the stored question and final response,
# exactly the input the judges received in the main experiment.
#
# Writes one line per run to eureka_chat/judge_evaluation_agent_experiment_<model>.jsonl.
# Runs already scored without errors are skipped, so the script can be
# re-run after an interruption or to retry judge errors.
#
# Usage (from src/):
#   python -m tool_use_failures.eureka_judge [deepseek GLM gpt-luna]

import json
import os
import sys

from tool_use_failures.judge_service import TASK_COMPLETION_METRICS
from tool_use_failures.rejudge import measure

EUREKA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eureka_chat")
SUFFIXES = ["deepseek", "GLM", "gpt-luna"]


def run_key(row):
    return row["experiment_id"], row["run_number"]


def load_done(path):
    """Runs already scored by every judge without errors."""

    if not os.path.exists(path):
        return {}

    done = {}

    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if all(row.get(f"deepeval_{key}_error") is None and row.get(f"deepeval_{key}_score") is not None
               for key in TASK_COMPLETION_METRICS):
            done[run_key(row)] = row

    return done


def judge_model(suffix):

    agent_path = os.path.join(EUREKA_DIR, f"agent_experiment_{suffix}.jsonl")
    output_path = os.path.join(EUREKA_DIR, f"judge_evaluation_agent_experiment_{suffix}.jsonl")

    done = load_done(output_path)
    rows = []

    for line in open(agent_path, encoding="utf-8"):
        run = json.loads(line)
        key = run_key(run)

        if key in done:
            rows.append(done[key])
            continue

        question = run["question"]
        answer = (run.get("backend") or {}).get("final_response") or ""

        row = {
            "experiment_id": run["experiment_id"],
            "run_number": run["run_number"],
            "agent_model_configured": run["agent_model_configured"],
            "repetition": run["repetition"],
            "question_id": run["question_id"],
        }

        print(f"{suffix} {run['question_id']} rep={run['repetition']}")

        for judge, metric in TASK_COMPLETION_METRICS.items():
            score, reason, error = measure(metric, question, answer)
            row[f"deepeval_{judge}_score"] = score
            row[f"deepeval_{judge}_reason"] = reason
            row[f"deepeval_{judge}_error"] = error
            print(f"    {judge}: {score if error is None else 'ERROR ' + error[:80]}")

        rows.append(row)

        # rewrite after every run, so an interruption loses at most one run
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    errors = sum(any(r.get(f"deepeval_{k}_error") for k in TASK_COMPLETION_METRICS) for r in rows)
    print(f"\n{suffix}: {len(rows)} runs, {len(done)} already scored, {errors} with judge errors")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    for model_suffix in sys.argv[1:] or SUFFIXES:
        judge_model(model_suffix)
