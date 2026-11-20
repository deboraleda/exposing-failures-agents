# rejudge.py
#
# Re-scores, with the LLM judges, only the runs whose judge returned an error
# (rate limit, server error...). The agent is NOT executed again: the judge
# sees the same query and final answer stored in the result file.
#
# A new result file is written next to the original, with a new timestamp, so
# the dashboard picks it up as the most recent one. The original is kept.
#
# Usage (from src/):
#   python -m tool_use_failures.rejudge <result_file.json> [<result_file.json> ...]

import json
import os
import sys
import time
from datetime import datetime

from tool_use_failures.judge_service import TASK_COMPLETION_METRICS
from deepeval.test_case import LLMTestCase

MAX_ATTEMPTS = 3
WAIT_SECONDS = 20


def measure(metric, query, answer):
    """Score one answer with one judge, waiting and retrying on errors."""

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            metric.measure(LLMTestCase(input=query, actual_output=answer))
            return metric.score, metric.reason, None
        except Exception as e:
            last_error = str(e)
            print(f"    attempt {attempt} failed: {last_error[:120]}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(WAIT_SECONDS * attempt)

    return None, None, last_error


def rejudge(path):

    results = json.load(open(path, encoding="utf-8"))
    fixed = 0
    still_failing = 0

    for case in results["cases"]:

        query = case.get("query")
        cid = case.get("case_id") or case.get("id")

        for run in case["runs"]:

            # crashed runs have no answer to judge: they must be executed again
            if run.get("failure_type") == "execution_error":
                continue

            for key, metric in TASK_COMPLETION_METRICS.items():

                if not run.get(f"deepeval_{key}_error"):
                    continue

                print(f"{cid} run={run.get('run')} judge={key}")
                score, reason, error = measure(metric, query, run.get("final_content") or "")

                run[f"deepeval_{key}_score"] = score
                run[f"deepeval_{key}_reason"] = reason
                run[f"deepeval_{key}_error"] = error

                if error:
                    still_failing += 1
                else:
                    fixed += 1
                    print(f"    score={score}")

    if not fixed:
        print(f"\nNothing re-scored in {path} (still failing: {still_failing})")
        return

    # same name with a new timestamp: <prefix><model>_YYYYMMDD_HHMM.json
    stem = os.path.basename(path)[: -len("YYYYMMDD_HHMM.json")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(os.path.dirname(path), f"{stem}{timestamp}.json")

    if os.path.abspath(output_file) == os.path.abspath(path):
        output_file = output_file.replace(".json", "_rejudged.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRe-scored: {fixed} | still failing: {still_failing}")
    print(f"Saved → {output_file}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit("usage: python -m tool_use_failures.rejudge <result_file.json> [...]")

    for result_path in sys.argv[1:]:
        print(f"\n=== {result_path} ===")
        rejudge(result_path)
