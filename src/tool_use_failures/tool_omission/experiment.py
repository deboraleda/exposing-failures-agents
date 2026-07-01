# experiment.py

import argparse
import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)

from tool_use_failures.llm_service import get_llm
from tool_use_failures.tool_omission.agent import create_t1_agent
from tool_use_failures.tool_omission.test_cases import TEST_CASES
from tool_use_failures.tool_omission.deepeval_eval import (
    eval_task_completion,
)

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    help="Model key to run (e.g. gpt-4o, qwen)"
)

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

target = args.model.lower()

llm, provider, model_name = get_llm(target)

print(f"\n=== Running {provider}/{model_name} ===")
print(f"Temperature: {TEMPERATURE}")
print(f"Runs per case: {N_RUNS}")

agent = create_t1_agent(llm)

# ─────────────────────────────────────────────────────────────────────────────
# Output structures
# ─────────────────────────────────────────────────────────────────────────────

model_results = {
    "provider": provider,
    "model": model_name,
    "temperature": TEMPERATURE,
    "n_runs": N_RUNS,
    "cases": []
}

csv_rows = []

# ─────────────────────────────────────────────────────────────────────────────
# Run experiment
# ─────────────────────────────────────────────────────────────────────────────

for case in TEST_CASES:

    print("\n" + "=" * 80)
    print(f"CASE: {case['id']}")
    print(case["query"])

    case_runs = []
    successes = 0

    for run_idx in range(N_RUNS):

        try:

            result = agent(case["query"])

            tool_called = result["tool_called"]

            tool_calls_info = result["tool_calls_info"]

            expected = case["should_call_tool"]

            correct = (tool_called == expected)

            if correct:
                successes += 1

            # ─────────────────────────────────────────────────────────────
            # Failure type
            # ─────────────────────────────────────────────────────────────

            failure_type = None

            if not correct:

                if expected and not tool_called:
                    failure_type = "omission"

                elif not expected and tool_called:
                    failure_type = "overcall"

            # ─────────────────────────────────────────────────────────────
            # DeepEval
            # ─────────────────────────────────────────────────────────────

            try:

                reason, score = eval_task_completion(
                    case["query"],
                    result["final_content"]
                )

            except Exception as deepeval_error:

                print(f"DeepEval ERROR: {deepeval_error}")

                reason = None
                score = None

            # ─────────────────────────────────────────────────────────────
            # Logging
            # ─────────────────────────────────────────────────────────────

            print(
                f"run={run_idx} | "
                f"expected={expected} | "
                f"tool_called={tool_called} | "
                f"correct={correct} | "
                f"deepeval={score}"
            )

            # ─────────────────────────────────────────────────────────────
            # JSON structure
            # ─────────────────────────────────────────────────────────────

            run_data = {
                "run": run_idx,

                "tool_called": tool_called,

                "tool_calls_info": tool_calls_info,

                "correct": correct,

                "failure_type": failure_type,

                "final_content": result["final_content"],

                "deepeval_score": score,

                "deepeval_reason": reason,
            }

            case_runs.append(run_data)

            # ─────────────────────────────────────────────────────────────
            # CSV row
            # ─────────────────────────────────────────────────────────────

            csv_rows.append({

                # model info
                "provider": provider,
                "model": model_name,
                "temperature": TEMPERATURE,

                # case info
                "case_id": case["id"],
                "case_type": case["type"],
                "query": case["query"],

                # execution info
                "run": run_idx,

                # deterministic evaluation
                "expected_tool_call": expected,
                "actual_tool_call": tool_called,
                "deterministic_correct": correct,
                "failure_type": failure_type,

                # outputs
                "tool_calls_info": json.dumps(
                    tool_calls_info,
                    ensure_ascii=False
                ),

                "final_content": result["final_content"],

                # deepeval
                "deepeval_score": score,
                "deepeval_reason": reason,
            })

        except Exception as e:

            print(f"run={run_idx} | ERROR: {e}")

            error_data = {
                "run": run_idx,
                "error": str(e),
                "correct": False,
                "failure_type": "execution_error",
            }

            case_runs.append(error_data)

            csv_rows.append({

                "provider": provider,
                "model": model_name,
                "temperature": TEMPERATURE,

                "case_id": case["id"],
                "case_type": case["type"],
                "query": case["query"],

                "run": run_idx,

                "expected_tool_call": case["should_call_tool"],
                "actual_tool_call": None,

                "deterministic_correct": False,

                "failure_type": "execution_error",

                "tool_calls_info": None,

                "final_content": None,

                "deepeval_score": None,
                "deepeval_reason": None,

                "error": str(e),
            })

    # ─────────────────────────────────────────────────────────────────────
    # Aggregate per-case stats
    # ─────────────────────────────────────────────────────────────────────

    success_rate = (successes / N_RUNS) * 100

    print(f"SUCCESS RATE: {success_rate:.1f}%")

    model_results["cases"].append({

        "case_id": case["id"],

        "input_type": case["type"],

        "query": case["query"],

        "should_call_tool": case["should_call_tool"],

        "success_rate": round(success_rate, 2),

        "runs": case_runs,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

safe_model_name = (
    model_name
    .replace("/", "_")
    .replace(":", "-")
)

# JSON
json_output_file = (
    f"results_{safe_model_name}_{timestamp}.json"
)

with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(
        model_results,
        f,
        indent=2,
        ensure_ascii=False
    )

# CSV
csv_output_file = (
    f"results_{safe_model_name}_{timestamp}.csv"
)

df = pd.DataFrame(csv_rows)

df.to_csv(
    csv_output_file,
    index=False
)

# ─────────────────────────────────────────────────────────────────────────────
# Final
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("Finished.")
print(f"JSON saved → {json_output_file}")
print(f"CSV saved  → {csv_output_file}")