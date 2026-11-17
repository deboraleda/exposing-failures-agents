# experiment.py — agent012 (travel destination assistant), wrong tool selection pattern
#
# Run:
#   python -m tool_use_failures.wrong_tool_selection.agent012.experiment --model <model key>

import argparse
import csv
import json
from datetime import datetime

from dotenv import load_dotenv

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)
from tool_use_failures.llm_service import get_llm

from tool_use_failures.wrong_tool_selection.agent012.travel_agent import create_agent
from tool_use_failures.wrong_tool_selection.agent012.test_cases import TEST_CASES
from tool_use_failures.wrong_tool_selection.agent012.deepeval_eval import (
    eval_task_completion,
)

load_dotenv(override=True)


# ============================================================================
# CLI
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    help="Model key to run"
)

args = parser.parse_args()


# ============================================================================
# Load model
# ============================================================================

target = args.model.lower()

llm, provider, model_name = get_llm(target)

print(f"\n=== Running {provider}/{model_name} ===")

print(f"Temperature: {TEMPERATURE}")

print(f"Runs per case: {N_RUNS}")

agent = create_agent(llm)


# ============================================================================
# Output structures
# ============================================================================

model_results = {

    "provider": provider,

    "model": model_name,

    "temperature": TEMPERATURE,

    "n_runs": N_RUNS,

    "cases": [],
}

csv_rows = []

# ── Run cases ─────────────────────────────────────────────────────────────────
for case in TEST_CASES:

    case_runs = []
    successes = 0

    expected = case["expected_tool"]
    max_tool_calls = case.get("max_tool_calls")

    for run_idx in range(N_RUNS):

        try:
            result = agent(case["query"])

            tool_calls_info = result["tool_calls_info"]
            tool_called = result["tool_called"]

            first_tool = tool_calls_info[0]["name"] if tool_calls_info else None

            # deterministic evaluation
            correct = (first_tool in expected)

            failure_type = None

            if not correct:
                if not tool_called:
                    failure_type = "omission"
                else:
                    failure_type = "wrong_tool"

            # more tool calls than the case allows
            elif max_tool_calls is not None and len(tool_calls_info) > max_tool_calls:
                correct = False
                failure_type = "multiple_tool_calls"

            if correct:
                successes += 1

            # ── DeepEval ──────────────────────────────────────────────────────
            deepeval_score = None
            deepeval_reason = None
            deepeval_error = None

            try:
                deepeval_reason, deepeval_score = eval_task_completion(
                    actual_input=case["query"],
                    actual_output=result["final_content"],
                )

            except Exception as deepeval_exception:
                deepeval_error = str(deepeval_exception)

            print(
                f"run={run_idx} | "
                f"expected={expected} | "
                f"first_tool={first_tool} | "
                f"n_tool_calls={len(tool_calls_info)} | "
                f"correct={correct} | "
                f"deepeval={deepeval_score}"
            )

            case_runs.append({
                "run": run_idx,

                # deterministic evaluation
                "tool_called": tool_called,
                "tool_calls_info": tool_calls_info,
                "first_tool": first_tool,
                "n_tool_calls": len(tool_calls_info),
                "correct": correct,
                "failure_type": failure_type,

                # response
                "final_content": result["final_content"],

                # deepeval
                "deepeval_score": deepeval_score,
                "deepeval_reason": deepeval_reason,
                "deepeval_error": deepeval_error,
            })

        except Exception as e:

            print(f"run={run_idx} | ERROR: {e}")

            case_runs.append({
                "run": run_idx,
                "error": str(e),
                "correct": False,
                "failure_type": "execution_error",
            })

    success_rate = (successes / N_RUNS) * 100

    model_results["cases"].append({
        "case_id": case["id"],
        "input_type": case["type"],
        "query": case["query"],
        "expected_tool": expected,
        "max_tool_calls": max_tool_calls,
        "success_rate": round(success_rate, 2),
        "runs": case_runs,
    })

    print(
        f"[{case['type']:10}] "
        f"expected={expected} "
        f"success_rate={success_rate:.1f}%"
    )

# ── Save ──────────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

safe_model = (
    model_name
    .replace("/", "_")
    .replace(":", "-")
)

output_file = f"results_agent012_{safe_model}_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(model_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved → {output_file}")

# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_output_file = output_file.replace(".json", ".csv")

for case_data in model_results["cases"]:

    for run_data in case_data["runs"]:

        csv_rows.append({

            # ── model metadata ──────────────────────────────────────────────
            "provider": model_results["provider"],
            "model": model_results["model"],
            "temperature": model_results["temperature"],
            "n_runs": model_results["n_runs"],

            # ── case metadata ───────────────────────────────────────────────
            "case_id": case_data["case_id"],
            "input_type": case_data["input_type"],
            "query": case_data["query"],
            "expected_tool": case_data["expected_tool"],
            "max_tool_calls": case_data["max_tool_calls"],
            "case_success_rate": case_data["success_rate"],

            # ── run metadata ────────────────────────────────────────────────
            "run": run_data.get("run"),

            # ── deterministic evaluation ───────────────────────────────────
            "tool_called": run_data.get("tool_called"),
            "first_tool": run_data.get("first_tool"),
            "n_tool_calls": run_data.get("n_tool_calls"),
            "correct": run_data.get("correct"),
            "failure_type": run_data.get("failure_type"),

            # ── tool call details ───────────────────────────────────────────
            "tool_calls_info": json.dumps(
                run_data.get("tool_calls_info", []),
                ensure_ascii=False
            ),

            # ── response ────────────────────────────────────────────────────
            "final_content": run_data.get("final_content"),

            # ── deepeval ────────────────────────────────────────────────────
            "deepeval_score": run_data.get("deepeval_score"),
            "deepeval_reason": run_data.get("deepeval_reason"),
            "deepeval_error": run_data.get("deepeval_error"),

            # ── execution errors ────────────────────────────────────────────
            "error": run_data.get("error"),
        })

if csv_rows:

    with open(csv_output_file, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=csv_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved CSV → {csv_output_file}")

else:
    print("No CSV rows generated.")
