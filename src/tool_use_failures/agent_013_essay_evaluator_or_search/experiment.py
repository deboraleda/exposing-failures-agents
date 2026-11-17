# experiment.py
#
# Usage:
#   python -m tool_use_failures.agent_013_essay_evaluator_or_search.experiment --model gpt-4o

import argparse
import csv
import json
import os
from datetime import datetime

from dotenv import load_dotenv

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)
from tool_use_failures.llm_service import get_llm
from tool_use_failures.agent_013_essay_evaluator_or_search.deepeval import eval_task_completion
from tool_use_failures.agent_013_essay_evaluator_or_search.test_cases import TEST_CASES
from tool_use_failures.agent_013_essay_evaluator_or_search.agent import agent as create_agent

load_dotenv(override=True)


# ============================================================================
# Helpers
# ============================================================================

def evaluate_args(expected_args, actual_args):
    """Check the arguments of the expected tool call against the test case."""

    if "url" in expected_args:
        return actual_args.get("url", "").strip() == expected_args["url"]

    if "topic_keywords" in expected_args:
        topic = str(actual_args.get("topic", "")).lower()
        return any(keyword.lower() in topic for keyword in expected_args["topic_keywords"])

    return None


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


# ============================================================================
# Run experiment
# ============================================================================

for case in TEST_CASES:

    print("\n" + "=" * 80)
    print(f"CASE: {case['id']} [{case['type']}]")
    print(repr(case["query"]))

    case_runs = []
    successes = 0

    for run_idx in range(N_RUNS):

        try:
            result = agent(case["query"])

            tool_called = result["tool_called"]
            tool_calls_info = result["tool_calls_info"]
            final_content = result["final_content"]

            expected = case["expected_tool"]

            first_tool = tool_calls_info[0]["name"] if tool_calls_info else None

            # deterministic evaluation
            tool_correct = (first_tool == expected)

            called_tools = [call["name"] for call in tool_calls_info]

            n_tool_calls = len(called_tools)

            # only one tool call is allowed per execution
            tool_count_ok = n_tool_calls <= 1

            # arguments are only checked when the expected tool was chosen
            args_correct = None

            if tool_correct and expected is not None:
                args_correct = evaluate_args(
                    case["expected_args"],
                    tool_calls_info[0]["args"],
                )

            failure_type = None

            if not tool_correct:
                if not tool_called:
                    failure_type = "omission"
                elif expected is None:
                    failure_type = "overcall"
                else:
                    failure_type = "wrong_tool"

            elif not tool_count_ok:
                # same tool again -> retry; another tool -> fallback
                if called_tools[1] == called_tools[0]:
                    failure_type = "retry"
                else:
                    failure_type = "fallback"

            elif args_correct is False:
                failure_type = "wrong_args"

            # correct = right tool, a single call and correct arguments
            correct = failure_type is None

            if correct:
                successes += 1

            # ── DeepEval ──────────────────────────────────────────────────────
            deepeval_score = None
            deepeval_reason = None
            deepeval_error = None

            try:
                deepeval_reason, deepeval_score = eval_task_completion(
                    actual_input=case["query"],
                    actual_output=final_content,
                )

            except Exception as deepeval_exception:
                deepeval_error = str(deepeval_exception)

                print(f"DeepEval ERROR: {deepeval_error}")

            print(
                f"run={run_idx} | "
                f"expected={expected} | "
                f"first_tool={first_tool} | "
                f"tool_correct={tool_correct} | "
                f"correct={correct} | "
                f"n_tool_calls={n_tool_calls} | "
                f"args_correct={args_correct} | "
                f"failure={failure_type} | "
                f"deepeval={deepeval_score}"
            )

            case_runs.append({
                "run": run_idx,

                # deterministic evaluation
                "tool_called": tool_called,
                "tool_calls_info": tool_calls_info,
                "first_tool": first_tool,
                "tool_correct": tool_correct,
                "n_tool_calls": n_tool_calls,
                "tool_count_ok": tool_count_ok,
                "args_correct": args_correct,
                "correct": correct,
                "failure_type": failure_type,

                # response
                "final_content": final_content,

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
        "expected_tool": case["expected_tool"],
        "expected_args": case["expected_args"],
        "success_rate": round(success_rate, 2),
        "runs": case_runs,
    })

    print(
        f"[{case['type']:16}] "
        f"expected={case['expected_tool']} "
        f"success_rate={success_rate:.1f}%"
    )


# ============================================================================
# Save JSON
# ============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

safe_model = (
    model_name
    .replace("/", "_")
    .replace(":", "-")
)

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

output_file = os.path.join(
    results_dir,
    f"results_013_{safe_model}_{timestamp}.json"
)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(model_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved → {output_file}")


# ============================================================================
# Save CSV
# ============================================================================

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
            "expected_args": json.dumps(case_data["expected_args"], ensure_ascii=False),
            "case_success_rate": case_data["success_rate"],

            # ── run metadata ────────────────────────────────────────────────
            "run": run_data.get("run"),

            # ── deterministic evaluation ───────────────────────────────────
            "tool_called": run_data.get("tool_called"),
            "first_tool": run_data.get("first_tool"),
            "tool_correct": run_data.get("tool_correct"),
            "n_tool_calls": run_data.get("n_tool_calls"),
            "tool_count_ok": run_data.get("tool_count_ok"),
            "args_correct": run_data.get("args_correct"),
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
