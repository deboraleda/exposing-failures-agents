# experiment.py
#
# Usage:
#   python -m tool_use_failures.agent_014_academic_triage_agent.experiment --model gpt-4o

import argparse
import csv
import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)
from tool_use_failures.llm_service import get_llm
from tool_use_failures.agent_014_academic_triage_agent.deepeval import (
    DEEPEVAL_COLUMNS,
    eval_task_completion,
    summarize_scores,
)
from tool_use_failures.agent_014_academic_triage_agent.test_cases import TEST_CASES
from tool_use_failures.agent_014_academic_triage_agent.agent import agent as create_agent

load_dotenv(override=True)


# ============================================================================
# Helpers
# ============================================================================

def _tokens(text):
    """Lowercase alphanumeric tokens with more than two characters."""

    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 2]


def evaluate_args(expected_args, actual_args):
    """Check the arguments of the chosen tool against the test case.

    The tools use different parameter names (research_query / concept), so the
    expected values are compared against all argument values, regardless of
    the key. Every significant token of the expected value must appear in the
    actual arguments (e.g. "predict" matches "prediction").
    """

    if not expected_args:
        return None

    actual_text = " ".join(str(value) for value in actual_args.values()).lower()

    return all(
        token in actual_text
        for expected_value in expected_args.values()
        for token in _tokens(expected_value)
    )


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

            acceptable = case["acceptable_tools"]

            first_tool = tool_calls_info[0]["name"] if tool_calls_info else None

            # deterministic evaluation: any acceptable tool is correct;
            # no acceptable tools means no tool should be called
            if acceptable:
                tool_correct = first_tool in acceptable
            else:
                tool_correct = not tool_called

            called_tools = [call["name"] for call in tool_calls_info]

            n_tool_calls = len(called_tools)

            # at most one tool call is allowed per execution
            tool_count_ok = n_tool_calls <= case["expected_tool_count"]

            # arguments are only checked when an acceptable tool was chosen
            args_correct = None

            if tool_correct and acceptable:
                args_correct = evaluate_args(
                    case["expected_args"],
                    tool_calls_info[0]["args"],
                )

            failure_type = None

            if not tool_correct:
                if not tool_called:
                    failure_type = "omission"
                elif not acceptable:
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
            deepeval_results = eval_task_completion(
                actual_input=case["query"],
                actual_output=final_content,
            )

            print(
                f"run={run_idx} | "
                f"acceptable={acceptable} | "
                f"first_tool={first_tool} | "
                f"tool_correct={tool_correct} | "
                f"correct={correct} | "
                f"n_tool_calls={n_tool_calls} | "
                f"args_correct={args_correct} | "
                f"failure={failure_type} | "
                f"deepeval=[{summarize_scores(deepeval_results)}]"
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
                **deepeval_results,
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
        "acceptable_tools": case["acceptable_tools"],
        "expected_tool_count": case["expected_tool_count"],
        "expected_args": case["expected_args"],
        "success_rate": round(success_rate, 2),
        "runs": case_runs,
    })

    print(
        f"[{case['type']:26}] "
        f"acceptable={case['acceptable_tools']} "
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
    f"results_014_{safe_model}_{timestamp}.json"
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
            "acceptable_tools": json.dumps(case_data["acceptable_tools"], ensure_ascii=False),
            "expected_tool_count": case_data["expected_tool_count"],
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
            **{column: run_data.get(column) for column in DEEPEVAL_COLUMNS},

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
