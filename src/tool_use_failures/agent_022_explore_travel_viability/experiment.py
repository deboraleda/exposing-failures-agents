# experiment.py
#
# Usage:
#   python -m tool_use_failures.agent_022_explore_travel_viability.experiment --model gpt-4o

import argparse
import csv
import json
import os
import re
import unicodedata
from datetime import datetime

from dotenv import load_dotenv

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)
from tool_use_failures.llm_service import get_llm
from tool_use_failures.agent_022_explore_travel_viability.deepeval import (
    DEEPEVAL_COLUMNS,
    eval_task_completion,
    summarize_scores,
)
from tool_use_failures.agent_022_explore_travel_viability.test_cases import TEST_CASES
from tool_use_failures.agent_022_explore_travel_viability.agent import agent as create_agent

load_dotenv(override=True)


# ============================================================================
# Helpers
# ============================================================================

def _tokens(text):
    """Lowercase, accent-free word tokens (e.g. "São Paulo" -> ["sao", "paulo"])."""

    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()

    return re.findall(r"[a-z0-9]+", normalized.lower())


def evaluate_args(expected_args, tool_calls_info, expected_country=None):
    """Check that each expected tool received the expected city.

    Only the city (the part before the comma) is compared, because the test
    cases mix "City" and "City, Country" and the parameter names differ
    (location / city_name). The free-text search_query is not checked.
    When expected_country is given (disambiguation cases), it must also appear
    in the arguments of at least one call.
    Returns None when there is nothing to check.
    """

    if not expected_args:
        return None

    for tool_name, expected in expected_args.items():

        call = next((c for c in tool_calls_info if c["name"] == tool_name), None)

        if call is None:
            return False

        expected_place = next(
            (value for key, value in expected.items() if key != "search_query"),
            "",
        )
        expected_city = expected_place.split(",")[0]

        actual_tokens = set(_tokens(" ".join(str(v) for v in call["args"].values())))

        if not set(_tokens(expected_city)) <= actual_tokens:
            return False

    if expected_country:
        all_tokens = set(_tokens(" ".join(
            str(value) for call in tool_calls_info for value in call["args"].values()
        )))

        if not set(_tokens(expected_country)) <= all_tokens:
            return False

    return True


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

            called_tools = [call["name"] for call in tool_calls_info]

            n_tool_calls = len(called_tools)

            unexpected_tools = sorted(set(called_tools) - set(acceptable))

            missing_tools = [name for name in acceptable if name not in called_tools]

            # deterministic evaluation: every acceptable tool must be called;
            # no acceptable tools means no tool should be called
            tool_correct = not missing_tools and not unexpected_tools

            # each tool must be called exactly once
            tool_count_ok = n_tool_calls == case["expected_tool_count"]

            # arguments are only checked when all the expected tools were called
            args_correct = None

            if tool_correct and acceptable:
                args_correct = evaluate_args(
                    case["expected_args"],
                    tool_calls_info,
                    case.get("expected_country"),
                )

            failure_type = None

            if not tool_correct:
                if not tool_called:
                    failure_type = "omission"
                elif not acceptable:
                    failure_type = "overcall"
                elif unexpected_tools:
                    failure_type = "wrong_tool"
                else:
                    # only part of the required tools was called
                    failure_type = "partial_omission"

            elif not tool_count_ok:
                # some tool was called more than once
                failure_type = "retry"

            elif args_correct is False:
                failure_type = "wrong_args"

            # correct = all expected tools, each once, with correct arguments
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
                f"called_tools={called_tools} | "
                f"missing_tools={missing_tools} | "
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
                "called_tools": called_tools,
                "missing_tools": missing_tools,
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
        "expected_country": case.get("expected_country"),
        "failure_target": case.get("failure_target"),
        "success_rate": round(success_rate, 2),
        "runs": case_runs,
    })

    print(
        f"[{case['type']:44}] "
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
    f"results_022_{safe_model}_{timestamp}.json"
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
            "expected_country": case_data["expected_country"],
            "failure_target": case_data["failure_target"],
            "case_success_rate": case_data["success_rate"],

            # ── run metadata ────────────────────────────────────────────────
            "run": run_data.get("run"),

            # ── deterministic evaluation ───────────────────────────────────
            "tool_called": run_data.get("tool_called"),
            "called_tools": json.dumps(run_data.get("called_tools", []), ensure_ascii=False),
            "missing_tools": json.dumps(run_data.get("missing_tools", []), ensure_ascii=False),
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
