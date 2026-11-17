# experiment.py — agent018 (recipe generator), parameter error pattern
#
# Run:
#   python -m tool_use_failures.agent_018_recipe_generator.experiment --model <model key>

import argparse
import json
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from tool_use_failures.llm_service import get_llm
from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)

from tool_use_failures.agent_018_recipe_generator.recipe_agent import create_agent
from tool_use_failures.agent_018_recipe_generator.test_cases import TEST_CASES
from tool_use_failures.agent_018_recipe_generator.deepeval import (
    eval_task_completion,
)

load_dotenv(override=True)


# ============================================================================
# Argument evaluation
# ============================================================================
def same_value(expected_value, actual_value):
    """
    Lists of strings (e.g. ingredients) are compared as sets, ignoring order
    and letter case; any other value must match exactly.
    """

    if (
        isinstance(expected_value, list)
        and isinstance(actual_value, list)
        and all(isinstance(v, str) for v in expected_value + actual_value)
    ):
        return (
            {v.strip().lower() for v in expected_value}
            == {v.strip().lower() for v in actual_value}
        )

    return actual_value == expected_value


def evaluate_args(expected_args, actual_args):
    """
    Compare expected vs actual tool arguments.

    Rules:
    - expected value != None:
        required argument with exact match

    - expected value == None:
        optional argument:
            - may be absent
            - may exist with value None
            - any non-None value is error
    """

    result = {
        "all_correct": True,
        "score": 0.0,

        "missing_args": [],
        "wrong_values": [],
        "unexpected_args": [],
        "correct_args": [],

        "correct_args_count": 0,
        "total_expected_args": 0,
    }

    correct_count = 0
    total_checks = 0

    for key, expected_value in expected_args.items():

        # Required argument
        if expected_value is not None:

            total_checks += 1

            if key not in actual_args:
                result["missing_args"].append(key)
                result["all_correct"] = False
                continue

            actual_value = actual_args[key]

            if not same_value(expected_value, actual_value):
                result["wrong_values"].append({
                    "arg": key,
                    "expected": expected_value,
                    "actual": actual_value,
                })
                result["all_correct"] = False
                continue

            result["correct_args"].append(key)
            correct_count += 1

        # Optional argument
        else:

            if key not in actual_args or actual_args[key] is None:
                continue

            result["wrong_values"].append({
                "arg": key,
                "expected": None,
                "actual": actual_args[key],
            })
            result["all_correct"] = False

    # Extra unexpected args
    for key in actual_args:
        if key not in expected_args:
            result["unexpected_args"].append(key)
            result["all_correct"] = False

    result["correct_args_count"] = correct_count
    result["total_expected_args"] = total_checks

    if total_checks > 0:
        result["score"] = round(correct_count / total_checks, 4)

    return result


def parse_args_payload(raw_tool_args):
    """Tool args as a dict (JSON strings are decoded)."""

    if isinstance(raw_tool_args, dict):
        return raw_tool_args

    if isinstance(raw_tool_args, str):
        try:
            return json.loads(raw_tool_args)
        except Exception:
            return {"__raw_string__": raw_tool_args}

    return {"__invalid_payload_type__": str(type(raw_tool_args))}


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

    print("\n--------------------------------------------------")
    print(f"CASE: {case['id']}")
    print(f"TYPE: {case['type']}")
    print(f"QUERY: {case['query']}")

    expected_tool_name = case["expected_tool"]
    expected_args = (case.get("expected_args") or {}).get(expected_tool_name)
    max_tool_calls = case.get("max_tool_calls")

    case_runs = []
    successes = 0

    for run_idx in range(N_RUNS):

        print(f"\n  run={run_idx}")

        raw_tool_args = {}
        actual_args = {}

        try:

            # ================================================================
            # Execute agent
            # ================================================================

            result = agent(case["query"])

            tool_called = result["tool_called"]
            tool_calls_info = result["tool_calls_info"]
            raw_output = result["final_content"] or ""

            malformed_tool_call = (
                "<function=" in raw_output
                or "(function=" in raw_output
            )

            correct = False
            failure_type = None
            argument_evaluation = None

            # ================================================================
            # No tool call
            # ================================================================

            if not tool_called or len(tool_calls_info) == 0:

                failure_type = "omission"

            # ================================================================
            # Tool exists
            # ================================================================

            else:

                actual_tool = tool_calls_info[0]
                actual_tool_name = actual_tool.get("name")
                raw_tool_args = actual_tool.get("args", {})
                actual_args = parse_args_payload(raw_tool_args)

                # Wrong tool
                if actual_tool_name != expected_tool_name:

                    failure_type = "tool_selection_error"

                # Evaluate args (only cases that define expected_args)
                else:

                    correct = True

                    if expected_args is not None:

                        argument_evaluation = evaluate_args(
                            expected_args=expected_args,
                            actual_args=actual_args,
                        )

                        if not argument_evaluation["all_correct"]:
                            correct = False
                            failure_type = "tool_parameter_error"

                    # More tool calls than the case allows
                    if (
                        correct
                        and max_tool_calls is not None
                        and len(tool_calls_info) > max_tool_calls
                    ):
                        correct = False
                        failure_type = "multiple_tool_calls"

            # ================================================================
            # Malformed tool call overrides
            # ================================================================

            if malformed_tool_call:
                correct = False
                failure_type = "malformed_tool_call"

            if correct:
                successes += 1

            # ================================================================
            # DeepEval
            # ================================================================

            try:
                reason, score = eval_task_completion(
                    actual_input=case["query"],
                    actual_output=raw_output,
                )
            except Exception as deepeval_error:
                print(f"DeepEval ERROR: {deepeval_error}")
                reason = None
                score = None

            # ================================================================
            # Console log
            # ================================================================

            print(f"    tool_called={tool_called}")
            print(f"    correct={correct}")
            print(f"    failure_type={failure_type}")
            print(f"    deepeval={score}")
            print(f"    n_tool_calls={len(tool_calls_info)}")

            if tool_calls_info:
                print("\n    TOOL CALLS:")
                for idx, tool_call in enumerate(tool_calls_info):
                    print(f"      [{idx}] tool={tool_call.get('name')}")
                    print(json.dumps(tool_call.get("args", {}), indent=8, ensure_ascii=False))

            if argument_evaluation:
                print(f"    arg_score={argument_evaluation['score']}")
                for key in ("correct_args", "missing_args", "wrong_values", "unexpected_args"):
                    if argument_evaluation[key]:
                        print(f"    {key}={argument_evaluation[key]}")

            # ================================================================
            # Save run
            # ================================================================

            case_runs.append({
                "run": run_idx,

                # Tool execution
                "tool_called": tool_called,
                "tool_calls_info": tool_calls_info,
                "n_tool_calls": len(tool_calls_info),
                "multiple_tool_calls": len(tool_calls_info) > 1,
                "raw_tool_args": raw_tool_args,
                "parsed_payload": actual_args,
                "malformed_tool_call": malformed_tool_call,

                # Evaluation
                "correct": correct,
                "failure_type": failure_type,
                "argument_evaluation": argument_evaluation,

                # Outputs
                "final_content": raw_output,

                # DeepEval
                "deepeval_score": score,
                "deepeval_reason": reason,
            })

        except Exception as e:

            print(f"    ERROR: {e}")

            case_runs.append({
                "run": run_idx,
                "error": str(e),
                "correct": False,
                "failure_type": "execution_error",
                "raw_tool_args": raw_tool_args,
                "parsed_payload": actual_args,
            })

    # ========================================================================
    # Case summary
    # ========================================================================

    success_rate = round((successes / N_RUNS) * 100, 2)

    print(f"\n  success_rate={success_rate}%")

    model_results["cases"].append({
        "case_id": case["id"],
        "input_type": case["type"],
        "query": case["query"],
        "expected_tool": expected_tool_name,
        "expected_args": expected_args,
        "max_tool_calls": max_tool_calls,
        "success_rate": success_rate,
        "runs": case_runs,
    })


# ============================================================================
# Flat CSV rows
# ============================================================================

for case_data in model_results["cases"]:

    for run_data in case_data["runs"]:

        arg_eval = run_data.get("argument_evaluation") or {}

        csv_rows.append({

            # Model
            "provider": provider,
            "model": model_name,
            "temperature": TEMPERATURE,

            # Case
            "case_id": case_data["case_id"],
            "input_type": case_data["input_type"],
            "query": case_data["query"],

            # Expected
            "expected_tool": case_data["expected_tool"],
            "expected_args": (
                json.dumps(case_data["expected_args"], ensure_ascii=False)
                if case_data["expected_args"] else None
            ),
            "max_tool_calls": case_data["max_tool_calls"],

            # Run
            "run": run_data["run"],

            # Tool execution
            "tool_called": run_data.get("tool_called"),
            "tool_calls_info": json.dumps(run_data.get("tool_calls_info", []), ensure_ascii=False),
            "n_tool_calls": run_data.get("n_tool_calls"),
            "multiple_tool_calls": run_data.get("multiple_tool_calls"),
            "raw_tool_args": json.dumps(run_data.get("raw_tool_args", {}), ensure_ascii=False),
            "parsed_payload": json.dumps(run_data.get("parsed_payload", {}), ensure_ascii=False),
            "malformed_tool_call": run_data.get("malformed_tool_call"),

            # Deterministic evaluation
            "deterministic_correct": run_data.get("correct"),
            "failure_type": run_data.get("failure_type"),

            # Argument evaluation
            "argument_score": arg_eval.get("score"),
            "correct_args": json.dumps(arg_eval.get("correct_args", []), ensure_ascii=False),
            "missing_args": json.dumps(arg_eval.get("missing_args", []), ensure_ascii=False),
            "wrong_values": json.dumps(arg_eval.get("wrong_values", []), ensure_ascii=False),
            "unexpected_args": json.dumps(arg_eval.get("unexpected_args", []), ensure_ascii=False),
            "correct_args_count": arg_eval.get("correct_args_count"),
            "total_expected_args": arg_eval.get("total_expected_args"),

            # DeepEval
            "deepeval_score": run_data.get("deepeval_score"),
            "deepeval_reason": run_data.get("deepeval_reason"),

            # Final response
            "final_content": run_data.get("final_content"),

            # Error
            "error": run_data.get("error"),
        })


# ============================================================================
# Save outputs
# ============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

safe_model_name = model_name.replace("/", "_").replace(":", "-")

json_output_file = f"results_agent018_{safe_model_name}_{timestamp}.json"

with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(model_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved JSON -> {json_output_file}")

csv_output_file = f"results_agent018_{safe_model_name}_{timestamp}.csv"

pd.DataFrame(csv_rows).to_csv(csv_output_file, index=False)

print(f"Saved CSV -> {csv_output_file}")
