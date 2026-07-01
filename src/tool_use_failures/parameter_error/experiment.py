# experiment.py

import argparse
import json
from datetime import datetime

import pandas as pd

from dotenv import load_dotenv

from tool_use_failures.llm_service import get_llm
from tool_use_failures.parameter_error.agent import agent
from tool_use_failures.parameter_error.test_cases import TEST_CASES
from tool_use_failures.parameter_error.deepeval import (
    eval_task_completion,
)

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
)

load_dotenv(override=True)


# ============================================================================
# Argument evaluation
# ============================================================================
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

    # =====================================================================
    # Expected args
    # =====================================================================

    for key, expected_value in expected_args.items():

        # ---------------------------------------------------------------
        # Required argument
        # ---------------------------------------------------------------

        if expected_value is not None:

            total_checks += 1

            if key not in actual_args:

                result["missing_args"].append(key)
                result["all_correct"] = False
                continue

            actual_value = actual_args[key]

            if actual_value != expected_value:

                result["wrong_values"].append({
                    "arg": key,
                    "expected": expected_value,
                    "actual": actual_value,
                })

                result["all_correct"] = False
                continue

            result["correct_args"].append(key)
            correct_count += 1

        # ---------------------------------------------------------------
        # Optional argument
        # ---------------------------------------------------------------

        else:

            # absent -> OK
            if key not in actual_args:
                continue

            # explicit None -> OK
            if actual_args[key] is None:
                continue

            # non-None value -> wrong
            result["wrong_values"].append({
                "arg": key,
                "expected": None,
                "actual": actual_args[key],
            })

            result["all_correct"] = False

    # =====================================================================
    # Extra unexpected args
    # =====================================================================

    for key in actual_args:

        if key not in expected_args:

            result["unexpected_args"].append(key)
            result["all_correct"] = False

    # =====================================================================
    # Score
    # =====================================================================

    result["correct_args_count"] = correct_count
    result["total_expected_args"] = total_checks

    if total_checks > 0:

        result["score"] = round(
            correct_count / total_checks,
            4
        )

    return result


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

agent = agent(llm)


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

for case in TEST_CASES[0]["cases"]:

    print("\n--------------------------------------------------")

    print(f"CASE: {case['case_id']}")

    print(f"TYPE: {case['input_type']}")

    print(f"QUERY: {case['query']}")

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

            tool_called = result.get(
                "tool_called",
                False
            )

            tool_calls_info = result.get(
                "tool_calls_info",
                []
            )

            raw_output = (
                result.get("final_content", "")
                or ""
            )

            malformed_tool_call = (

                "<function=" in raw_output
                or "(function=" in raw_output
            )

            expected_tool_call = (
                case["should_call_tool"]
            )

            correct = False

            failure_type = None

            argument_evaluation = None

            # ================================================================
            # Tool expected
            # ================================================================

            if expected_tool_call:

                # ------------------------------------------------------------
                # No tool call
                # ------------------------------------------------------------

                if (

                    not tool_called
                    or len(tool_calls_info) == 0
                ):

                    correct = False

                    failure_type = "omission"

                # ------------------------------------------------------------
                # Tool exists
                # ------------------------------------------------------------

                else:

                    actual_tool = tool_calls_info[0]

                    actual_tool_name = actual_tool.get(
                        "name"
                    )

                    raw_tool_args = actual_tool.get(
                        "args",
                        {}
                    )

                    payload_candidate = raw_tool_args.get(
                        "payload",
                        raw_tool_args
                    )

                    # --------------------------------------------------------
                    # Parse payload safely
                    # --------------------------------------------------------

                    if isinstance(payload_candidate, str):

                        try:

                            actual_args = json.loads(
                                payload_candidate
                            )

                        except Exception:

                            actual_args = {

                                "__raw_string__": (
                                    payload_candidate
                                )
                            }

                    elif isinstance(payload_candidate, dict):

                        actual_args = payload_candidate

                    else:

                        actual_args = {

                            "__invalid_payload_type__": str(
                                type(payload_candidate)
                            )
                        }

                    expected_tool_name = (
                        case["expected_tool"]
                    )

                    # --------------------------------------------------------
                    # Wrong tool
                    # --------------------------------------------------------

                    if actual_tool_name != expected_tool_name:

                        correct = False

                        failure_type = (
                            "tool_selection_error"
                        )

                    # --------------------------------------------------------
                    # Evaluate args
                    # --------------------------------------------------------

                    else:

                        argument_evaluation = evaluate_args(

                            expected_args=case[
                                "expected_args"
                            ],

                            actual_args=actual_args,
                        )

                        correct = (
                            argument_evaluation[
                                "all_correct"
                            ]
                        )

                        if not correct:

                            failure_type = (
                                "tool_parameter_error"
                            )

            # ================================================================
            # Tool NOT expected
            # ================================================================

            else:

                if (

                    tool_called
                    and len(tool_calls_info) > 0
                ):

                    correct = False

                    failure_type = "tool_overcall"

                else:

                    correct = True

            # ================================================================
            # Malformed tool call overrides
            # ================================================================

            if malformed_tool_call:

                correct = False

                failure_type = "malformed_tool_call"

            # ================================================================
            # Success tracking
            # ================================================================

            if correct:

                successes += 1

            # ================================================================
            # DeepEval
            # ================================================================

            try:

                reason, score = eval_task_completion(
                    actual_input=case["query"], actual_output=raw_output)

            except Exception as deepeval_error:

                print(
                    f"DeepEval ERROR: "
                    f"{deepeval_error}"
                )

                reason = None

                score = None

            # ================================================================
            # Console log
            # ================================================================

            print(f"    tool_called={tool_called}")

            print(f"    correct={correct}")

            print(f"    failure_type={failure_type}")

            print(f"    deepeval={score}")

            print(
                f"    n_tool_calls="
                f"{len(tool_calls_info)}"
            )

            if tool_calls_info:

                print("\n    TOOL CALLS:")

                for idx, tool_call in enumerate(
                    tool_calls_info
                ):

                    print(
                        f"      [{idx}] "
                        f"tool={tool_call.get('name')}"
                    )

                    print(
                        json.dumps(
                            tool_call.get("args", {}),
                            indent=8,
                            ensure_ascii=False,
                        )
                    )

            if argument_evaluation:

                print(
                    f"    arg_score="
                    f"{argument_evaluation['score']}"
                )

                if argument_evaluation["correct_args"]:

                    print(
                        f"    correct_args="
                        f"{argument_evaluation['correct_args']}"
                    )

                if argument_evaluation["missing_args"]:

                    print(
                        f"    missing_args="
                        f"{argument_evaluation['missing_args']}"
                    )

                if argument_evaluation["wrong_values"]:

                    print(
                        f"    wrong_values="
                        f"{argument_evaluation['wrong_values']}"
                    )

                if argument_evaluation["unexpected_args"]:

                    print(
                        f"    unexpected_args="
                        f"{argument_evaluation['unexpected_args']}"
                    )

            # ================================================================
            # Save run
            # ================================================================

            run_result = {

                # ------------------------------------------------------------
                # Metadata
                # ------------------------------------------------------------

                "run": run_idx,

                # ------------------------------------------------------------
                # Tool execution
                # ------------------------------------------------------------

                "tool_called": tool_called,

                "tool_calls_info": tool_calls_info,

                "n_tool_calls": len(tool_calls_info),

                "multiple_tool_calls": (
                    len(tool_calls_info) > 1
                ),

                "raw_tool_args": raw_tool_args,

                "parsed_payload": actual_args,

                "malformed_tool_call": (
                    malformed_tool_call
                ),

                # ------------------------------------------------------------
                # Evaluation
                # ------------------------------------------------------------

                "correct": correct,

                "failure_type": failure_type,

                "argument_evaluation": (
                    argument_evaluation
                ),

                # ------------------------------------------------------------
                # Outputs
                # ------------------------------------------------------------

                "final_content": raw_output,

                # ------------------------------------------------------------
                # DeepEval
                # ------------------------------------------------------------

                "deepeval_score": score,

                "deepeval_reason": reason,
            }

            case_runs.append(run_result)

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

    success_rate = round(

        (successes / N_RUNS) * 100,

        2
    )

    print(f"\n  success_rate={success_rate}%")

    model_results["cases"].append({

        "case_id": case["case_id"],

        "input_type": case["input_type"],

        "query": case["query"],

        "should_call_tool": (
            case["should_call_tool"]
        ),

        "expected_tool": (
            case.get("expected_tool")
        ),

        "expected_args": (
            case.get("expected_args")
        ),

        "success_rate": success_rate,

        "runs": case_runs,
    })


# ============================================================================
# Flat CSV rows
# ============================================================================

for case_data in model_results["cases"]:

    for run_data in case_data["runs"]:

        arg_eval = (

            run_data.get(
                "argument_evaluation"
            )
            or {}
        )

        csv_rows.append({

            # ================================================================
            # Model
            # ================================================================

            "provider": provider,

            "model": model_name,

            "temperature": TEMPERATURE,

            # ================================================================
            # Case
            # ================================================================

            "case_id": case_data["case_id"],

            "input_type": (
                case_data["input_type"]
            ),

            "query": case_data["query"],

            # ================================================================
            # Expected
            # ================================================================

            "should_call_tool": (
                case_data["should_call_tool"]
            ),

            "expected_tool": (
                case_data["expected_tool"]
            ),

            "expected_args": json.dumps(

                case_data["expected_args"],

                ensure_ascii=False
            )
            if case_data["expected_args"]
            else None,

            # ================================================================
            # Run
            # ================================================================

            "run": run_data["run"],

            # ================================================================
            # Tool execution
            # ================================================================

            "tool_called": (
                run_data.get("tool_called")
            ),

            "tool_calls_info": json.dumps(

                run_data.get(
                    "tool_calls_info",
                    []
                ),

                ensure_ascii=False
            ),

            "n_tool_calls": (
                run_data.get("n_tool_calls")
            ),

            "multiple_tool_calls": (
                run_data.get(
                    "multiple_tool_calls"
                )
            ),

            "raw_tool_args": json.dumps(

                run_data.get(
                    "raw_tool_args",
                    {}
                ),

                ensure_ascii=False
            ),

            "parsed_payload": json.dumps(

                run_data.get(
                    "parsed_payload",
                    {}
                ),

                ensure_ascii=False
            ),

            "malformed_tool_call": (
                run_data.get(
                    "malformed_tool_call"
                )
            ),

            # ================================================================
            # Deterministic evaluation
            # ================================================================

            "deterministic_correct": (
                run_data.get("correct")
            ),

            "failure_type": (
                run_data.get("failure_type")
            ),

            # ================================================================
            # Argument evaluation
            # ================================================================

            "argument_score": (
                arg_eval.get("score")
            ),

            "correct_args": json.dumps(

                arg_eval.get(
                    "correct_args",
                    []
                ),

                ensure_ascii=False
            ),

            "missing_args": json.dumps(

                arg_eval.get(
                    "missing_args",
                    []
                ),

                ensure_ascii=False
            ),

            "wrong_values": json.dumps(

                arg_eval.get(
                    "wrong_values",
                    []
                ),

                ensure_ascii=False
            ),

            "unexpected_args": json.dumps(

                arg_eval.get(
                    "unexpected_args",
                    []
                ),

                ensure_ascii=False
            ),

            "correct_args_count": (
                arg_eval.get(
                    "correct_args_count"
                )
            ),

            "total_expected_args": (
                arg_eval.get(
                    "total_expected_args"
                )
            ),

            # ================================================================
            # DeepEval
            # ================================================================

            "deepeval_score": (
                run_data.get(
                    "deepeval_score"
                )
            ),

            "deepeval_reason": (
                run_data.get(
                    "deepeval_reason"
                )
            ),

            # ================================================================
            # Final response
            # ================================================================

            "final_content": (
                run_data.get(
                    "final_content"
                )
            ),

            # ================================================================
            # Error
            # ================================================================

            "error": run_data.get("error"),
        })


# ============================================================================
# Save outputs
# ============================================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M"
)

safe_model_name = (

    model_name
    .replace("/", "_")
    .replace(":", "-")
)

# ----------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------

json_output_file = (

    f"results_parameter_error_"
    f"{safe_model_name}_"
    f"{timestamp}.json"
)

with open(

    json_output_file,

    "w",

    encoding="utf-8"
) as f:

    json.dump(

        model_results,

        f,

        indent=2,

        ensure_ascii=False,
    )

print(f"\nSaved JSON -> {json_output_file}")

# ----------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------

csv_output_file = (

    f"results_parameter_error_"
    f"{safe_model_name}_"
    f"{timestamp}.csv"
)

csv_df = pd.DataFrame(csv_rows)

csv_df.to_csv(

    csv_output_file,

    index=False,
)

print(f"Saved CSV -> {csv_output_file}")