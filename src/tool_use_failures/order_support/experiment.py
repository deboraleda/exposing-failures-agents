# experiment_t2.py
import argparse
import json
from datetime import datetime
import os
import csv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from tool_use_failures.config import (
    N_RUNS,
    TEMPERATURE,
    PROVIDER_CONFIG,
    MODELS,
)
from tool_use_failures.llm_service import get_llm

from tool_use_failures.order_support.agent import create_t2_agent
from tool_use_failures.order_support.test_cases import TEST_CASES


from tool_use_failures.order_support.deepeval import (
    DEEPEVAL_COLUMNS,
    eval_task_completion,
    summarize_scores,
)

load_dotenv(override=True)

CLIENT_LOOKUPS = {"get_client", "search_client"}
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

agent = create_t2_agent(llm)


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

    for run_idx in range(N_RUNS):

        try:
            result = agent(case["query"])

            tool_calls_info = result["tool_calls_info"]
            tool_called = result["tool_called"]

            expected = case["expected_tool"]

            first_tool = tool_calls_info[0]["name"] if tool_calls_info else None

            # client lookups before the action are allowed (identifying the
            # customer first), unless the lookup itself is the expected tool
            action_tool = next(
                (call["name"] for call in tool_calls_info
                 if call["name"] in expected or call["name"] not in CLIENT_LOOKUPS),
                None,
            )

            # deterministic evaluation
            correct = (action_tool in expected)

            if correct:
                successes += 1

            failure_type = None

            if not correct:
                if not tool_called:
                    failure_type = "omission"
                else:
                    failure_type = "wrong_tool"

            # ── DeepEval ──────────────────────────────────────────────────────
            deepeval_results = eval_task_completion(
                actual_input=case["query"],
                actual_output=result["final_content"],
            )

            print(
                f"run={run_idx} | "
                f"expected={expected} | "
                f"first_tool={first_tool} | "
                f"action_tool={action_tool} | "
                f"correct={correct} | "
                f"deepeval=[{summarize_scores(deepeval_results)}]"
            )

            case_runs.append({
                "run": run_idx,

                # deterministic evaluation
                "tool_called": tool_called,
                "tool_calls_info": tool_calls_info,
                "first_tool": first_tool,
                "action_tool": action_tool,
                "correct": correct,
                "failure_type": failure_type,

                # response
                "final_content": result["final_content"],

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
        "expected_tool": case["expected_tool"],
        "success_rate": round(success_rate, 2),
        "runs": case_runs,
    })

    print(
        f"[{case['type']:10}] "
        f"expected={case['expected_tool']} "
        f"success_rate={success_rate:.1f}%"
    )

# ── Save ──────────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

safe_model = (
    model_name
    .replace("/", "_")
    .replace(":", "-")
)

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

output_file = os.path.join(results_dir, f"results_t2_{safe_model}_{timestamp}.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(model_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved → {output_file}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
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
            "case_success_rate": case_data["success_rate"],

            # ── run metadata ────────────────────────────────────────────────
            "run": run_data.get("run"),

            # ── deterministic evaluation ───────────────────────────────────
            "tool_called": run_data.get("tool_called"),
            "first_tool": run_data.get("first_tool"),
            "action_tool": run_data.get("action_tool"),
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