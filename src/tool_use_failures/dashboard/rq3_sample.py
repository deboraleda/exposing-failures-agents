# rq3_sample.py
#
# Draws the sample of misclassified runs for the manual analysis of RQ3 and
# writes one CSV per annotator, with the same runs and empty code columns.
#
#   FP (corrupt success): oracle fails, at least one judge approves
#   FN: oracle passes, at least one judge rejects
#
# Stratified by agent: FP_PER_AGENT false positives and FN_PER_AGENT false
# negatives per agent (all of them when the agent has fewer), drawn with a
# fixed seed so the sample can be reproduced. One FP and one FN per agent are
# pilot runs, coded together to build the codebook; the remaining runs are
# split between the annotators, alternating within each agent.
#
# The draw order is kept from the first version of the sample (DRAW_FP and
# DRAW_FN per agent), so the sample ids and the pilot runs do not change when
# the per-agent sizes are reduced: the sample is the first runs of that order.
#
# Usage (from src/):
#   python -m tool_use_failures.dashboard.rq3_sample

import csv
import json
import os
import random

from tool_use_failures.dashboard.build_dashboard_data import (
    AGENTS, EXCLUDED_CASES, EXECUTOR_MODELS, JUDGE_KEYS, JUDGE_LABELS, JUDGE_PASS_THRESHOLD,
    case_group, case_id_of, clean, expected_calls_text, expected_text, find_result_file,
    load_test_cases, tool_calls_text,
)

SEED = 42
FP_PER_AGENT = 4
FN_PER_AGENT = 4

# sizes of the original draw, kept so that ids and pilot runs stay stable
DRAW_FP = 15
DRAW_FN = 10
ANNOTATORS = ["A", "B"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rq3")

CODE_COLUMNS = ["agent_behavior_code", "judge_rationale_code", "notes"]


def judge_name(key):
    return JUDGE_LABELS[key].split("/")[-1]


def collect_runs():
    """Every FP and FN run, with what the annotators need to read."""

    rows = []

    for agent in AGENTS:

        if agent.get("external"):
            continue

        excluded_cases = EXCLUDED_CASES.get(agent["agent"], set())
        cases_by_id = {case_id_of(c): c for c in load_test_cases(agent["package"])}

        for model in EXECUTOR_MODELS:

            path = find_result_file(agent, model)

            if path is None:
                continue

            for case in json.load(open(path, encoding="utf-8")).get("cases", []):

                cid = case.get("case_id") or case.get("id")

                if cid in excluded_cases:
                    continue

                test_case = cases_by_id.get(cid, {})

                for run in case.get("runs", []):

                    if run.get("failure_type") == "execution_error":
                        continue

                    oracle_pass = bool(run.get("correct"))
                    scores = {k: run.get(f"deepeval_{k}_score") for k in JUDGE_KEYS}
                    verdicts = {k: s is not None and s >= JUDGE_PASS_THRESHOLD for k, s in scores.items()}

                    # judges that disagree with the oracle
                    wrong = [k for k in JUDGE_KEYS if scores[k] is not None and verdicts[k] != oracle_pass]

                    if not wrong:
                        continue

                    row = {
                        "outcome": "FN" if oracle_pass else "FP",
                        "agent": agent["label"],
                        "failure_targeted": agent["failure_focus"],
                        "executor": model.split("/")[-1],
                        "case_id": cid,
                        "case_type": case_group(cid),
                        "run": run.get("run"),
                        "judges_wrong": " + ".join(judge_name(k) for k in wrong),
                        "query": clean(case.get("query") or test_case.get("query") or ""),
                        "expected_tools": expected_text(test_case),
                        "expected_calls": expected_calls_text(test_case),
                        "tool_calls": tool_calls_text(run.get("tool_calls_info")),
                        "oracle_verdict": "pass" if oracle_pass else "fail",
                        "oracle_failure_type": run.get("failure_type") or "",
                        "final_response": run.get("final_content") or "",
                    }

                    for k in JUDGE_KEYS:
                        row[f"{judge_name(k)}_score"] = scores[k]
                        row[f"{judge_name(k)}_reason"] = clean(run.get(f"deepeval_{k}_reason") or "")

                    rows.append(row)

    return rows


def draw_sample(rows):
    rng = random.Random(SEED)
    drawn = []

    for agent in [a["label"] for a in AGENTS if not a.get("external")]:
        for outcome, draw, size in (("FP", DRAW_FP, FP_PER_AGENT), ("FN", DRAW_FN, FN_PER_AGENT)):

            pool = [r for r in rows if r["agent"] == agent and r["outcome"] == outcome]
            picked = rng.sample(pool, min(draw, len(pool)))

            for i, row in enumerate(picked):
                drawn.append({**row, "pilot": "yes" if i == 0 else "", "keep": i < size})

    for i, row in enumerate(drawn, start=1):
        row["sample_id"] = f"S{i:03d}"

    sample = [r for r in drawn if r.pop("keep")]

    # alternate the non-pilot runs between annotators, so each one gets
    # runs from every agent and both outcomes
    # (separate turns per outcome, FN starting on the other annotator, so the
    # FP/FN split is also balanced)
    turn = {"FP": 0, "FN": 1}
    for row in sample:
        if row["pilot"]:
            row["assignment"] = "both"
        else:
            row["assignment"] = ANNOTATORS[turn[row["outcome"]] % len(ANNOTATORS)]
            turn[row["outcome"]] += 1

    return sample


def main():
    rows = collect_runs()
    sample = draw_sample(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    first = ["sample_id", "pilot", "assignment"]
    columns = first + [c for c in sample[0] if c not in first] + CODE_COLUMNS

    for annotator in ANNOTATORS:
        path = os.path.join(OUTPUT_DIR, f"rq3_sample_annotator_{annotator}.csv")

        # utf-8-sig so that Excel / Google Sheets read the accents correctly
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {**row, **{c: "" for c in CODE_COLUMNS}}
                for row in sample if row["assignment"] in ("both", annotator)
            )

        print(f"Saved → {path}")

    fp = sum(r["outcome"] == "FP" for r in sample)
    print(f"\npopulation: {sum(r['outcome'] == 'FP' for r in rows)} FP, {sum(r['outcome'] == 'FN' for r in rows)} FN")
    print(f"sample: {len(sample)} runs ({fp} FP, {len(sample) - fp} FN), {sum(r['pilot'] == 'yes' for r in sample)} pilot")
    for annotator in ANNOTATORS:
        print(f"  annotator {annotator}: {sum(r['assignment'] == annotator for r in sample)} runs + pilot")

    for key, label in (("agent", "per agent"), ("executor", "per executor"), ("assignment", "per assignment")):
        print(f"\n{label}:")
        for value in sorted({r[key] for r in sample}):
            s = [r for r in sample if r[key] == value]
            print(f"  {value}: {sum(r['outcome'] == 'FP' for r in s)} FP, {sum(r['outcome'] == 'FN' for r in s)} FN")


if __name__ == "__main__":
    main()
