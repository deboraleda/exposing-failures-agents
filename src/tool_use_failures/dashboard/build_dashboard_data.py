# build_dashboard_data.py
#
# Builds the data files of the "Corrupt Success in Tool-Calling Agents" dashboard
# from the experiments' result JSON files.
#
# Usage (from src/):
#   python -m tool_use_failures.dashboard.build_dashboard_data
#
# Only result files of the executor models in config.MODELS are used, and only
# the verdicts of the judges in config.JUDGES. For each (agent, model) the most
# recent result file is kept.

import ast
import glob
import importlib
import json
import os
import re
from datetime import datetime, timezone

from tool_use_failures.config import JUDGES, MODELS, N_RUNS

# TaskCompletionMetric threshold used by judge_service.py
JUDGE_PASS_THRESHOLD = 0.7

# Cases whose id starts with one of these prefixes are "conflicting"
# (ambiguous / adversarial); every other case is "baseline".
CONFLICTING_PREFIXES = ("am_", "inj_")

MAX_RESPONSE_CHARS = 1500

# Test cases left out of the study (every agent keeps 4 baseline + 4 conflicting
# cases). Their runs stay in the result files but are not counted anywhere.
EXCLUDED_CASES = {
    "agent_014": {"nt_1"},                    # empty input: says nothing about tool selection
    "wrong_tool_selection": {"eq_1", "eq_2"},  # extra get_client / search_client cases
}

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE_DIR = os.path.join(SRC_DIR, "tool_use_failures")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

AGENTS = [
    {"agent": "agent_012", "package": "agent_012_explore_travel_destination", "module": "agent.py",
     "label": "Travel destination assistant", "failure_focus": "Only one tool", "file_prefix": "results_agent012_"},
    {"agent": "agent_013", "package": "agent_013_essay_evaluator_or_search", "module": "agent.py",
     "label": "Essay evaluator or search", "failure_focus": "Only one tool", "file_prefix": "results_013_"},
    {"agent": "agent_014", "package": "agent_014_academic_triage_agent", "module": "agent.py",
     "label": "Academic triage", "failure_focus": "Wrong tool selection", "file_prefix": "results_014_"},
    {"agent": "agent_018", "package": "agent_018_recipe_generator", "module": "agent.py",
     "label": "Recipe generator", "failure_focus": "Parameter error", "file_prefix": "results_agent018_"},
    {"agent": "agent_022", "package": "agent_022_explore_travel_viability", "module": "agent.py",
     "label": "Travel viability", "failure_focus": "Partial tool omission", "file_prefix": "results_022_"},
    {"agent": "agent_030", "package": "agent_030_trip_budget_weather", "module": "agent.py",
     "label": "Trip budget and weather", "failure_focus": "Parameter error", "file_prefix": "results_agent030_"},
    {"agent": "parameter_error", "package": "flight_booking", "module": "agent.py",
     "label": "Flight booking", "failure_focus": "Parameter error", "file_prefix": "results_parameter_error_"},
    {"agent": "tool_omission", "package": "replacement_orders", "module": "agent.py",
     "label": "Replacement orders", "failure_focus": "Tool omission", "file_prefix": "results_"},
    {"agent": "wrong_tool_selection", "package": "order_support", "module": "agent.py",
     "label": "Order support", "failure_focus": "Wrong tool selection", "file_prefix": "results_t2_"},
]

# Eureca Chat: a beta agent developed outside this study, kept out of every
# count above and shown in its own tab (exploratory study in the Discussion).
# Its runs come from eureka_chat/ at the repository root: for each executor,
# agent_experiment_<model>.jsonl (trace and final response) and
# tool_evaluation_agent_experiment_<model>.jsonl (deterministic verdict).
EUREKA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "eureka_chat")
EUREKA_MODELS = {"deepseek": "DeepSeek-V4.1-Flash", "GLM": "GLM-5.3-Flash", "gpt-luna": "gpt-6-luna"}
EUREKA_INTENTIONS = {"clara": "clear", "ambigua": "ambiguous"}
EUREKA_TOOLS = [
    # available to every user
    "eureca_database", "buscar_fluxograma", "buscar_plano_curso_disciplina",
    "buscar_grade_curricular_curso", "buscar_calendarios", "execute_bash",
    # student only
    "estudante_info", "historico_estudante",
    # professor only
    "professor_info",
]
EUREKA_TOOLS_NOTE = "6 shared; 8 for students, 7 for professors, 7 for staff"


def judge_key(model_name):
    """Same naming as judge_service.judge_key, e.g. "Qwen/Qwen3.8-27B" -> "qwen3_8_27b"."""

    return re.sub(r"[^a-z0-9]+", "_", model_name.split("/")[-1].lower()).strip("_")


def safe_model(model_name):
    """Model name as the experiments write it in file names."""

    return model_name.replace("/", "_").replace(":", "-")


JUDGE_KEYS = [judge_key(j["model"]) for j in JUDGES]
JUDGE_LABELS = {judge_key(j["model"]): j["model"] for j in JUDGES}
EXECUTOR_MODELS = [m for models in MODELS.values() for m in models]


def case_group(case_id):
    return "conflicting" if str(case_id).startswith(CONFLICTING_PREFIXES) else "baseline"


def tool_names(path):
    """Names of the functions decorated with @tool in an agent module."""

    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if getattr(target, "id", None) == "tool" or getattr(target, "attr", None) == "tool":
                    names.append(node.name)

    return names


def load_test_cases(package):
    cases = importlib.import_module(f"tool_use_failures.{package}.test_cases").TEST_CASES
    if cases and isinstance(cases[0], dict) and "cases" in cases[0]:
        cases = cases[0]["cases"]
    return cases


def case_id_of(case):
    return case.get("id") or case.get("case_id")


def find_result_file(agent, model):
    """Most recent result JSON of an agent for one executor model."""

    stem = f"{agent['file_prefix']}{safe_model(model)}_"
    folders = [
        SRC_DIR,
        os.path.join(PACKAGE_DIR, agent["package"]),
        os.path.join(PACKAGE_DIR, agent["package"], "results"),
    ]
    candidates = []

    for folder in folders:
        for path in glob.glob(os.path.join(folder, glob.escape(stem) + "*.json")):
            # only a timestamp may follow the model name: YYYYMMDD_HHMM
            if re.fullmatch(r"\d{8}_\d{4}\.json", os.path.basename(path)[len(stem):]):
                candidates.append(path)

    # by the timestamp in the name: copies and checkouts reset the modification times
    return max(candidates, key=lambda p: (os.path.basename(p)[len(stem):], os.path.getmtime(p))) if candidates else None


def expected_text(case):
    for key in ("acceptable_tools", "expected_tool"):
        if key in case:
            value = case[key]
            if isinstance(value, list):
                value = ", ".join(value) if value else "no tool"
            return f"{value or 'no tool'}"
    if "should_call_tool" in case:
        return "a tool call" if case["should_call_tool"] else "no tool"
    return ""


def expected_calls_text(case):
    """Expected calls with their arguments, e.g. "book_flight({...})"; "" when the case has no expected_args.

    expected_args is either the arguments of a single expected tool or a {tool: args} map.
    With acceptable_tools, any of them may take the arguments: "search_arxiv | search_pubmed({...})"."""

    args = case.get("expected_args")
    if not args:
        return ""
    tools = case.get("expected_tool") or case.get("acceptable_tools") or []
    tools = tools if isinstance(tools, list) else [tools]
    if not set(args) <= set(tools):
        args = {" | ".join(tools) or "any tool": args}
    return "; ".join(f"{name}({json.dumps(a, ensure_ascii=False)})" for name, a in args.items())


def clean(text):
    return " ".join(str(text).split()) if text else text


def tool_calls_text(tool_calls_info):
    if not tool_calls_info:
        return "no tool called"
    return "; ".join(
        f"{call.get('name')}({json.dumps(call.get('args', {}), ensure_ascii=False)})"
        for call in tool_calls_info
    )


def eureka_outcome(evaluation):
    """Oracle outcome of an Eureca Chat run, one label per run."""

    if evaluation["passed"]:
        return "pass"
    if evaluation["loop"]:
        return "retry_loop"
    if evaluation["missing_tools"]:
        return "omission"
    if evaluation["extra_tools"]:
        return "extra_call"
    return "other"


def build_eureka():
    """One row per Eureca Chat run, plus the list of input files."""

    rows = []
    inputs = []

    for suffix, model in EUREKA_MODELS.items():

        agent_path = os.path.join(EUREKA_DIR, f"agent_experiment_{suffix}.jsonl")
        eval_path = os.path.join(EUREKA_DIR, f"tool_evaluation_agent_experiment_{suffix}.jsonl")

        if not (os.path.exists(agent_path) and os.path.exists(eval_path)):
            continue

        inputs += [os.path.relpath(p, SRC_DIR) for p in (agent_path, eval_path)]
        traces = {
            (t["run_number"], t["question_id"], t["repetition"]): t
            for t in map(json.loads, open(agent_path, encoding="utf-8"))
        }

        for e in map(json.loads, open(eval_path, encoding="utf-8")):

            trace = traces.get((e["run_number"], e["question_id"], e["repetition"]), {})
            calls = [i for i in trace.get("interactions", []) if i.get("kind") == "tool_call"]

            rows.append({
                "key": f"{model}|{e['question_id']}|{e['repetition']}",
                "model": model,
                "question_id": e["question_id"],
                "intention": EUREKA_INTENTIONS.get(e["intention"], e["intention"]),
                "question": e["question"],
                "repetition": e["repetition"],
                "passed": bool(e["passed"]),
                "outcome": eureka_outcome(e),
                "expected_tools": ", ".join(e["expected_tools"]),
                "optional_tools": ", ".join(e["optional_tools"]),
                "called_tools": ", ".join(
                    f"{name} ×{n}" if n > 1 else name for name, n in e["called_tool_counts"].items()
                ),
                "n_tool_calls": len(calls),
                "extra_tools": ", ".join(e["extra_tool_names"]),
                "missing_tools": ", ".join(e["missing_tools"]),
                # the database tool called with a flag in place of the user's question
                "malformed_call": any(
                    c.get("name") == "eureca_database" and str(c.get("arguments")).strip() in ("", "False", "True", "None")
                    for c in calls
                ),
                "comment": e["comment"],
                "duration_s": (trace.get("backend") or {}).get("duration_seconds"),
                "response": ((trace.get("backend") or {}).get("final_response") or "")[:MAX_RESPONSE_CHARS],
            })

    return rows, inputs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    own_code = open(os.path.abspath(__file__), encoding="utf-8").read()

    agent_rows = []
    run_rows = []
    corrupt_rows = []
    excluded_rows = []
    inputs = []

    for agent in AGENTS:

        excluded_cases = EXCLUDED_CASES.get(agent["agent"], set())
        cases = [c for c in load_test_cases(agent["package"]) if case_id_of(c) not in excluded_cases]
        cases_by_id = {case_id_of(c): c for c in cases}
        n_conflicting = sum(case_group(case_id_of(c)) == "conflicting" for c in cases)
        tools = tool_names(os.path.join(PACKAGE_DIR, agent["package"], agent["module"]))

        agent_rows.append({
            "agent": agent["agent"],
            "label": agent["label"],
            "n_tools": len(tools),
            "tools": ", ".join(tools),
            "failure_focus": agent["failure_focus"],
            "status": "Run in this study",
            "n_cases": len(cases),
            "n_baseline": len(cases) - n_conflicting,
            "n_conflicting": n_conflicting,
            # runs if every case ran N_RUNS times with every executor model
            "expected_runs": len(cases) * N_RUNS * len(EXECUTOR_MODELS),
        })

        candidates = []

        for model in EXECUTOR_MODELS:

            path = find_result_file(agent, model)

            if path is None:
                continue

            inputs.append(os.path.relpath(path, SRC_DIR))
            results = json.load(open(path, encoding="utf-8"))

            for case in results.get("cases", []):

                cid = case.get("case_id") or case.get("id")

                if cid in excluded_cases:
                    continue
                group = case_group(cid)

                for run in case.get("runs", []):

                    run_key = f"{agent['agent']}|{model}|{cid}|{run.get('run')}"

                    # a crashed run (e.g. provider overloaded) says nothing about
                    # tool calling: it is left out of every count and listed apart
                    if run.get("failure_type") == "execution_error":
                        excluded_rows.append({
                            "key": run_key,
                            "agent": agent["agent"],
                            "model": model,
                            "case_id": cid,
                            "error": clean(run.get("error") or "")[:200],
                        })
                        continue
                    det_pass = bool(run.get("correct"))
                    failure_type = run.get("failure_type") or ("none" if det_pass else "unknown")
                    judges_passed = []
                    scores = {}

                    for jk in JUDGE_KEYS:
                        score = run.get(f"deepeval_{jk}_score")
                        judge_pass = None if score is None else score >= JUDGE_PASS_THRESHOLD

                        run_rows.append({
                            "key": f"{run_key}|{jk}",
                            "run_key": run_key,
                            "agent": agent["agent"],
                            "model": model,
                            "case_id": cid,
                            "group": group,
                            "det_pass": det_pass,
                            "failure_type": failure_type,
                            "judge": jk,
                            "judge_label": JUDGE_LABELS[jk],
                            "judge_score": score,
                            "judge_pass": judge_pass,
                        })

                        if judge_pass:
                            judges_passed.append(jk)
                            scores[jk] = score

                    if not det_pass and judges_passed and failure_type != "execution_error":
                        best = max(judges_passed, key=lambda k: scores[k])
                        candidates.append((len(judges_passed), sum(scores.values()) / len(scores), {
                            "agent": agent["agent"],
                            "model": model,
                            "case_id": cid,
                            "group": group,
                            "query": clean(case.get("query")),
                            "expected": expected_text(case),
                            # for parameter errors the expected arguments matter, not just the tool
                            "expected_calls": expected_calls_text(cases_by_id.get(cid, {}))
                            if failure_type == "tool_parameter_error" else "",
                            "tool_calls": tool_calls_text(run.get("tool_calls_info") or []),
                            "failure_type": failure_type,
                            "judges_passed": f"{len(judges_passed)} of {len(JUDGE_KEYS)}",
                            "judge_scores": ", ".join(
                                f"{JUDGE_LABELS[k]}: {run.get(f'deepeval_{k}_score')}" for k in JUDGE_KEYS
                            ),
                            "judge_reason_by": JUDGE_LABELS[best],
                            "judge_reason": run.get(f"deepeval_{best}_reason"),
                            "response": (run.get("final_content") or "")[:MAX_RESPONSE_CHARS],
                        }))

        if candidates:
            # passed by the most judges, then the highest mean judge score
            corrupt_rows.append(max(candidates, key=lambda c: (c[0], c[1]))[2])

    eureka_rows, eureka_inputs = build_eureka()
    inputs += eureka_inputs

    meta = [
        {"label": "build_dashboard_data.py", "value": own_code, "type": "code", "lang": "python"},
        {"label": "Input", "value": "\n".join(inputs) or "no result files yet"},
        {"label": "Built", "value": built_at},
    ]

    outputs = {
        "agents.json": agent_rows,
        "runs.json": run_rows,
        "corrupt_examples.json": corrupt_rows,
        "excluded_runs.json": excluded_rows,
        "eureka.json": eureka_rows,
        "eureka_info.json": [{"tools": ", ".join(EUREKA_TOOLS), "n_tools": len(EUREKA_TOOLS), "tools_note": EUREKA_TOOLS_NOTE}],
    }

    for name, rows in outputs.items():
        with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "meta": meta}, f, ensure_ascii=False, indent=1)

    # standalone page: index.html inside the local shell, with the data inlined
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    shell = open(os.path.join(dashboard_dir, "local_shell.html"), encoding="utf-8").read()
    body = open(os.path.join(dashboard_dir, "index.html"), encoding="utf-8").read()
    data = json.dumps(
        {name[:-len(".json")]: rows for name, rows in outputs.items()},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    local_page = os.path.join(dashboard_dir, "dashboard.html")

    with open(local_page, "w", encoding="utf-8") as f:
        f.write(shell.replace("/*__DATA__*/", data).replace("<!--__BODY__-->", body))

    print(
        f"agents: {len(agent_rows)} | verdict rows: {len(run_rows)} | "
        f"corrupt examples: {len(corrupt_rows)} | excluded runs: {len(excluded_rows)} | "
        f"eureca chat runs: {len(eureka_rows)}"
    )
    print("inputs:", *inputs, sep="\n  ")
    print(f"local dashboard: {local_page}")


if __name__ == "__main__":
    main()
