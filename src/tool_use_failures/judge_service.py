# judge_service.py
import os
import re

from deepeval.metrics import TaskCompletionMetric
from deepeval.models import GPTModel, LocalModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from tool_use_failures.config import JUDGES, MAX_TOKENS, PROVIDER_CONFIG

load_dotenv(override=True)


def judge_key(model_name):
    """Column-safe judge name, e.g. "Qwen/Qwen3.8-27B" -> "qwen3_8_27b"."""

    return re.sub(r"[^a-z0-9]+", "_", model_name.split("/")[-1].lower()).strip("_")


def get_judge_model(provider, model_name):
    config = PROVIDER_CONFIG[provider]

    if config["base_url"] is None:
        return GPTModel(
            model=model_name,
            api_key=os.getenv(config["api_key_env"]),
        )

    # OpenAI-compatible endpoint (e.g. DeepInfra)
    return LocalModel(
        model=model_name,
        base_url=config["base_url"],
        api_key=os.getenv(config["api_key_env"]),
        generation_kwargs={"max_tokens": MAX_TOKENS},
    )


TASK_COMPLETION_METRICS = {
    judge_key(judge["model"]): TaskCompletionMetric(
        threshold=0.7,
        model=get_judge_model(judge["provider"], judge["model"]),
    )
    for judge in JUDGES
}

# Columns produced for every run: deepeval_<judge>_score / _reason / _error
DEEPEVAL_COLUMNS = [
    f"deepeval_{key}_{field}"
    for key in TASK_COMPLETION_METRICS
    for field in ("score", "reason", "error")
]


def eval_task_completion(actual_input, actual_output):
    """Evaluate the response with every judge.

    A failing judge does not stop the others: its error is stored in
    deepeval_<judge>_error and its score/reason stay None.
    """

    test_case = LLMTestCase(
        input=actual_input,
        actual_output=actual_output,
    )

    results = {}

    for key, metric in TASK_COMPLETION_METRICS.items():

        score = None
        reason = None
        error = None

        try:
            metric.measure(test_case)
            score = metric.score
            reason = metric.reason

        except Exception as e:
            error = str(e)

        print("=" * 80)
        print(f"JUDGE: {key}")

        if error:
            print(f"DeepEval ERROR: {error}")
        else:
            print("SCORE:", score)
            print("REASON:", reason)

        results[f"deepeval_{key}_score"] = score
        results[f"deepeval_{key}_reason"] = reason
        results[f"deepeval_{key}_error"] = error

    return results


def summarize_scores(results):
    """Short text for the run log, e.g. "qwen3_8_27b=0.9, minimax_m3=0.8"."""

    return ", ".join(
        f"{key}={results.get(f'deepeval_{key}_score')}"
        for key in TASK_COMPLETION_METRICS
    )
