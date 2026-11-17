import os

from deepeval.metrics import TaskCompletionMetric
from deepeval.models import GPTModel, LocalModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from tool_use_failures.config import (
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    MAX_TOKENS,
    PROVIDER_CONFIG,
)

load_dotenv(override=True)


def get_judge_model():
    config = PROVIDER_CONFIG[JUDGE_PROVIDER]

    if config["base_url"] is None:
        return GPTModel(
            model=JUDGE_MODEL,
            api_key=os.getenv(config["api_key_env"]),
        )

    # OpenAI-compatible endpoint (e.g. DeepInfra)
    return LocalModel(
        model=JUDGE_MODEL,
        base_url=config["base_url"],
        api_key=os.getenv(config["api_key_env"]),
        generation_kwargs={"max_tokens": MAX_TOKENS},
    )


task_completion = TaskCompletionMetric(
    threshold=0.7,
    model=get_judge_model()
)


def eval_task_completion(actual_input, actual_output):
    test_case = LLMTestCase(
        input=actual_input,
        actual_output=actual_output,
    )
    task_completion.measure(test_case)
    print("=" * 80)
    print("SCORE:", task_completion.score)
    print("REASON:", task_completion.reason)

    return task_completion.reason, task_completion.score
