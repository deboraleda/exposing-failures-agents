from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

task_completion = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4o"
)


from deepeval.test_case import LLMTestCase

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