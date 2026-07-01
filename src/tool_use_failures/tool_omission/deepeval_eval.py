from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

task_completion = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4o"
)

def get_goldens(inputs_list):
    return [EvaluationDataset(
        goldens=[
            Golden(
                input=input
            ),
            Golden(
                input=input
            ),
        ]
    ) for input in inputs_list]


from deepeval.test_case import LLMTestCase

def eval_task_completion(input, actual_output):
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
    )
    task_completion.measure(test_case)
    print("=" * 80)
    print("SCORE:", task_completion.score)
    print("REASON:", task_completion.reason)

    return task_completion.reason, task_completion.score