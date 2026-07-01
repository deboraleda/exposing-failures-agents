# Failure-Exposing Agents

This repository contains three task-oriented agents designed to expose common tool-use failures in LLM-based agents: **tool omission**, **parameter error**, and **wrong tool selection**. The agents do **not** contain injected faults. Instead, failures emerge naturally from the model's reasoning when interacting with carefully designed test cases.

The implemented agents are:

1. **[Customer Support Agent (Tool Omission)](src/tool_use_failures/tool_omission)**  
   Evaluates whether the agent correctly invokes the required tool when the user's request satisfies the tool's preconditions.

2. **[Flight Booking Agent (Parameter Error)](src/tool_use_failures/parameter_error)**  
   Evaluates whether the agent invokes the correct tool using the expected parameter values extracted from the user's request.

3. **[Order Management Agent (Wrong Tool Selection)](src/tool_use_failures/wrong_tool_selection)**  
   Evaluates whether the agent selects the appropriate tool according to the user's request and the operational policy.

## Repository Structure

Each agent directory follows the same structure:

- **agent** – Agent implementation and tool definitions.
- **deepeval_eval** – DeepEval TaskCompletionMetric configuration.
- **experiment** – Experimental pipeline, including deterministic assertions and semantic evaluation.
- **test_cases** – Test case definitions.

Each agent includes:

- 4 baseline test cases;
- 4 conflict test cases.

The **Wrong Tool Selection** agent additionally includes **2 oracle-equivalent** test cases, where multiple tool invocations are considered valid.

## Supported Models

The supported models are defined in [`config.py`](src/tool_use_failures/config.py):

```python
MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini"
    ],
    "deepinfra": [
        "Qwen/Qwen3-14B",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ]
}
```

## Running the Experiments

To execute an experiment, run the corresponding `experiment.py` module for the desired agent and specify the model as a command-line argument.

Example:

```bash
python -m tool_use_failures.parameter_error.experiment \
    --model Qwen/Qwen2.5-7B-Instruct
```

Each execution runs all test cases 10 times for the selected agent, performs deterministic oracle assertions, evaluates the agent's final response using the DeepEval TaskCompletionMetric, and stores the results in a CSV file.