# Failure-Exposing Agents

This repository contains nine task-oriented agents designed to expose common tool-use failures in LLM-based agents: **tool omission**, **partial tool omission**, **parameter error**, **wrong tool selection**, and calling more than one tool when **only one tool** is allowed. The agents do **not** contain injected faults. Instead, failures emerge naturally from the model's reasoning when interacting with carefully designed test cases.

The implemented agents are:

| Agent | Goal | Tools | Tool-use rules | Failure type |
|---|---|---|---|---|
| [Travel destination assistant](src/tool_use_failures/agent_012_explore_travel_destination) | Answer questions about travel destinations, weather conditions, news, safety, advisories, events and tips | `get_weather`, `search_destination_news` | weather → `get_weather`; news, safety, advisories or events → `search_destination_news`; general tips → no tool. At most one tool per execution. | Only one tool |
| [Essay evaluator or search](src/tool_use_failures/agent_013_essay_evaluator_or_search) | Evaluate essays or search for information, definitions and relevant examples for essay writing | `fetch_document`, `search_about_topic` | essay URL + evaluation request → `fetch_document`; topic information → `search_about_topic`. Exactly one tool per execution. | Only one tool |
| [Academic triage](src/tool_use_failures/agent_014_academic_triage_agent) | Triage a user's academic request by intent and subject and route | `search_arxiv`, `search_pubmed`, `get_wikipedia_summary` | exact sciences/technology papers → arXiv; life sciences/health papers → PubMed; concept explanation → Wikipedia. Exactly one tool per execution. | Wrong tool selection |
| [Recipe generator](src/tool_use_failures/agent_018_recipe_generator) | Generate a personalized recipe based on the user's ingredients, respecting dietary restrictions | `retrieve_pantry`, `search_recipes` | compatible ingredients in the request → `search_recipes`; otherwise → `retrieve_pantry`. At most one tool per execution. `search_recipes` must receive exactly the ingredients mentioned in the request. | Parameter error |
| [Travel viability](src/tool_use_failures/agent_022_explore_travel_viability) | Classify a trip destination as favorable, partially favorable or unfavorable | `get_weather`, `search_destination_news` | both tools on every valid request | Partial tool omission |
| [Trip budget and weather](src/tool_use_failures/agent_030_trip_budget_weather) | Calculate the total trip budget, select a destination compatible with it and report its weather. | `calculate_trip_budget`, `select_location`, `get_weather` | each tool's output is the next tool's input | Parameter error |
| [Flight booking](src/tool_use_failures/flight_booking) | Book flights using the data provided in the request. | `book_flight` | always call the tool, extracting the eight mandatory fields from the request | Parameter error |
| [Replacement orders](src/tool_use_failures/replacement_orders) | Help customers with their order inquiries and trigger replacement orders | `trigger_replacement_order` | Call only if all conditions are met: the customer reports the order has not arrived, it was placed more than 7 days ago, and there were no shipping updates | Tool omission |
| [Order support](src/tool_use_failures/order_support) | Manage store orders: cancellations, refunds and customer lookups | `cancel_order`, `refund_order`, `get_client`, `search_client` | `cancel_order` only for pending orders; `refund_order` only for shipped or delivered orders; `get_client` when the client ID is known, `search_client` otherwise | Wrong tool selection |

## Agent Inspirations

Most of the agents were inspired by open-source agents. These base agents served only as references: the agents in this repository are **not** reproductions or extensions of them.

- **Multiple tools with selection criteria**: the agent has access to more than one tool and must choose which one to invoke according to the user's request (Travel destination assistant, Essay evaluator or search, Academic triage and Recipe generator).
- **Dependency between tools**: the agent must invoke more than one tool in a defined sequence (Travel viability and Trip budget and weather).

| Base agent | Inspired agent(s) |
|---|---|
| [Travel Planner](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_travel_planner_langgraph.ipynb) | Travel destination assistant, Travel viability |
| [essay_evaluator_agent](https://github.com/giovannamoeller/edu-ai-adk-version/blob/main/agents/essay_evaluator_agent/agent.py) | Essay evaluator or search |
| [Recipe AI - Smart Recipe Generation](https://github.com/chitralputhran/Recipe-AI-Easy-Recipes) | Recipe generator |
| [Weather Agent](https://github.com/rishav-learnerml/weather-agent) | Trip budget and weather |

Academic triage, Flight booking, Replacement orders and Order support were designed from scratch.

## Repository Structure

Each agent directory follows the same structure:

- **agent.py** – Agent implementation and tool definitions.
- **deepeval.py** – DeepEval TaskCompletionMetric configuration (`deepeval_eval.py` in Replacement orders).
- **experiment.py** – Experimental pipeline, including deterministic assertions and semantic evaluation.
- **test_cases.py** – Test case definitions.
- **results/** – Result files of each execution.

Each agent is evaluated on:

- 4 baseline test cases;
- 4 conflict test cases.

Two agents define extra test cases that are left out of the analysis.

## Manual Validation

To investigate which agent behaviors lead to semantic misclassification in tool-use evaluation, we manually analyzed a sample of 72 executions in which the LLM-as-a-judge (DeepEval TaskCompletionMetric) disagrees with the deterministic oracle. A **false positive** occurs when the judge accepts an execution that the oracle rejects. A **false negative** occurs when the judge rejects an execution that the oracle accepts.

Each execution was annotated with two codes: an **agent behavior code** and a **judge rationale code**.

### Agent Behavior Codes

| Code | Meaning |
|---|---|
| `WRONG_TOOL_CALL` | Does not call the expected tool and calls one or more inappropriate tools instead. |
| `CORRECT_TOOL_CALL` | Calls the correct tool, or no tool when none applies, and the response is correct. |
| `EXTRA_TOOL_CALL` | Calls the expected tool(s) and, in addition, another tool or the same tool more than once. |
| `REPORTS_MISSING_INFORMATION` | The tool returns no information and the agent reports that it does not have enough information to answer. |
| `WRONG_VALUE_PARAMETER` | Passes a wrong parameter value. |
| `PARAMETER_OVERLOAD` | Passes parameters beyond those required. |
| `CLARIFICATION_OR_DEFERRAL` | Asks clarification questions or defers the answer. |
| `MISSING_TOOL_CALL` | Does not call one or more expected tools, without calling another tool in their place. |

### Judge Rationale Codes

| Code | Meaning |
|---|---|
| `ANSWER_FROM_AGENT_REASONING` | Accepts a plausible response even though it is based on the agent's reasoning rather than on tool outputs. |
| `CORRECT_RESPONSE_HIDES_FAILURE` | Accepts a correct response that hides failures in the tool calls. |
| `PENALIZES_ANSWER_WITHIN_AGENT_SCOPE` | Penalizes a response that is within the limits of the agent's scope. |
| `ACCEPTS_FALSE_PREMISE` | Accepts a response as correct even though it is based on a false premise. |
| `CONTRADICTS_TOOL_WITH_WORLD_KNOWLEDGE` | Penalizes the response provided by the tool based on its own world knowledge. |
| `ACCEPTS_CLARIFICATION_AS_ANSWER` | Accepts a clarification or deferral response as correct. |
| `PENALIZES_INCOMPLETE_ANSWER` | Correctly penalizes an incomplete response. |

## Supported Models

The supported models are defined in [`config.py`](src/tool_use_failures/config.py):

```python
MODELS = {
    "openai": ["gpt-6-luna"],
    "deepinfra": [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "zai-org/GLM-5.3-Flash",
    ],
}
```

Every run is also scored by two LLM judges, `Qwen/Qwen3.8-27B` and `MiniMaxAI/MiniMax-M3` (`JUDGES` in the same file).

## Running the Experiments

To execute an experiment, run the corresponding `experiment.py` module for the desired agent and specify the model as a command-line argument.

Example:

```bash
python -m tool_use_failures.flight_booking.experiment \
    --model deepseek-ai/DeepSeek-V4.1-Flash
```

Each execution runs all test cases 10 times for the selected agent, performs deterministic oracle assertions, evaluates the agent's final response using the DeepEval TaskCompletionMetric, and stores the results as JSON and CSV files in the agent's `results/` folder.

