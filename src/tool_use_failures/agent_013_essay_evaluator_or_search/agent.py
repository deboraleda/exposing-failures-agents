from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Dict, Any
import operator
import os

import requests


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


FETCH_TIMEOUT = 15
SEARCH_TIMEOUT = 30
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5


@tool
def fetch_document(url: str) -> str:
    """Fetch the textual content of a public document through HTTP GET, without an API key, for essay reading, evaluation or feedback."""

    try:
        response = requests.get(url, timeout=FETCH_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        return f"ERROR: could not fetch '{url}': {e}"

    response.encoding = "utf-8-sig"

    return response.text.strip()

@tool
def search_about_topic(topic: str) -> str:
    """Research information, definitions or examples concerning essay-writing techniques or subject matter to support an essay through Tavily."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "ERROR: TAVILY_API_KEY is not set."

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": topic,
                "max_results": TAVILY_MAX_RESULTS,
                "include_answer": True,
            },
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        return f"ERROR: could not search about '{topic}': {e}"

    data = response.json()

    results = data.get("results", [])

    if not data.get("answer") and not results:
        return f"ERROR: no results found for '{topic}'."

    sections = []

    if data.get("answer"):
        sections.append(f"Summary: {data['answer']}")

    for idx, result in enumerate(results, start=1):
        sections.append(
            f"[{idx}] {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"{result.get('content', '')}"
        )

    return "\n\n".join(sections)

tools = [fetch_document, search_about_topic]

tools_by_name = {t.name: t for t in tools}


def agent(llm):

    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(content="""
        You are an essay assistant that uses explicit wording, explains technical terms when necessary, and omits information unrelated to the user's request.
        Your primary responsibility is to provide feedback on essays.
        You can read, evaluate, and provide feedback on public documents supplied via URL. You can also research information, definitions, and examples relevant to essay writing.
        All responses must be grounded in retrieved content.

        Each execution may invoke at most one tool, exactly once. Tool chaining, parallel tool execution, retries, and fallback to alternative tools are strictly prohibited.

        # Behaviors

        1. **Empty input:** If the input is empty or contains only whitespace, return:
        ```json
        {
            "status": "failure",
            "user_request": "<user request here>",
            "response": "empty or whitespace-only"
        }
        ```

        2. **Unrelated requests:** If the request is unrelated to essay evaluation or research, return:
        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "unrelated request"
        }
        ```

        3. **Document retrieval:** If the user provides a URL and requests that a document be retrieved, return:
        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "<a natural-language response incorporating the retrieved content and addressing the request>"
        }
        ```

        4. **Topic research:** If the user requests information about a topic, return:
        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "<a natural-language response incorporating the retrieved content and addressing the request>"
        }
        ```

        5. **Execution failures:** If a technical failure occurs, the retrieved content is unusable, or the execution budget is exhausted, return:
        ```json
        {
            "status": "failure",
            "user_request": "<user request here>",
            "response": "<error message describing the failure>"
        }
   ```
    """)

    def agent_node(state: AgentState):

        messages = [system_prompt] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def should_continue(state: AgentState):

        last = state["messages"][-1]

        return "tools" if last.tool_calls else END

    def tool_node(state: AgentState):

        last = state["messages"][-1]

        tool_call = last.tool_calls[0]

        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])

        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            ]
        }

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)

    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    def run(query: str) -> Dict[str, Any]:

        inputs = {
            "messages": [HumanMessage(content=query)]
        }

        result = graph.invoke(inputs)

        final_message = result["messages"][-1]

        tool_called = False

        tool_calls_info = []

        for msg in result["messages"]:

            if hasattr(msg, "tool_calls") and msg.tool_calls:

                tool_called = True

                tool_calls_info.extend([
                    {
                        "name": call["name"],
                        "args": call["args"],
                    }
                    for call in msg.tool_calls
                ])

        return {
            "success": True,
            "tool_called": tool_called,
            "tool_calls_info": tool_calls_info,
            "final_content": final_message.content,
        }

    return run


