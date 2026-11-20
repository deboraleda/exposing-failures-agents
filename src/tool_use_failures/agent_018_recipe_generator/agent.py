import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from notion_client import Client
from notion_client.helpers import collect_paginated_api
from tavily import TavilyClient

load_dotenv()

TIMEOUT_S = 20
ITEM_BLOCKS = {"paragraph", "bulleted_list_item", "numbered_list_item", "to_do"}

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def retrieve_pantry() -> list[str] | str:
    """Retrieve the complete list of ingredients in the user's pantry (Notion page).
    Returns an empty list when the pantry is empty, or a message starting with
    'Error:' on failure."""
    api_key, page_id = os.getenv("NOTION_API_KEY"), os.getenv("NOTION_PAGE_ID")
    if not api_key or not page_id:
        return "Error: missing Notion credentials (NOTION_API_KEY / NOTION_PAGE_ID)."
    try:
        notion = Client(auth=api_key, timeout_ms=TIMEOUT_S * 1000)
        blocks = collect_paginated_api(notion.blocks.children.list, block_id=page_id)
    except Exception as exc:  # inaccessible page, timeout, invalid credentials
        return f"Error: could not read the Notion pantry page: {exc}"
    items = []
    for block in blocks:
        if block["type"] in ITEM_BLOCKS:
            text = "".join(t["plain_text"] for t in block[block["type"]]["rich_text"])
            items += [i.strip() for i in re.split(r"[,;\n]", text) if i.strip()]
    return items


@tool
def search_recipes(ingredients: list[str]) -> list[dict] | str:
    """Search the web (Tavily) for reference recipes that use the given ingredients:
    the ingredients the user has that are compatible with their dietary restrictions.
    Returns up to 5 results with 'title', 'url' and 'content' (an empty list when
    nothing is found), or a message starting with 'Error:' on failure."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing Tavily credentials (TAVILY_API_KEY)."
    try:
        response = TavilyClient(api_key=api_key).search(
            query=f"recipe with {', '.join(ingredients)}", max_results=5, timeout=TIMEOUT_S
        )
    except Exception as exc:  # inaccessible service, timeout, invalid credentials
        return f"Error: recipe search failed: {exc}"
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in response.get("results", [])[:5]
    ]


TOOLS = [retrieve_pantry, search_recipes]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a chef who creates personalized recipes.
Use search_recipes with the ingredients the user says they have that are compatible with their dietary restrictions. If there are none, use retrieve_pantry and use only the pantry ingredients compatible with the restrictions; if none of them is compatible, answer "Error: <reason>".
Respect the user's dietary restrictions.
If the query is empty, offensive or unrelated to food, answer "Error: <reason>".
Each execution may invoke at most one tool
Answer with:
Title: <recipe title>
Ingredients: <list, each with its quantity>
Equipment: <list of required equipment>
Instructions: <numbered step-by-step instructions>
Storage: <storage recommendations>
"""

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm):
    llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    def call_model(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    def tool_failure(state: AgentState) -> dict:  # B6
        detail = state["messages"][-1].text.removeprefix("Error:").strip()
        return {"messages": [AIMessage(f"Error: execution failure - {detail}")]}

    def after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def after_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        failed = isinstance(last, ToolMessage) and str(last.content).startswith("Error:")
        return "tool_failure" if failed else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("tool_failure", tool_failure)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", "tool_failure"])
    graph.add_edge("tool_failure", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def create_agent(llm):
    graph = build_graph(llm)

    def run(query: str) -> dict:
        final = graph.invoke({"messages": [HumanMessage(query)]}, {"recursion_limit": 12})

        tool_calls_info = [
            {"name": call["name"], "args": call["args"]}
            for msg in final["messages"]
            if isinstance(msg, AIMessage)
            for call in msg.tool_calls
        ]

        return {
            "tool_called": bool(tool_calls_info),
            "tool_calls_info": tool_calls_info,
            "final_content": str(final["messages"][-1].text),
        }

    return run
