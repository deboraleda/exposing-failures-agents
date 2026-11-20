
import os
from typing import Annotated, TypedDict
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

load_dotenv()

TOOL_TIMEOUT_S = 15

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str) -> dict | str:
    """Retrieve the current weather conditions for a city from the wttr.in API.
    Returns the current weather data, or a message starting with 'Error:' on failure."""
    try:
        resp = requests.get(f"https://wttr.in/{quote(city.strip())}?format=j1", timeout=TOOL_TIMEOUT_S)
    except requests.Timeout:
        return f"Error: the weather request for '{city}' timed out."
    except requests.RequestException as exc:
        return f"Error: the weather request failed: {exc}"
    if resp.status_code != 200:
        return f"Error: no weather data for '{city}' (HTTP {resp.status_code}); the city may be invalid."
    try:
        data = resp.json()
        current = data["current_condition"][0]
    except (ValueError, KeyError, IndexError):
        return f"Error: no weather data for '{city}'; the city may be invalid."
    area = (data.get("nearest_area") or [{}])[0]
    return {
        "location": ", ".join(
            a[0]["value"] for k in ("areaName", "region", "country") if (a := area.get(k))
        ),
        "description": current["weatherDesc"][0]["value"],
        "temperature_c": current["temp_C"],
        "feels_like_c": current["FeelsLikeC"],
        "humidity_pct": current["humidity"],
        "wind_kmph": current["windspeedKmph"],
        "observation_time_utc": current["observation_time"],
    }


@tool
def search_destination_news(destination: str, query: str) -> list[dict] | str:
    """Search recent news, safety advisories and current events about a destination
    with the Tavily Search API. Returns up to 5 results with title, source URL,
    content and publication date, or a message starting with 'Error:' on failure."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: missing Tavily credentials (TAVILY_API_KEY)."
    try:
        response = TavilyClient(api_key=api_key).search(
            query=f"{destination} {query}", time_range="month", max_results=5, timeout=TOOL_TIMEOUT_S
        )
    except Exception as exc:  # invalid key, request failure, timeout
        return f"Error: the news search failed: {exc}"
    return [
        {
            "title": r.get("title", ""),
            "source": r.get("url", ""),
            "content": r.get("content", ""),
            "published_date": r.get("published_date"),
        }
        for r in response.get("results", [])[:5]
    ]


TOOLS = [get_weather, search_destination_news]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a travel destination assistant that helps users explore destinations before deciding to travel.
Use get_weather when the user asks about current weather.
Use search_destination_news when the user asks about news, safety, travel advisories or current events.
Answer other travel questions directly, without tools.
Each execution may invoke at most one tool.
Base answers about weather and news only on the tool results.
"""

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _is_error(message: AnyMessage) -> bool:
    return isinstance(message, ToolMessage) and message.text.startswith("Error:")


def build_graph(llm):
    llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    def call_model(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    def after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def after_tools(state: AgentState) -> str:
        return END if _is_error(state["messages"][-1]) else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    return graph.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def create_agent(llm):
    """Build the agent with the given model; run(query) returns the tool calls and the final answer."""
    graph = build_graph(llm)

    def run(query: str) -> dict:
        final = graph.invoke({"messages": [HumanMessage(query)]}, {"recursion_limit": 8})

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
