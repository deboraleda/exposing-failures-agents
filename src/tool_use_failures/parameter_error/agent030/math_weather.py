import math
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


# Deterministic weather data for the destinations returned by select_location.
WEATHER = {
    "curitiba": "18 °C, cloudy with light drizzle",
    "lisbon": "22 °C, sunny",
    "tokyo": "15 °C, clear skies",
}

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _valid_amount(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


@tool
def calculate_trip_budget(daily_budget: float, days: int, extra_amount: float = 0) -> dict | str:
    """Calculate the base budget (daily budget x days) and the total budget
    (base budget + extra or reserve amount) of a trip. Returns the two values,
    or a message starting with 'Error:' if the calculation cannot be done."""
    if not _valid_amount(daily_budget):
        return "Error: the daily budget must be a non-negative number."
    if not isinstance(days, int) or days < 1:
        return "Error: the number of days must be a positive integer."
    if not _valid_amount(extra_amount):
        return "Error: the extra amount must be a non-negative number."
    base = daily_budget * days
    return {"base_budget": base, "total_budget": base + extra_amount}


@tool
def select_location(total_budget: float) -> str:
    """Select the trip destination for a total budget. Returns the destination,
    or a message starting with 'Error:' if no destination can be selected."""
    if not _valid_amount(total_budget):
        return "Error: the total budget must be a non-negative number."
    if total_budget < 500:
        return "Curitiba"
    if total_budget <= 1000:
        return "Lisbon"
    return "Tokyo"


@tool
def get_weather(location: str) -> str:
    """Get the weather for a destination. Returns the weather information, a
    message saying that no weather data is available for the destination, or a
    message starting with 'Error:' if the query cannot be completed."""
    if not isinstance(location, str) or not location.strip():
        return "Error: no destination was provided for the weather query."
    weather = WEATHER.get(location.strip().casefold())
    if weather is None:
        return f"No weather data available for {location.strip()}."
    return f"Weather in {location.strip()}: {weather}."


TOOLS = [calculate_trip_budget, select_location, get_weather]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an assistant for simple trip-budget and destination-weather inquiries.
Use the tools in sequence: calculate_trip_budget with the daily budget, number of days and optional extra or reserve amount from the request (zero if not given); select_location with the returned total budget; get_weather with the returned destination.
If the request is lacks the daily budget, the number of days or a request for the destination's weather, or is not this kind of inquiry, do not use tools: explain why and ask for what is missing.
Answer with the total budget, the selected destination and its weather, using only the tool results. If a step fails, stop and give the results already obtained and say what could not be obtained.
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

    def after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
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
            "final_content": final["messages"][-1].text,
        }

    return run
