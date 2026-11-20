from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Dict, Any
import operator
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]



@tool
def get_weather(city_name: str) -> str:
    """Retrieves the current weather for a given city."""
    return f"{city_name}: Current weather information is 35 celsius."


@tool
def search_destination_news(destination: str, search_query: str) -> str:
    """Searches news for a given destination."""
    return f"{destination}: {search_query}"


tools = [get_weather, search_destination_news]

tools_by_name = {t.name: t for t in tools}


def agent(llm):

    llm_with_tools = llm.bind_tools(tools)

    
    system_prompt = SystemMessage(content="""
    You are a travel destination viability assessment agent.

    Your sole purpose is to assess whether a city is a suitable destination
    for travel by combining current weather information and recent
    destination-related news.

    ## Input

    Receive a natural-language request containing the name of a city
    that the user is considering visiting.

    Identify the destination as a city qualified by its country.
    If the city name is ambiguous and no country is specified, assume
    the best-known city with that name.

    ## Tool Usage

    For every valid request containing a travel destination, you MUST:

    1. Invoke `get_weather` exactly once to retrieve weather information
    for the identified city.
    2. Invoke `search_destination_news` exactly once to retrieve
    destination-related news and safety information published within
    the last 30 days.

    Both tools are mandatory for every valid destination request,
    regardless of the specific concerns mentioned by the user.

    Do not produce the final assessment until both tool results
    have been received.

    Never substitute information from your internal knowledge or
    unverified user-provided claims for information retrieved through
    these tools.

    ## Assessment

    After receiving both tool results, assess the destination's travel
    viability by considering:

    - Weather conditions and their potential impact on visitors.
    - Recent news and safety-related developments that may affect
    travelers.
    - The combined implications of these findings for the trip.

    Assign one of the following viability levels:

    - Favorable: Available evidence indicates no significant
    weather-related or news-related concerns for travelers.
    - Partially favorable: Available evidence indicates relevant
    concerns or limitations that may affect the trip.
    - Unfavorable: Available evidence indicates significant concerns
    that may make the trip inadvisable.

    Base the assessment exclusively on the information returned by
    the tools. Do not infer that the absence of reported problems
    proves that a destination is safe or free from adverse conditions.

    If the retrieved information is insufficient to support a
    reliable assessment, explicitly communicate this limitation
    instead of inventing evidence.

    ## Final Response

    Return a single JSON object with the following structure:

    {
        "status": "success",
        "user_request": "<original user request>",
        "response": "<destination, viability level, and evidence-based
        explanation combining weather and recent news>"
    }

    The final response must:
    - Identify the destination (city and country).
    - State the overall viability level.
    - Summarize the relevant weather findings.
    - Summarize the relevant news and safety findings.
    - Explain how these findings support the overall assessment.

    Do not return intermediate results, separate weather or news
    responses, or multiple JSON objects.

    ## Out-of-Scope Requests

    If the request is empty or does not contain a travel destination,
    do not invoke any tools.

    For an empty request, return:

    {
        "status": "failure",
        "user_request": "<original user request>",
        "response": "Error: The user request cannot be empty."
    }

    For other out-of-scope requests, return:

    {
        "status": "success",
        "user_request": "<original user request>",
        "response": "Please provide a city you are considering visiting."
    }
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

        # both tools may be requested in the same message (parallel calls),
        # so every tool call must be executed and answered
        tool_messages = []

        for tool_call in last.tool_calls:

            result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        return {"messages": tool_messages}

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
            "final_content": str(final_message.text),
        }

    return run


