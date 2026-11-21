from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Dict, Any
import operator


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


@tool
def book_flight(
    traveler_name: str,
    departure_code: str,
    arrival_code: str,
    travel_date: str,
    journey_class: str,
    document_ref: str,
    record_locator: str,
    routing_ref: str,
    return_date: str | None = None,
) -> dict:
    """
    Expected payload format:

    {
        "traveler_name": str,
        "departure_code": str,
        "arrival_code": str,
        "travel_date": str,
        "journey_class": "one_way" | "round_trip",
        "document_ref": str,
        "return_date": str | None,
        "record_locator": str,
        "routing_ref": str
    }
    """

    payload = {
        "traveler_name": traveler_name,
        "departure_code": departure_code,
        "arrival_code": arrival_code,
        "travel_date": travel_date,
        "journey_class": journey_class,
        "document_ref": document_ref,
        "return_date": return_date,
        "record_locator": record_locator,
        "routing_ref": routing_ref,
    }

    allowed_fields = {
        "traveler_name",
        "departure_code",
        "arrival_code",
        "travel_date",
        "journey_class",
        "document_ref",
        "return_date",
        "record_locator",
        "routing_ref",
    }

    # ================================================================
    # Only return_date is optional
    # ================================================================

    required_fields = {
        "traveler_name",
        "departure_code",
        "arrival_code",
        "travel_date",
        "journey_class",
        "document_ref",
        "record_locator",
        "routing_ref",
    }

    # =====================================================================
    # Validation
    # =====================================================================

    missing_fields = [
        field
        for field in required_fields
        if payload.get(field) is None
    ]

    unexpected_fields = [
        field
        for field in payload
        if field not in allowed_fields
    ]

    invalid_journey_class = (
        payload.get("journey_class")
        not in ("one_way", "round_trip")
    )

    # ---------------------------------------------------------------
    # round_trip requires return_date
    # ---------------------------------------------------------------

    round_trip_missing_return = (
        payload.get("journey_class") == "round_trip"
        and not payload.get("return_date")
    )

    # ---------------------------------------------------------------
    # one_way must not include return_date
    # ---------------------------------------------------------------

    one_way_with_return = (
        payload.get("journey_class") == "one_way"
        and payload.get("return_date") is not None
    )

    valid = (
        not missing_fields
        and not unexpected_fields
        and not invalid_journey_class
        and not round_trip_missing_return
        and not one_way_with_return
    )

    # =====================================================================
    # Structured return
    # =====================================================================

    return {
        "valid": valid,

        "received_payload": payload,

        "validation": {
            "missing_fields": missing_fields,
            "unexpected_fields": unexpected_fields,
            "invalid_journey_class": invalid_journey_class,
            "round_trip_missing_return": round_trip_missing_return,
            "one_way_with_return": one_way_with_return,
        },

        "message": (
            f"Flight booked for "
            f"{payload.get('traveler_name')} "
            f"from {payload.get('departure_code')} "
            f"to {payload.get('arrival_code')}"
        ) if valid else "Payload validation failed."
    }


tools = [book_flight]


def agent(llm):

    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(content="""
        You are a flight booking agent.

        Use the book_flight tool to book flights.

        Always call the tool immediately using ONLY the data provided.

        The following fields are REQUIRED in every tool call:
        - traveler_name
        - departure_code
        - arrival_code
        - travel_date
        - journey_class
        - document_ref
        - record_locator
        - routing_ref

        return_date is OPTIONAL only for one_way flights.
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

        result = tools[0].invoke(tool_call["args"])

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
            "final_content": str(final_message.text),
        }

    return run