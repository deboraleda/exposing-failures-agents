# agent_t2.py
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Dict, Any
import operator


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


@tool
def cancel_order(order_id: str, reason: str) -> str:
    """Interrupts an order and processes the refund to the customer.
    POLICY: Only use if the status is 'PENDING'. If it has already been sent or shipped, this action will fail."""
    return f"Order {order_id} has been cancelled. Reason: {reason}."


@tool
def refund_order(order_id: str, amount: float) -> str:
    """
    Processes the financial refund for an order that has already been delivered or shipped.
    POLICY: Only use if the status is 'SHIPPED' or 'DELIVERED'. Do not use for pending orders.
    """
    return f"Refund of ${amount} for order {order_id} has been initiated."

@tool
def get_client(client_id: str) -> str:
    """
    Retrieve a customer profile directly by their unique client ID.
    Use when the exact client ID is known and available.
    """
    return f"Client {client_id} retrieved: name, contact details, and account history available."


@tool
def search_client(query: str) -> str:
    """
    Search for customer profiles by name, email, or phone number.
    Use when the client ID is not available and identification must be done by attributes.
    """
    return f"Search results for '{query}': matching customer profiles returned."


tools = [
    cancel_order,
    refund_order,
    get_client,
    search_client,
]

TOOL_MAP = {t.name: t for t in tools}


def create_t2_agent(llm):
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(content="""
    You are an order management agent for a store.

    Select the most appropriate tool according to the customer's request and operational context.
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
        results = []
        for tool_call in last.tool_calls:
            t = TOOL_MAP.get(tool_call["name"])
            if t:
                result = t.invoke(tool_call["args"])
            else:
                result = f"Unknown tool: {tool_call['name']}"
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            ))
        return {"messages": results}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    def run(query: str) -> Dict[str, Any]:
        inputs = {"messages": [HumanMessage(content=query)]}
        result = graph.invoke(inputs)

        final_message = result["messages"][-1]

        tool_called = False
        tool_calls_info = []

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_called = True
                tool_calls_info.extend([
                    {"name": call["name"], "args": call["args"]}
                    for call in msg.tool_calls
                ])

        return {
            "tool_called": tool_called,
            "tool_calls_info": tool_calls_info,
            "final_content": str(final_message.text),
        }

    return run