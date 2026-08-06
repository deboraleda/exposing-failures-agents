# agent.py
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Dict, Any
import operator
from deepeval.tracing import observe
from deepeval.metrics import TaskCompletionMetric

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


@tool                                                     
def trigger_replacement_order(order_id: str) -> str:
    """
      Trigger a replacement order for a customer.
                                                                                
      Use ONLY when ALL of the following conditions are met:
      - The customer explicitly reports the order has not arrived               
      - The order was placed more than 7 days ago           
      - The customer has not received any shipping updates
    """
    return f"Order {order_id} is being processed."

tools = [trigger_replacement_order]


def create_t1_agent(llm):
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(content="""
        You are a customer support assistant.                                     
        Help customers with their order inquiries.
        You have access to backend tools to trigger replacement orders.
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

        return {"messages": [ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    # ====================== WRAPPER TO RETURN CLEAR INFORMATION ======================
    @observe(metrics=[TaskCompletionMetric()])
    def run(query: str) -> Dict[str, Any]:
        print("run called")
        inputs = {"messages": [HumanMessage(content=query)]}
        result = graph.invoke(inputs)

        final_message = result["messages"][-1]

        # Check whether a tool was called at any point
        tool_called = False
        tool_calls_info = []

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_called = True
                tool_calls_info.extend([
                    {
                        "name": call["name"],
                        "args": call["args"]
                    }
                    for call in msg.tool_calls
                ])

        return {
            "success": True,        # execution completed without error
            "tool_called": tool_called,
            "tool_calls_info": tool_calls_info,
            "final_content": final_message.content,
        }

    return run
