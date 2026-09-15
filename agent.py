from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition

from nodes import call_model, tool_node

graph = StateGraph(MessagesState)

graph.add_node("llm", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    tools_condition,
)

graph.add_edge("tools", "llm")