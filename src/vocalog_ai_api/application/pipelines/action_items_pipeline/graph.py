from langgraph.graph import StateGraph, START, END
from vocalog_ai_api.application.pipelines.action_items_pipeline.state import ActionItemsState
from vocalog_ai_api.application.pipelines.action_items_pipeline.nodes import extract_actions

# Define the graph
builder = StateGraph(ActionItemsState)

# Add Nodes
builder.add_node("extract_actions", extract_actions)

# Define Edges
builder.add_edge(START, "extract_actions")
builder.add_edge("extract_actions", END)

# Compile Pipeline
action_items_graph = builder.compile()
