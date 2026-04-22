from langgraph.graph import StateGraph, START, END
from vocalog_ai_api.application.pipelines.action_items_pipeline.state import ActionItemsState
from vocalog_ai_api.application.pipelines.action_items_pipeline.nodes import extract_actions
from vocalog_ai_api.infrastructure.database.checkpointer import checkpointer

builder = StateGraph(ActionItemsState)

builder.add_node("extract_actions", extract_actions)

builder.add_edge(START, "extract_actions")
builder.add_edge("extract_actions", END)

# Compiled with SQLite checkpointer — every invocation is persisted by thread_id.
# Pass config={"configurable": {"thread_id": "<user_id>:<session_id>"}} at call-site.
action_items_graph = builder.compile(checkpointer=checkpointer)
