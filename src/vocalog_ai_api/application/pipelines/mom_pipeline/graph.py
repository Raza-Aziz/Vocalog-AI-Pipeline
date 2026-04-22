from langgraph.graph import START, END, StateGraph
from vocalog_ai_api.application.pipelines.mom_pipeline.state import MoMGraphState
from vocalog_ai_api.application.pipelines.mom_pipeline.nodes import generate_markdown_mom, generate_mom
from vocalog_ai_api.infrastructure.database.checkpointer import checkpointer

builder = StateGraph(MoMGraphState)

builder.add_node("generate_mom", generate_mom)
builder.add_node("generate_markdown_mom", generate_markdown_mom)

builder.add_edge(START, "generate_mom")
builder.add_edge("generate_mom", "generate_markdown_mom")
builder.add_edge("generate_markdown_mom", END)

# Compiled with SQLite checkpointer — every invocation is persisted by thread_id.
# Pass config={"configurable": {"thread_id": "<user_id>:<meeting_id>"}} at call-site.
mom_graph = builder.compile(checkpointer=checkpointer)