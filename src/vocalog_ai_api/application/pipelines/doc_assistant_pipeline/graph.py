from langgraph.graph import StateGraph, START, END

from vocalog_ai_api.application.pipelines.doc_assistant_pipeline.state import DocAssistantState
from vocalog_ai_api.application.pipelines.doc_assistant_pipeline.nodes import (
    load_document_state,
    retrieve_project_context,
    generate_answer,
)


def _build_graph():
    builder = StateGraph(DocAssistantState)

    builder.add_node("load_state", load_document_state)
    builder.add_node("retrieve", retrieve_project_context)
    builder.add_node("answer", generate_answer)

    builder.add_edge(START, "load_state")
    builder.add_edge("load_state", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


doc_assistant_graph = _build_graph()
