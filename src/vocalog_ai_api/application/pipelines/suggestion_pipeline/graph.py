from langgraph.graph import StateGraph, START, END

from vocalog_ai_api.application.pipelines.suggestion_pipeline.state import SuggestionState
from vocalog_ai_api.application.pipelines.suggestion_pipeline.nodes import (
    load_document_state,
    ingest_new_meeting,
    analyze_sections,
)


def _build_graph():
    builder = StateGraph(SuggestionState)

    builder.add_node("load_state", load_document_state)
    builder.add_node("ingest", ingest_new_meeting)
    builder.add_node("analyze", analyze_sections)

    builder.add_edge(START, "load_state")
    builder.add_edge("load_state", "ingest")
    builder.add_edge("ingest", "analyze")
    builder.add_edge("analyze", END)

    return builder.compile()


suggestion_graph = _build_graph()
