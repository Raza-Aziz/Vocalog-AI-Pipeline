from langgraph.graph import StateGraph, START, END

from vocalog_ai_api.application.pipelines.gap_analysis_pipeline.state import GapAnalysisState
from vocalog_ai_api.application.pipelines.gap_analysis_pipeline.nodes import (
    load_document_state,
    analyze_gaps,
)


def _build_graph():
    builder = StateGraph(GapAnalysisState)

    builder.add_node("load_state", load_document_state)
    builder.add_node("analyze", analyze_gaps)

    builder.add_edge(START, "load_state")
    builder.add_edge("load_state", "analyze")
    builder.add_edge("analyze", END)

    return builder.compile()


gap_analysis_graph = _build_graph()
