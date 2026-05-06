from langgraph.graph import StateGraph, START, END

from vocalog_ai_api.application.pipelines.project_analysis_pipeline.state import ProjectAnalysisState
from vocalog_ai_api.application.pipelines.project_analysis_pipeline.nodes import (
    ingest_all_meetings,
    detect_conflicts,
)


def _build_graph():
    builder = StateGraph(ProjectAnalysisState)

    builder.add_node("ingest", ingest_all_meetings)
    builder.add_node("analyze", detect_conflicts)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "analyze")
    builder.add_edge("analyze", END)

    return builder.compile()


project_analysis_graph = _build_graph()
