from langgraph.graph import StateGraph, START, END

from vocalog_ai_api.application.pipelines.meeting_qa_pipeline.state import MeetingQAState
from vocalog_ai_api.application.pipelines.meeting_qa_pipeline.nodes import retrieve_context, generate_answer


def _build_graph() -> StateGraph:
    builder = StateGraph(MeetingQAState)
    builder.add_node("retrieve", retrieve_context)
    builder.add_node("answer", generate_answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


meeting_qa_graph = _build_graph()
