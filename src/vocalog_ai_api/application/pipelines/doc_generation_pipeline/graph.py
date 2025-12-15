# graph/document_graph.py
from langgraph.graph import StateGraph, END
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.schemas import GraphState
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes import ( apply_hitl_feedback,
                                                                                    assemble_document,
                                                                                    retrieve_context,
                                                                                    start_document,
                                                                                    load_next_section,
                                                                                    build_prompt,
                                                                                    generate_section,
                                                                                    persist_draft,
                                                                                    emit_hitl_event,
                                                                                    wait_for_hitl,
                                                                                    finalize_section,
                                                                                    check_remaining_sections,
                                                                                    export_document
) 

graph = StateGraph(GraphState)

graph.add_node("start_document", start_document)
graph.add_node("load_next_section", load_next_section)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("build_prompt", build_prompt)
graph.add_node("generate_section", generate_section)
graph.add_node("persist_draft", persist_draft)
graph.add_node("emit_hitl_event", emit_hitl_event)
graph.add_node("wait_for_hitl", wait_for_hitl)
graph.add_node("apply_hitl_feedback", apply_hitl_feedback)
graph.add_node("finalize_section", finalize_section)
graph.add_node("check_remaining_sections", check_remaining_sections)
graph.add_node("assemble_document", assemble_document)
graph.add_node("export_document", export_document)

graph.set_entry_point("start_document")

graph.add_edge("start_document", "load_next_section")
graph.add_edge("load_next_section", "retrieve_context")
graph.add_edge("retrieve_context", "build_prompt")
graph.add_edge("build_prompt", "generate_section")
graph.add_edge("generate_section", "persist_draft")
graph.add_edge("persist_draft", "emit_hitl_event")
graph.add_edge("emit_hitl_event", "wait_for_hitl")

# Resume after HITL
graph.add_edge("wait_for_hitl", "apply_hitl_feedback")
graph.add_edge("apply_hitl_feedback", "finalize_section")
graph.add_edge("finalize_section", "check_remaining_sections")

graph.add_conditional_edges(
    "check_remaining_sections",
    lambda state: "next" if state.current_section_index < len(state.sections) else "done",
    {
        "next": "load_next_section",
        "done": "assemble_document"
    }
)

graph.add_edge("assemble_document", "export_document")
graph.add_edge("export_document", END)

document_graph = graph.compile()
