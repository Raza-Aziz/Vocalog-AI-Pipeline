"""
Document Generation LangGraph pipeline with SQLite-backed persistence.

Graph topology
──────────────
  START → init → generate ──[interrupt]──► (API reads/writes state)
                    ▲                           │
                    │  (refine/regenerate)       │ pending_action="approve"
                    │                           ▼
                    └──────────────── save_approved ──► END (if complete)
                                                 │
                                                 └──► generate (next section)

Human-in-the-loop flow
──────────────────────
1. Client calls POST /generate-document  → graph runs init → generate, then
   interrupts.  Checkpoint holds the current draft.
2. Client calls POST /provide-feedback:
   • "approve"     → API calls graph.update_state(pending_action="approve"),
                      then graph.invoke(None, config).  Graph runs save_approved,
                      then either ends or generates the next section (new interrupt).
   • "refine"      → API injects feedback_notes + pending_action="refine" into
                      the checkpoint, resumes.  Graph re-runs generate (same section).
   • "regenerate"  → same as refine but with no specific feedback_notes.
3. Repeat until is_complete=True.
"""

from langgraph.graph import StateGraph, END
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes import (
    initialize_document,
    generate_section,
    process_approval,
)
from vocalog_ai_api.infrastructure.database.checkpointer import checkpointer as _default_checkpointer


# ── Routing functions ────────────────────────────────────────────────────────

def _route_after_generate(state: DocumentGenerationState) -> str:
    """
    Called on graph resume (after the interrupt that follows every generate run).
    The API has already injected pending_action into the checkpoint before resuming.
    """
    action = state.get("pending_action")
    if action == "approve":
        return "save_approved"
    # "refine", "regenerate", or None (first run before any human action)
    return "generate"


def _route_after_save(state: DocumentGenerationState) -> str:
    if state.get("is_complete"):
        return END
    return "generate"


# ── Graph assembly ───────────────────────────────────────────────────────────

def create_doc_gen_graph(checkpointer=None):
    """
    Build and compile the document-generation graph.

    Args:
        checkpointer: Optional LangGraph checkpointer.  Pass a custom one
                      (e.g. an in-memory SqliteSaver) for testing so that
                      tests never touch the production database.
                      Defaults to the shared application checkpointer.
    """
    _cp = checkpointer if checkpointer is not None else _default_checkpointer

    workflow = StateGraph(DocumentGenerationState)

    workflow.add_node("init", initialize_document)
    workflow.add_node("generate", generate_section)
    workflow.add_node("save_approved", process_approval)

    workflow.set_entry_point("init")
    workflow.add_edge("init", "generate")

    workflow.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"save_approved": "save_approved", "generate": "generate"},
    )
    workflow.add_conditional_edges(
        "save_approved",
        _route_after_save,
        {"generate": "generate", END: END},
    )

    return workflow.compile(
        checkpointer=_cp,
        interrupt_after=["generate"],   # pause after every draft for human review
    )


# Module-level singleton — import this in the API layer
doc_gen_graph = create_doc_gen_graph()
