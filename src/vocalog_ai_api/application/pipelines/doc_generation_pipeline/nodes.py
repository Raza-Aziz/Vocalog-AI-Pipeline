"""
Document Generation pipeline nodes.

Each node is a pure function: DocumentGenerationState → dict (partial state update).
LangGraph merges the returned dict into the persisted checkpoint.

Strategy injection (Task 5 — Dynamic Persona):
  Every node that calls the LLM resolves the active DocumentStrategy from the
  state's `document_type` field.  This means:
  • The section outline is always strategy-specific (not hardcoded).
  • Initial draft prompts inject the strategy's persona + section focus.
  • Refinement prompts use the strategy's refinement persona.
"""

from langchain_core.messages import HumanMessage

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import (
    ingest_minutes,
    retrieve_context,
)
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import get_strategy
from vocalog_ai_api.infrastructure.llm_providers.groq import llm


def initialize_document(state: DocumentGenerationState) -> dict:
    """
    1. Ingests raw meeting minutes into the Qdrant vector store (idempotent).
    2. Resolves the document strategy and creates the section outline.

    The outline is determined entirely by the chosen strategy — no hardcoded
    section lists exist in this file.
    """
    print(f"--- Initializing Document (type={state['document_type']}) ---")

    session_id = state["thread_id"]
    raw_minutes = state["meeting_minutes"]
    strategy = get_strategy(state["document_type"])

    try:
        ingest_minutes(session_id, raw_minutes)
        print(f"Vectorised minutes for thread: {session_id}")
    except Exception as exc:
        print(f"Vectorisation error (non-fatal): {exc}")

    return {
        "sections_outline": strategy.sections,
        "current_section_index": 0,
        "final_document": [],
        "refinement_history": {},
        "is_complete": False,
        "pending_action": None,
        "feedback_notes": None,
        "current_section_content": "",
    }


def generate_section(state: DocumentGenerationState) -> dict:
    """
    Generates (or refines) the current section draft using RAG + strategy-aware LLM prompt.

    On an initial call  : builds an initial-draft prompt via strategy.build_initial_prompt().
    On a refine call    : builds a refinement prompt via strategy.build_refinement_prompt()
                          and records the previous draft + feedback in refinement_history.
    On a regenerate call: re-runs initial-draft logic (fresh attempt, no feedback injected).

    After generation the node clears pending_action and feedback_notes so the graph
    re-enters a clean interrupt state waiting for the next human decision.
    """
    current_idx = state["current_section_index"]
    sections = state["sections_outline"]

    if current_idx >= len(sections):
        return {"is_complete": True}

    section_title = sections[current_idx]
    session_id = state["thread_id"]
    feedback = state.get("feedback_notes")
    action = state.get("pending_action")
    history: dict = dict(state.get("refinement_history") or {})
    strategy = get_strategy(state["document_type"])

    print(
        f"--- Generating section '{section_title}' "
        f"[{state['document_type'].upper()}] action={action or 'initial'} ---"
    )

    # RAG retrieval — query by section title for relevance
    relevant_context = retrieve_context(session_id, query=section_title, limit=4)
    print(f"Context retrieved: {len(relevant_context)} chars")

    is_refinement = action == "refine" and feedback and state.get("current_section_content")

    # Record the outgoing draft + feedback in history before overwriting
    if is_refinement:
        str_idx = str(current_idx)
        if str_idx not in history:
            history[str_idx] = []
        history[str_idx].append(
            {"draft": state["current_section_content"], "feedback": feedback}
        )

    # Build prompt via strategy (Task 5 — dynamic persona injection)
    if is_refinement:
        prompt = strategy.build_refinement_prompt(
            section_title=section_title,
            current_draft=state["current_section_content"],
            feedback=feedback,
            context=relevant_context,
        )
    else:
        prompt = strategy.build_initial_prompt(
            section_title=section_title,
            context=relevant_context,
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    return {
        "current_section_content": content,
        "refinement_history": history,
        # Clear HITL fields — graph re-interrupts; API sets them again before next resume
        "pending_action": None,
        "feedback_notes": None,
    }


def process_approval(state: DocumentGenerationState) -> dict:
    """
    Moves the approved section to final_document, advances the section pointer,
    and determines whether the document is now complete.
    """
    print("--- Processing Approval ---")

    approved_section = {
        "title": state["sections_outline"][state["current_section_index"]],
        "content": state["current_section_content"],
    }
    updated_doc = list(state["final_document"]) + [approved_section]
    next_index = state["current_section_index"] + 1

    # Drop the completed section's refinement history (keep memory lean)
    history = dict(state.get("refinement_history") or {})
    history.pop(str(state["current_section_index"]), None)

    is_done = next_index >= len(state["sections_outline"])

    return {
        "final_document": updated_doc,
        "current_section_index": next_index,
        "current_section_content": "",
        "pending_action": None,
        "feedback_notes": None,
        "refinement_history": history,
        "is_complete": is_done,
    }
