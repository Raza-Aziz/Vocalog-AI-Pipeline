"""
Vocalog AI — FastAPI application.

Document Generation endpoints follow a thread_id / document_id persistence model:
  • document_id  = LangGraph thread_id = permanent key into the SQLite checkpoint store.
  • No in-memory session state.  Every request loads/saves state via the checkpointer.
  • The graph uses interrupt_after=["generate"] so each draft generation pauses and
    waits for human feedback before continuing.

Human-in-the-loop cycle (POST /generate-document → POST /provide-feedback loop):
  1. POST /generate-document  → allocates document_id, invokes graph (init + first draft).
  2. POST /provide-feedback   → injects pending_action (+feedback_notes) into checkpoint,
                                resumes graph, returns next draft or completion.
  3. GET /document-status/{id} → reads checkpoint state without advancing the graph.
"""

import uuid

from fastapi import FastAPI, HTTPException

from vocalog_ai_api.api.schemas import (
    TranscriptInput,
    MoMAndActionsResponse,
    DemoDocumentGenerationRequest,
    DemoSectionDraftResponse,
    SectionFeedbackRequest,
    DemoDocumentStatusResponse,
    ActionItemsForFrontendRequest,
    ActionItemsForFrontendResponse,
)
from vocalog_ai_api.application.pipelines.mom_pipeline.graph import mom_graph
from vocalog_ai_api.application.pipelines.action_items_pipeline.graph import action_items_graph
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import doc_gen_graph

app = FastAPI(title="Vocalog AI — Document Generation Module", version="2.0.0")


# ── Minutes of Meeting ───────────────────────────────────────────────────────

@app.post("/generate-mom", response_model=MoMAndActionsResponse)
def generate_minutes_of_meeting(data: TranscriptInput):
    """
    Generate Minutes of Meeting and extract action items from a transcript.
    Both graphs share the same thread_id so all outputs land in one SQLite session.
    """
    user_id = data.user_id or "anonymous"
    session_suffix = data.session_id or data.meeting_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{session_suffix}"}}
    mom_result = mom_graph.invoke({"raw_transcript": data.transcript.text}, config=config)
    ai_result = action_items_graph.invoke(
        {"transcript": data.transcript.text, "session_id": session_suffix},
        config=config,
    )
    actions = ai_result.get("extracted_actions", [])
    return MoMAndActionsResponse(
        meeting_minutes=mom_result.get("mom_markdown", ""),
        action_items=actions,
        total_count=len(actions),
    )


# ── Document Generation (persistence-first HITL) ─────────────────────────────

def _build_config(document_id: str) -> dict:
    return {"configurable": {"thread_id": document_id}}


def _state_to_draft_response(document_id: str, state: dict) -> DemoSectionDraftResponse:
    """Build the API response from a checkpoint state snapshot."""
    outline = state.get("sections_outline", [])
    idx = state.get("current_section_index", 0)
    is_complete = state.get("is_complete", False)
    doc_type = state.get("document_type", "srs")

    if is_complete:
        return DemoSectionDraftResponse(
            document_id=document_id,
            document_type=doc_type,
            section_title="Document Complete",
            content="All sections have been approved.",
            is_complete=True,
            refinement_count=0,
            message="Your document is ready.",
        )

    section_title = outline[idx] if idx < len(outline) else "Unknown Section"
    history = state.get("refinement_history", {})
    ref_count = len(history.get(str(idx), []))

    return DemoSectionDraftResponse(
        document_id=document_id,
        document_type=doc_type,
        section_title=section_title,
        content=state.get("current_section_content", ""),
        is_complete=False,
        refinement_count=ref_count,
        message="Review the section draft and provide feedback.",
    )


@app.post("/generate-document", response_model=DemoSectionDraftResponse)
async def start_document_generation(request: DemoDocumentGenerationRequest):
    """
    Allocates a new document_id (= LangGraph thread_id), ingests meeting minutes
    into the vector store, and generates the first section draft.

    The graph runs init → generate then interrupts, saving all state to SQLite.
    Returns the first section draft for human review.
    """
    document_id = str(uuid.uuid4())
    config = _build_config(document_id)

    initial_state = {
        "thread_id": document_id,
        "project_id": request.project_id,
        "document_type": request.document_type,
        "meeting_minutes": request.meeting_minutes,
        "sections_outline": [],
        "current_section_index": 0,
        "current_section_content": "",
        "pending_action": None,
        "feedback_notes": None,
        "refinement_history": {},
        "final_document": [],
        "is_complete": False,
    }

    print(f"Starting {request.document_type.upper()} generation — document_id={document_id}")
    result_state = doc_gen_graph.invoke(initial_state, config=config)

    return _state_to_draft_response(document_id, result_state)


@app.post("/provide-feedback", response_model=DemoSectionDraftResponse)
async def provide_feedback(request: SectionFeedbackRequest):
    """
    Inject human feedback into the persisted checkpoint and resume the graph.

    Actions:
      • approve     → graph saves section, advances to next (or marks complete).
      • refine      → graph regenerates same section incorporating feedback_notes.
      • regenerate  → graph regenerates same section fresh (no notes required).

    The document_id is the permanent key to the SQLite checkpoint — resilient to
    server restarts or long periods of inactivity.
    """
    document_id = request.document_id
    config = _build_config(document_id)

    # Load current checkpoint to validate session exists
    checkpoint = doc_gen_graph.get_state(config)
    if checkpoint is None or checkpoint.values is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    current = checkpoint.values
    doc_type = current.get("document_type", "srs")

    if current.get("is_complete"):
        return DemoSectionDraftResponse(
            document_id=document_id,
            document_type=doc_type,
            section_title="Document Complete",
            content="All sections have already been approved.",
            is_complete=True,
            refinement_count=0,
            message="Your document is ready.",
        )

    # Inject the human decision into the checkpoint before resuming
    state_update: dict = {"pending_action": request.action, "feedback_notes": None}
    if request.action == "refine":
        state_update["feedback_notes"] = request.feedback_notes

    doc_gen_graph.update_state(config, state_update)

    # Resume — graph continues from the interrupt point
    result_state = doc_gen_graph.invoke(None, config=config)

    return _state_to_draft_response(document_id, result_state)


@app.get("/document-status/{document_id}", response_model=DemoDocumentStatusResponse)
async def get_document_status(document_id: str):
    """
    Read the current document state directly from the SQLite checkpoint.
    Does not advance the graph.
    """
    config = _build_config(document_id)
    checkpoint = doc_gen_graph.get_state(config)

    if checkpoint is None or checkpoint.values is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    state = checkpoint.values
    outline = state.get("sections_outline", [])
    idx = state.get("current_section_index", 0)
    is_complete = state.get("is_complete", False)

    current_title = None
    if outline and not is_complete and idx < len(outline):
        current_title = outline[idx]

    return DemoDocumentStatusResponse(
        document_id=document_id,
        document_type=state.get("document_type", "srs"),
        status="completed" if is_complete else "in_progress",
        current_section_title=current_title,
        completed_sections=len(state.get("final_document", [])),
        total_sections=len(outline),
    )


# ── Action Items ─────────────────────────────────────────────────────────────

@app.post("/action-items/extract-for-frontend", response_model=ActionItemsForFrontendResponse)
async def extract_action_items_for_frontend(request: ActionItemsForFrontendRequest):
    """
    Extracts action items for direct frontend review (no side-effects).
    """
    user_id = request.user_id or "anonymous"
    session_id = request.session_id or request.meeting_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    result = action_items_graph.invoke(
        {"transcript": request.transcript.text, "session_id": session_id},
        config=config,
    )
    actions = result.get("extracted_actions", [])
    return ActionItemsForFrontendResponse(actions=actions, total_count=len(actions))



# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
