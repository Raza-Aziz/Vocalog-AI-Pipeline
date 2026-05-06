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
    MeetingQARequest,
    MeetingQAResponse,
    CitedSource,
    DocAssistantRequest,
    DocAssistantResponse,
    DocAssistantCitation,
    DocAssistantDocumentSnapshot,
    Suggestion,
    SuggestUpdatesRequest,
    SuggestUpdatesResponse,
    GapAnalysisRequest,
    GapAnalysisResponse,
    GapItemSchema,
    GapOptionSchema,
    ResolveGapRequest,
    ResolveGapResponse,
    ProjectAnalysisRequest,
    ProjectAnalysisResponse,
    ConflictItemSchema,
    AlignedTopicSchema,
    ConflictResolutionInput,
    ResolveConflictRequest,
    ResolveConflictResponse,
    AcceptSuggestionRequest,
    AcceptSuggestionResponse,
    RejectSuggestionRequest,
    RejectSuggestionResponse,
)
from vocalog_ai_api.application.pipelines.mom_pipeline.graph import mom_graph
from vocalog_ai_api.application.pipelines.action_items_pipeline.graph import action_items_graph
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import doc_gen_graph
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import (
    ingest_transcript_for_qa,
    ingest_mom_for_qa,
)
from vocalog_ai_api.application.pipelines.meeting_qa_pipeline.graph import meeting_qa_graph
from vocalog_ai_api.application.pipelines.doc_assistant_pipeline.graph import doc_assistant_graph
from vocalog_ai_api.application.pipelines.suggestion_pipeline.graph import suggestion_graph
from vocalog_ai_api.application.pipelines.gap_analysis_pipeline.graph import gap_analysis_graph
from vocalog_ai_api.application.pipelines.gap_analysis_pipeline.nodes import resolve_gap_to_suggestion
from vocalog_ai_api.application.pipelines.project_analysis_pipeline.graph import project_analysis_graph

app = FastAPI(title="Vocalog AI — Document Generation Module", version="2.0.0")


# ── Minutes of Meeting ───────────────────────────────────────────────────────

@app.post("/generate-mom", response_model=MoMAndActionsResponse)
def generate_minutes_of_meeting(data: TranscriptInput):
    """
    Generate Minutes of Meeting and extract action items from a transcript.

    When a meeting_id is supplied the raw transcript and generated MoM are
    automatically ingested into the vector store so the /meeting-qa endpoint
    can answer questions about this specific meeting.
    """
    user_id = data.user_id or "anonymous"
    session_suffix = data.session_id or data.meeting_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{session_suffix}"}}

    mom_result = mom_graph.invoke({"raw_transcript": data.transcript.text}, config=config)
    ai_result = action_items_graph.invoke(
        {"transcript": data.transcript.text, "session_id": session_suffix},
        config=config,
    )

    mom_markdown: str = mom_result.get("mom_markdown", "")

    # Auto-ingest both sources when a meeting_id is present so Meeting Q&A works
    if data.meeting_id:
        ingest_transcript_for_qa(data.meeting_id, data.transcript.text)
        if mom_markdown:
            ingest_mom_for_qa(data.meeting_id, mom_markdown)

    actions = ai_result.get("extracted_actions", [])
    return MoMAndActionsResponse(
        meeting_minutes=mom_markdown,
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
        "meeting_sources": [
            {"meeting_id": s.meeting_id, "content": s.content}
            for s in request.sources
        ],
        "conflict_resolutions": [
            {
                "topic": r.topic,
                "authoritative_meeting_id": r.authoritative_meeting_id,
                "authoritative_position": r.authoritative_position,
            }
            for r in (request.conflict_resolutions or [])
        ],
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



# ── Meeting Q&A ──────────────────────────────────────────────────────────────

@app.post("/meeting-qa", response_model=MeetingQAResponse)
async def meeting_qa(request: MeetingQARequest):
    """
    Answer a natural-language question about a specific meeting using RAG.

    Retrieves relevant chunks from both the raw transcript and the generated
    Minutes of Meeting (MoM), cross-references them, and returns a grounded
    answer with inline citations (MOM-N / TRANSCRIPT-N).

    The query is scoped strictly to the supplied meeting_id — no data from
    other meetings in the system will influence the response.

    The meeting must have been previously processed via POST /generate-mom
    with the same meeting_id for its content to be available.
    """
    result = meeting_qa_graph.invoke({
        "meeting_id": request.meeting_id,
        "question": request.question,
        "transcript_chunks": [],
        "mom_chunks": [],
        "answer": "",
        "citations": [],
    })

    citations = [CitedSource(**c) for c in result.get("citations", [])]

    return MeetingQAResponse(
        meeting_id=request.meeting_id,
        question=request.question,
        answer=result.get("answer", ""),
        citations=citations,
    )


# ── Document Generation Assistant ────────────────────────────────────────────

@app.post("/doc-assistant", response_model=DocAssistantResponse)
async def doc_assistant(request: DocAssistantRequest):
    """
    Intelligent companion for users navigating the document generation process.

    State-aware: reads the live document checkpoint (approved sections, active
    draft, refinement/feedback history) directly from SQLite so every answer
    reflects the document's real-time state.

    Project-wide RAG: runs multi-probe hybrid retrieval (Vector + BM25 + RRF +
    CrossEncoder) across all meeting transcripts and Minutes of Meeting ingested
    for the project, then cross-references them with the document content.

    Useful questions:
      - "How did we arrive at this requirement?"
      - "Which meeting decided on this architecture?"
      - "Why was Section 2 rewritten?"
      - "What did the team say about the auth system?"
    """
    initial_state = {
        "document_id": request.document_id,
        "question": request.question,
        # doc-state fields — populated by load_document_state node
        "project_id": "",
        "document_type": "",
        "sections_outline": [],
        "current_section_index": 0,
        "current_section_content": "",
        "final_document": [],
        "refinement_history": {},
        "meeting_sources": [],
        "is_complete": False,
        "document_found": False,
        # retrieval + output fields
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
    }

    result = doc_assistant_graph.invoke(initial_state)

    # Build the document snapshot for the response envelope
    sections_outline = result.get("sections_outline", [])
    current_idx = result.get("current_section_index", 0)
    is_complete = result.get("is_complete", False)
    document_found = result.get("document_found", False)

    snapshot = DocAssistantDocumentSnapshot(
        document_type=result.get("document_type", ""),
        status=(
            "not_found" if not document_found
            else "completed" if is_complete
            else "in_progress"
        ),
        approved_sections=len(result.get("final_document", [])),
        total_sections=len(sections_outline),
        active_section=(
            sections_outline[current_idx]
            if sections_outline and current_idx < len(sections_outline) and not is_complete
            else None
        ),
        source_meetings=[
            s.get("meeting_id", "") for s in result.get("meeting_sources", []) if s.get("meeting_id")
        ],
    )

    citations = [DocAssistantCitation(**c) for c in result.get("citations", [])]

    return DocAssistantResponse(
        document_id=request.document_id,
        question=request.question,
        answer=result.get("answer", ""),
        citations=citations,
        document_snapshot=snapshot,
    )


# ── Inline Suggestion & Synchronization ──────────────────────────────────────

@app.post("/suggest-updates", response_model=SuggestUpdatesResponse)
async def suggest_updates(request: SuggestUpdatesRequest):
    """
    Synchronize an existing document with a newly added meeting transcript.

    Ingests the new meeting into the project knowledge base, then performs a
    surgical section-by-section analysis of every approved document section.
    Returns a list of localized Suggestions — each containing the exact original
    text to modify, the proposed replacement, a rationale, and the source meeting.

    Suggestions are structured for a "Suggesting Mode" UX (Google Docs / Tiptap
    annotations): clients can present each change inline and let the user
    accept or reject it individually without touching the rest of the document.

    Only approved (finalized) sections are analyzed. The active draft and
    unapproved sections are not modified.
    """
    initial_state = {
        "document_id": request.document_id,
        "new_meeting_id": request.new_meeting_id,
        "new_meeting_content": request.new_meeting_content,
        "project_id": "",
        "document_type": "",
        "sections_outline": [],
        "final_document": [],
        "document_found": False,
        "suggestions": [],
    }

    result = suggestion_graph.invoke(initial_state)

    if not result.get("document_found"):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document_id}' not found. Verify the document_id.",
        )

    raw_suggestions = result.get("suggestions", [])
    suggestions = [Suggestion(**s) for s in raw_suggestions]

    return SuggestUpdatesResponse(
        document_id=request.document_id,
        new_meeting_id=request.new_meeting_id,
        suggestions=suggestions,
        total_count=len(suggestions),
    )


# ── Gap Analysis ──────────────────────────────────────────────────────────────

@app.post("/gap-analysis", response_model=GapAnalysisResponse)
async def gap_analysis(request: GapAnalysisRequest):
    """
    Analyze a document's approved sections and active draft against strategy-specific
    completeness requirements (e.g., verifying an SRS has measurable NFRs, or a PRD
    defines success KPIs with baselines and targets).

    For every identified gap, returns a natural-language question linked to the relevant
    section plus 2–3 multiple-choice options derived from the project's meeting history
    or industry standards — whichever is more applicable.

    Gaps are ordered by section, matching the document outline. Each gap carries a
    unique gap_id to be used when calling POST /resolve-gap.
    """
    initial_state = {
        "document_id": request.document_id,
        "project_id": "",
        "document_type": "",
        "sections_outline": [],
        "final_document": [],
        "current_section_content": "",
        "current_section_index": 0,
        "document_found": False,
        "gaps": [],
    }

    result = gap_analysis_graph.invoke(initial_state)

    if not result.get("document_found"):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document_id}' not found. Verify the document_id.",
        )

    raw_gaps = result.get("gaps", [])
    gaps = [
        GapItemSchema(
            gap_id=g["gap_id"],
            section_title=g["section_title"],
            section_index=g["section_index"],
            gap_type=g["gap_type"],
            gap_description=g["gap_description"],
            question=g["question"],
            options=[GapOptionSchema(**o) for o in g["options"]],
        )
        for g in raw_gaps
    ]

    return GapAnalysisResponse(
        document_id=request.document_id,
        document_type=result.get("document_type", ""),
        gaps=gaps,
        total_count=len(gaps),
    )


@app.post("/resolve-gap", response_model=ResolveGapResponse)
async def resolve_gap(request: ResolveGapRequest):
    """
    Convert a user's gap resolution into an inline Suggestion.

    The user provides either a selected multiple-choice option or a custom free-text
    response. The system generates a surgical suggestion (original_text + suggested_text
    + rationale) that follows the exact same schema as /suggest-updates — so the frontend
    can feed it into the same suggesting-mode review flow without any special casing.

    Resolving a gap never directly edits the document. The returned Suggestion must be
    explicitly accepted by the user through the editor's suggest/approve UI.
    """
    if not request.selected_option_text and not request.custom_response:
        raise HTTPException(
            status_code=422,
            detail="Provide either selected_option_text or custom_response.",
        )

    # Load the section content from the checkpoint to validate original_text later
    from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import doc_gen_graph
    config = {"configurable": {"thread_id": request.document_id}}
    checkpoint = doc_gen_graph.get_state(config)

    if checkpoint is None or not checkpoint.values:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document_id}' not found.",
        )

    # Find the section content — check approved sections first, then active draft
    doc_state = checkpoint.values
    section_content = ""
    for section in doc_state.get("final_document", []):
        if section.get("title") == request.section_title:
            section_content = section.get("content", "")
            break
    if not section_content:
        section_content = doc_state.get("current_section_content", "")

    if not section_content:
        raise HTTPException(
            status_code=422,
            detail=f"Section '{request.section_title}' has no content to anchor the suggestion to.",
        )

    selected_text = request.selected_option_text or request.custom_response

    raw = resolve_gap_to_suggestion(
        document_id=request.document_id,
        section_title=request.section_title,
        section_index=request.section_index,
        section_content=section_content,
        gap_id=request.gap_id,
        gap_description=request.gap_description,
        question=request.question,
        selected_text=selected_text,
    )

    if raw is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate suggestion from gap resolution. Please try again.",
        )

    return ResolveGapResponse(suggestion=Suggestion(**raw))


# ── Suggestion Accept / Reject ────────────────────────────────────────────────

def _apply_suggestion_to_content(content: str, original_text: str, suggested_text: str, suggestion_type: str) -> str:
    """Apply a suggestion's text change to section content. Returns updated content."""
    if suggestion_type == "update":
        return content.replace(original_text, suggested_text, 1)
    elif suggestion_type == "addition":
        return content.replace(original_text, original_text + "\n\n" + suggested_text, 1)
    elif suggestion_type == "deletion":
        return content.replace(original_text, "", 1)
    return content


@app.post("/accept-suggestion", response_model=AcceptSuggestionResponse)
async def accept_suggestion(request: AcceptSuggestionRequest):
    """
    Apply an accepted suggestion directly to the relevant section in the document checkpoint.

    Finds the section by title in final_document (approved sections) or the active draft,
    applies the text change (update / addition / deletion), and writes it back to SQLite
    via update_state — without resuming or interrupting the generation graph.

    The original_text must be a verbatim substring of the section content (this is guaranteed
    by the suggestion pipeline's validation). If it is not found, a 422 is returned.
    """
    from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import doc_gen_graph

    s = request.suggestion
    config = {"configurable": {"thread_id": request.document_id}}
    checkpoint = doc_gen_graph.get_state(config)

    if checkpoint is None or not checkpoint.values:
        raise HTTPException(status_code=404, detail=f"Document '{request.document_id}' not found.")

    doc_state = checkpoint.values
    final_document = list(doc_state.get("final_document", []))
    current_content = doc_state.get("current_section_content", "")
    sections_outline = doc_state.get("sections_outline", [])
    current_idx = doc_state.get("current_section_index", 0)

    # Try approved sections first
    for i, section in enumerate(final_document):
        if section.get("title") == s.section_title:
            content = section.get("content", "")
            if s.original_text and s.original_text not in content:
                raise HTTPException(
                    status_code=422,
                    detail=f"original_text not found verbatim in section '{s.section_title}'.",
                )
            final_document[i] = {
                "title": section["title"],
                "content": _apply_suggestion_to_content(content, s.original_text, s.suggested_text, s.suggestion_type),
            }
            doc_gen_graph.update_state(config, {"final_document": final_document})
            return AcceptSuggestionResponse(
                document_id=request.document_id,
                section_title=s.section_title,
                applied=True,
                message=f"Suggestion applied to approved section '{s.section_title}'.",
            )

    # Fall back to active draft
    active_title = sections_outline[current_idx] if current_idx < len(sections_outline) else None
    if active_title == s.section_title and current_content:
        if s.original_text and s.original_text not in current_content:
            raise HTTPException(
                status_code=422,
                detail=f"original_text not found verbatim in active draft '{s.section_title}'.",
            )
        updated = _apply_suggestion_to_content(current_content, s.original_text, s.suggested_text, s.suggestion_type)
        doc_gen_graph.update_state(config, {"current_section_content": updated})
        return AcceptSuggestionResponse(
            document_id=request.document_id,
            section_title=s.section_title,
            applied=True,
            message=f"Suggestion applied to active draft '{s.section_title}'.",
        )

    raise HTTPException(
        status_code=404,
        detail=f"Section '{s.section_title}' not found in document '{request.document_id}'.",
    )


@app.post("/reject-suggestion", response_model=RejectSuggestionResponse)
async def reject_suggestion(request: RejectSuggestionRequest):
    """
    Discard a suggestion. No document state is modified.

    The document checkpoint is unchanged — rejection is a client-side decision
    that simply tells the system the suggestion will not be applied.
    """
    return RejectSuggestionResponse(
        document_id=request.document_id,
        suggestion_id=request.suggestion_id,
        message="Suggestion rejected. No changes applied to the document.",
    )


# ── Conflict Resolution ───────────────────────────────────────────────────────

@app.post("/resolve-conflict", response_model=ResolveConflictResponse)
async def resolve_conflict(request: ResolveConflictRequest):
    """
    Convert a user's meeting selection into a ConflictResolution ready to pass
    to POST /generate-document.

    The user picks either meeting_a_id or meeting_b_id from a conflict returned
    by /analyze-project-meetings. This endpoint validates the choice and packages
    it as a ConflictResolutionInput with the correct authoritative_position text.

    The returned resolution should be collected alongside any others and passed
    as conflict_resolutions in the /generate-document request body.
    """
    conflict = request.conflict
    chosen_id = request.authoritative_meeting_id

    if chosen_id == conflict.meeting_a_id:
        position = conflict.meeting_a_position
    elif chosen_id == conflict.meeting_b_id:
        position = conflict.meeting_b_position
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                f"authoritative_meeting_id '{chosen_id}' does not match either "
                f"meeting_a_id ('{conflict.meeting_a_id}') or "
                f"meeting_b_id ('{conflict.meeting_b_id}') in the conflict."
            ),
        )

    return ResolveConflictResponse(
        resolution=ConflictResolutionInput(
            topic=conflict.topic,
            authoritative_meeting_id=chosen_id,
            authoritative_position=position,
        )
    )


# ── Project Analysis ─────────────────────────────────────────────────────────

@app.post("/analyze-project-meetings", response_model=ProjectAnalysisResponse)
async def analyze_project_meetings(request: ProjectAnalysisRequest):
    """
    Ingest multiple meeting transcripts and assess cross-meeting consistency
    before generating a document.

    For each section defined in the target document strategy (SRS / PRD / SDD),
    retrieves the most relevant excerpts from every meeting independently, then
    asks an LLM to identify:

      • Conflicts  — meetings that state contradictory positions on the same topic.
      • Aligned    — topics where all meetings agree, ready for document generation.
      • Thin areas — topics only mentioned in one meeting (insufficient to verify).

    Returns an overall readiness verdict:
      • 'ready'              — no conflicts, sufficient multi-meeting coverage.
      • 'conflicts_detected' — one or more contradictions must be resolved first.
      • 'insufficient_data'  — too few meetings or content to establish consensus.

    At least two meeting sources are required to detect conflicts.
    """
    initial_state = {
        "project_id": request.project_id,
        "document_type": request.document_type,
        "meeting_sources": [
            {"meeting_id": s.meeting_id, "content": s.content}
            for s in request.meeting_sources
        ],
        "conflicts": [],
        "aligned_topics": [],
        "thin_coverage_areas": [],
        "overall_readiness": "",
        "analysis_summary": "",
    }

    result = project_analysis_graph.invoke(initial_state)

    conflicts = [ConflictItemSchema(**c) for c in result.get("conflicts", [])]
    aligned = [AlignedTopicSchema(**a) for a in result.get("aligned_topics", [])]

    return ProjectAnalysisResponse(
        project_id=request.project_id,
        document_type=request.document_type,
        overall_readiness=result.get("overall_readiness", "insufficient_data"),
        analysis_summary=result.get("analysis_summary", ""),
        conflicts=conflicts,
        aligned_topics=aligned,
        thin_coverage_areas=result.get("thin_coverage_areas", []),
        meetings_analyzed=len(request.meeting_sources),
    )


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
