import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from vocalog_ai_api.api.schemas import TranscriptInput
from vocalog_ai_api.application.pipelines.mom_pipeline.graph import mom_graph
from vocalog_ai_api.api.schemas import (
    DemoDocumentGenerationRequest,
    DemoSectionDraftResponse,
    SectionFeedbackRequest,
    DemoDocumentStatusResponse,
    ActionItemsExtractRequest,
    ActionItemsExtractResponse,
    ActionItemsExecuteRequest,
    ActionItemsForFrontendRequest,
    ActionItemsForFrontendResponse,
)
from vocalog_ai_api.application.pipelines.action_items_pipeline.graph import action_items_graph
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.session_manager import session_manager
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    run_init_step, 
    run_generation_step, 
    run_approval_step
)

app = FastAPI(title="Vocalog AI - Document Generation Module", version="1.0.0")

# @app.post("/generate-mom")
@app.post("/generate-mom", response_class=PlainTextResponse)
def generate_minutes_of_meeting(data: TranscriptInput):
    """
    Generate standardized Minutes of Meeting from a transcript.
    State is persisted to SQLite via the LangGraph checkpointer.
    Provide user_id + meeting_id for a stable thread_id; a UUID is used as fallback.
    """
    user_id = data.user_id or "anonymous"
    session_suffix = data.session_id or data.meeting_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{session_suffix}"}}

    transcript_text = data.transcript.text
    result_state = mom_graph.invoke({"raw_transcript": transcript_text}, config=config)
    return result_state.get("mom_markdown", "")


# @app.post("/generate-document", response_model=DemoSectionDraftResponse)
# async def start_document_generation(request: DemoDocumentGenerationRequest):
#     """
#     Starts the process. Initializes the graph and generates Section 1.
#     """
#     # 1. Create Session
#     doc_id = session_manager.create_session(request.meeting_minutes, request.project_id)
#     session = session_manager.get_session(doc_id)
    
#     # 2. Run Init + First Generation
#     # In a real LangGraph setup with checkpointers, we would stream events.
#     # For this synchronous demo, we call the logic wrapper.
#     new_state = await run_init_step(session)
    
#     # 3. Update Session
#     session_manager.update_session(doc_id, new_state)
    
#     return DemoSectionDraftResponse(
#         document_id=doc_id,
#         section_title=new_state["sections_outline"][new_state["current_section_index"]],
#         content=new_state["current_section_content"],
#         is_complete=False
#     )

@app.post("/generate-document", response_model=DemoSectionDraftResponse)
async def start_document_generation(request: DemoDocumentGenerationRequest):
    """
    Starts the process. 
    1. Creates Session ID.
    2. Ingests minutes into Qdrant (inside run_init_step).
    3. Generates Section 1.
    """
    # 1. Create Session (Allocates the UUID)
    doc_id = session_manager.create_session(request.meeting_minutes, request.project_id)
    session = session_manager.get_session(doc_id)
    
    # 2. Run Init + First Generation
    # note: run_init_step calls 'initialize_document' which now performs Qdrant Ingestion
    print(f"Starting generation for Doc ID: {doc_id}")
    new_state = await run_init_step(session)
    
    # 3. Update Session
    session_manager.update_session(doc_id, new_state)
    
    return DemoSectionDraftResponse(
        document_id=doc_id,
        section_title=new_state["sections_outline"][new_state["current_section_index"]],
        content=new_state["current_section_content"],
        is_complete=False
    )

@app.post("/provide-feedback", response_model=DemoSectionDraftResponse)
async def provide_feedback(request: SectionFeedbackRequest):
    """
    User approves or edits the current section.
    """
    doc_id = request.document_id
    session = session_manager.get_session(doc_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if session["is_complete"]:
        return DemoSectionDraftResponse(
            document_id=doc_id,
            section_title="Document Complete",
            content="All sections approved.",
            is_complete=True
        )

    next_state = session

    # Logic Flow
    if request.action == "approve":
        # 1. Save current section to 'final_document'
        state_after_save = await run_approval_step(session)
        
        # 2. Check if done
        if state_after_save["is_complete"]:
            session_manager.update_session(doc_id, state_after_save)
            return DemoSectionDraftResponse(
                document_id=doc_id,
                section_title="Completed",
                content="Document Generation Finished.",
                is_complete=True,
                refinement_count=0
            )
            
        # 3. Generate NEXT section
        next_state = await run_generation_step(state_after_save)
        
    elif request.action in ["regenerate", "refine"]:
        # 1. Add feedback to state
        session["feedback_notes"] = request.feedback_notes
        
        # 2. Re-run generation on SAME section (HITL Refinement)
        next_state = await run_generation_step(session)
    
    # Update Store
    session_manager.update_session(doc_id, next_state)
    
    # Calculate refinement count for the current section
    history = next_state.get("refinement_history", {})
    current_idx_str = str(next_state["current_section_index"])
    ref_count = len(history.get(current_idx_str, []))
    
    return DemoSectionDraftResponse(
        document_id=doc_id,
        section_title=next_state["sections_outline"][next_state["current_section_index"]],
        content=next_state["current_section_content"],
        is_complete=False,
        refinement_count=ref_count
    )

@app.get("/document-status/{document_id}", response_model=DemoDocumentStatusResponse)
async def get_document_status(document_id: str):
    session = session_manager.get_session(document_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    outline_len = len(session.get("sections_outline", []))
    
    status_str = "completed" if session["is_complete"] else "in_progress"
    
    current_title = None
    if outline_len > 0 and not session["is_complete"]:
        current_title = session["sections_outline"][session["current_section_index"]]

    return DemoDocumentStatusResponse(
        document_id=document_id,
        status=status_str,
        current_section_title=current_title,
        completed_sections=session["current_section_index"],
        total_sections=outline_len
    )

# --- Action Items Pipeline Endpoints ---

@app.post("/action-items/extract", response_model=ActionItemsExtractResponse)
async def extract_action_items(request: ActionItemsExtractRequest):
    """
    Extracts action items (Assignee, Task, Deadline) from a given transcript.
    Runs the independent Action Items LangGraph pipeline.
    State is persisted to SQLite; a fresh UUID thread_id is used per call.
    """
    user_id = request.user_id or "anonymous"
    session_suffix = request.session_id or request.meeting_id or str(uuid.uuid4())
    thread_id = f"{user_id}:{session_suffix}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"transcript": request.transcript, "session_id": session_suffix}
    result_state = action_items_graph.invoke(initial_state, config=config)
    return ActionItemsExtractResponse(actions=result_state.get("extracted_actions", []))


@app.post("/action-items/extract-for-frontend", response_model=ActionItemsForFrontendResponse)
async def extract_action_items_for_frontend(request: ActionItemsForFrontendRequest):
    """
    Extracts action items for direct frontend review and manual persistence.
    No automatic Slack routing or side-effects — the caller decides what to do with the results.
    Accepts optional session_id, meeting_id, and user_id for multi-session tracking.
    """
    user_id = request.user_id or "anonymous"
    session_id = request.session_id or request.meeting_id or str(uuid.uuid4())
    thread_id = f"{user_id}:{session_id}"
    config = {"configurable": {"thread_id": thread_id}}
    # Extract the plain-text field from the TranscriptData object
    transcript_text = request.transcript.text
    initial_state = {"transcript": transcript_text, "session_id": session_id}
    result_state = action_items_graph.invoke(initial_state, config=config)
    actions = result_state.get("extracted_actions", [])
    return ActionItemsForFrontendResponse(
        session_id=session_id,
        meeting_id=request.meeting_id,
        user_id=request.user_id,
        actions=actions,
        total_count=len(actions),
    )

from vocalog_ai_api.application.services.action_executor import execute_slack_actions as run_slack_executor

@app.post("/action-items/execute/slack")
async def execute_slack_actions(request: ActionItemsExecuteRequest):
    """
    Executes the given action items by sending them to Slack via MCP.
    Connects to the @zencoderai/slack-mcp-server process.
    """
    
    result = await run_slack_executor(actions=request.actions, channel_id=request.channel_id)
    
    return {
        "message": f"Processed {len(request.actions)} actions.", 
        "execution_result": result
    }

@app.get("/health")
def health():
    return {"status": "ok"}
