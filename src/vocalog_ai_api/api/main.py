from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src.vocalog_ai_api.api.schemas import MoMResponse, TranscriptInput
from src.vocalog_ai_api.application.pipelines.mom_pipeline.graph import mom_graph
from src.vocalog_ai_api.api.schemas import (
    DemoDocumentGenerationRequest, 
    DemoSectionDraftResponse, 
    SectionFeedbackRequest,
    DemoDocumentStatusResponse
)
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.session_manager import session_manager
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    run_init_step, 
    run_generation_step, 
    run_approval_step
)

app = FastAPI(title="Vocalog AI - Document Generation Module", version="1.0.0")


# @app.post("/generate-mom")
@app.post(
    "/generate-mom",response_class=PlainTextResponse)
def generate_minutes_of_meeting(data: TranscriptInput):
    """
    Generate standardized Minutes of Meeting from a transcript.
    """
    # 1. Run LangGraph pipeline
    result_state = mom_graph.invoke({"raw_transcript": data.raw_transcript})

    # 2. Extract Markdown output from state
    markdown = result_state.get("mom_markdown", "")

    # 3. Return Markdown response
    return markdown


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
                is_complete=True
            )
            
        # 3. Generate NEXT section
        next_state = await run_generation_step(state_after_save)
        
    elif request.action == "regenerate":
        # 1. Add feedback to state
        session["feedback_notes"] = request.feedback_notes
        session["feedback_action"] = "regenerate"
        
        # 2. Re-run generation on SAME section
        next_state = await run_generation_step(session)
    
    # Update Store
    session_manager.update_session(doc_id, next_state)
    
    return DemoSectionDraftResponse(
        document_id=doc_id,
        section_title=next_state["sections_outline"][next_state["current_section_index"]],
        content=next_state["current_section_content"],
        is_complete=False
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

