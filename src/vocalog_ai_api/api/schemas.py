from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from vocalog_ai_api.application.pipelines.action_items_pipeline.schema import ActionItem


# ── Transcript format (matches Vocalog transcription output) ─────────────────

class TranscriptWord(BaseModel):
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    type: Optional[str] = None
    speaker_id: Optional[str] = None


class TranscriptData(BaseModel):
    text: str = Field(..., description="Full transcript text as a single string.")
    language_code: Optional[str] = None
    language_probability: Optional[float] = None
    words: Optional[List[TranscriptWord]] = None


# ── MoM endpoint schemas ─────────────────────────────────────────────────────

class TranscriptInput(BaseModel):
    transcript: TranscriptData = Field(
        ..., description="Transcript object from the Vocalog transcription pipeline."
    )
    user_id: Optional[str] = Field(None, description="User ID — used as LangGraph thread_id prefix.")
    session_id: Optional[str] = Field(None, description="Session ID — preferred thread_id suffix.")
    meeting_id: Optional[str] = Field(None, description="Meeting ID — fallback if session_id is absent.")


class MoMAndActionsResponse(BaseModel):
    meeting_minutes: str = Field(..., description="Generated Minutes of Meeting in Markdown.")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items.")
    total_count: int = Field(..., description="Number of extracted action items.")


# ── Document Generation schemas ──────────────────────────────────────────────

class DemoDocumentGenerationRequest(BaseModel):
    meeting_minutes: str = Field(..., description="Raw meeting minutes text.")
    project_id: str = Field(default="demo-project", description="Project identifier.")
    document_type: Literal["srs", "prd", "sdd"] = Field(
        default="srs",
        description=(
            "Document type to generate. "
            "'srs' = Software Requirements Specification, "
            "'prd' = Product Requirements Document, "
            "'sdd' = Software Design Document."
        ),
    )


class SectionFeedbackRequest(BaseModel):
    document_id: str = Field(
        ..., description="Thread ID returned by /generate-document (permanent session key)."
    )
    action: Literal["approve", "regenerate", "refine"]
    feedback_notes: Optional[str] = Field(
        None, description="Required when action='refine'. Ignored otherwise."
    )


class DemoDocumentStatusResponse(BaseModel):
    document_id: str
    document_type: str
    status: Literal["in_progress", "completed", "error"]
    current_section_title: Optional[str] = None
    completed_sections: int
    total_sections: int


class DemoSectionDraftResponse(BaseModel):
    document_id: str
    document_type: str
    section_title: str
    content: str
    is_complete: bool = False
    refinement_count: int = 0
    message: str = "Review the section draft."


# ── Action Items schemas ─────────────────────────────────────────────────────

class ActionItemsForFrontendRequest(BaseModel):
    transcript: TranscriptData = Field(
        ..., description="Transcript object from the Vocalog transcription pipeline."
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID linking to a meeting or user session. Auto-generated UUID if omitted.",
    )
    meeting_id: Optional[str] = Field(None, description="Optional meeting ID — fallback thread_id suffix.")
    user_id: Optional[str] = Field(None, description="Optional user ID for context.")


class ActionItemsForFrontendResponse(BaseModel):
    actions: List[ActionItem]
    total_count: int = Field(description="Total number of extracted action items.")

