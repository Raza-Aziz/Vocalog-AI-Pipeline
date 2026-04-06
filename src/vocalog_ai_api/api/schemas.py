from pydantic import BaseModel
from typing import Optional, Literal


class TranscriptInput(BaseModel):
    raw_transcript: dict


class MoMResponse(BaseModel):
    markdown: str


# # Document Generation Schemas
# class DocumentGenerationRequest(BaseModel):
#     """Request body for starting document generation."""
#     document_type: Literal["SRS"] = "SRS"
#     initial_context: Optional[str] = None


# class SectionDraftResponse(BaseModel):
#     """Response with section draft for review."""
#     document_id: str
#     section_id: str
#     section_title: str
#     draft_content: str
#     status: str
#     current_section_index: int
#     total_sections: int


# class SectionFeedbackRequest(BaseModel):
#     """User feedback payload for a section."""
#     action: Literal["approve", "edit", "refine"]
#     edited_content: Optional[str] = None
#     feedback_text: Optional[str] = None


# class DocumentStatusResponse(BaseModel):
#     """Current document generation status."""
#     document_id: str
#     project_id: str
#     status: str  # "generating", "waiting_feedback", "completed"
#     current_section_id: Optional[str] = None
#     current_section_title: Optional[str] = None
#     draft_content: Optional[str] = None
#     current_section_index: int
#     total_sections: int
#     completed_sections: int


# class DocumentResponse(BaseModel):
#     """Final completed document response."""
#     document_id: str
#     project_id: str
#     document_type: str
#     content: str
#     status: str


from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from vocalog_ai_api.application.pipelines.action_items_pipeline.schema import ActionItem

# --- Request Models ---

class DemoDocumentGenerationRequest(BaseModel):
    meeting_minutes: str = Field(..., description="Raw meeting minutes text")
    # ADD THIS LINE BELOW:
    project_id: str = Field(default="demo-project", description="Project Identifier")

class SectionFeedbackRequest(BaseModel):
    document_id: str
    action: Literal["approve", "regenerate", "refine"]
    feedback_notes: Optional[str] = None

# --- Response Models ---

class DemoDocumentStatusResponse(BaseModel):
    document_id: str
    status: Literal["in_progress", "completed", "error"]
    current_section_title: Optional[str] = None
    completed_sections: int
    total_sections: int

class ActionItemsExtractRequest(BaseModel):
    transcript: str = Field(..., description="Transcript text to extract actions from.")

class ActionItemsExtractResponse(BaseModel):
    actions: List[ActionItem]

class ActionItemsExecuteRequest(BaseModel):
    actions: List[ActionItem] = Field(..., description="List of action items to execute via MCP.")
    channel_id: str = Field(default="general", description="The Slack channel ID to send messages to.")

class DemoSectionDraftResponse(BaseModel):
    document_id: str
    section_title: str
    content: str
    is_complete: bool = False
    refinement_count: int = 0
    message: str = "Review the section draft."