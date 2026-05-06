from typing import List, Optional, Literal, Union
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

class MeetingSourceInput(BaseModel):
    meeting_id: str = Field(..., description="Unique identifier for this meeting.")
    content: Union[str, dict] = Field(
        ...,
        description="Raw transcript text or structured transcript object with 'text' and 'words' fields.",
    )


class DemoDocumentGenerationRequest(BaseModel):
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
    sources: List[MeetingSourceInput] = Field(
        ...,
        description="One or more meeting transcripts to synthesise the document from.",
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


# ── Meeting Q&A schemas ──────────────────────────────────────────────────────

class MeetingQARequest(BaseModel):
    meeting_id: str = Field(
        ..., description="Unique identifier of the meeting to query. Must match the ID used during ingestion."
    )
    question: str = Field(..., description="Natural-language question about the meeting.")


class CitedSource(BaseModel):
    label: str = Field(..., description="Citation label used in the answer, e.g. 'MOM-1' or 'TRANSCRIPT-2'.")
    source: Literal["minutes", "transcript"]
    doc_type: str
    chunk_index: int
    speakers: List[str] = Field(default_factory=list)
    excerpt: str = Field(..., description="First 250 characters of the source chunk.")


class MeetingQAResponse(BaseModel):
    meeting_id: str
    question: str
    answer: str = Field(..., description="Grounded answer citing relevant discussion points and formal conclusions.")
    citations: List[CitedSource] = Field(
        default_factory=list,
        description="Source chunks the answer draws from, labelled MOM-N or TRANSCRIPT-N.",
    )


# ── Document Assistant schemas ───────────────────────────────────────────────

class DocAssistantRequest(BaseModel):
    document_id: str = Field(
        ...,
        description=(
            "The document_id (= LangGraph thread_id) returned by POST /generate-document. "
            "Used to look up the live document state from the SQLite checkpoint."
        ),
    )
    question: str = Field(
        ...,
        description=(
            "Natural-language question about the document or the meetings behind it. "
            "Examples: 'Why was this architecture chosen?', "
            "'Which meeting discussed the auth requirements?', "
            "'What feedback caused Section 2 to be rewritten?'"
        ),
    )


class DocAssistantCitation(BaseModel):
    label: str = Field(..., description="Citation label used inline in the answer, e.g. 'SECTION-1', 'MEETING-3', 'DRAFT'.")
    source: Literal["document", "meeting"]
    doc_type: str = Field(..., description="'approved_section', 'active_draft', 'transcript', or 'mom'.")
    section_title: Optional[str] = None
    meeting_id: Optional[str] = None
    chunk_index: int
    speakers: List[str] = Field(default_factory=list)
    excerpt: str = Field(..., description="First 200 characters of the cited content.")


class DocAssistantDocumentSnapshot(BaseModel):
    document_type: str
    status: Literal["in_progress", "completed", "not_found"]
    approved_sections: int
    total_sections: int
    active_section: Optional[str] = None
    source_meetings: List[str] = Field(default_factory=list)


class DocAssistantResponse(BaseModel):
    document_id: str
    question: str
    answer: str = Field(
        ...,
        description="Grounded, citation-rich answer in markdown, cross-referencing document state and meeting evidence.",
    )
    citations: List[DocAssistantCitation] = Field(
        default_factory=list,
        description="All sources the answer draws from, labelled SECTION-N, DRAFT, or MEETING-N.",
    )
    document_snapshot: DocAssistantDocumentSnapshot = Field(
        ..., description="Live snapshot of the document's progress at query time."
    )


# ── Suggestion & Synchronization schemas ────────────────────────────────────

class Suggestion(BaseModel):
    suggestion_id: str = Field(..., description="UUID — stable identifier for accept/reject tracking.")
    section_title: str
    section_index: int
    original_text: str = Field(
        ...,
        description=(
            "Verbatim substring of the section content. "
            "For 'addition': acts as the anchor sentence after which new text is inserted. "
            "For 'deletion': the text to be removed. Empty for pure-append additions."
        ),
    )
    suggested_text: str = Field(
        ...,
        description="The proposed replacement or new text. Empty for 'deletion' type.",
    )
    suggestion_type: Literal["update", "addition", "deletion"]
    rationale: str = Field(..., description="Why this change is necessary.")
    source_meeting_id: Optional[str] = Field(None, description="Meeting that triggered this suggestion.")
    source_gap_id: Optional[str] = Field(None, description="Gap ID if triggered by gap resolution.")
    confidence: float = Field(..., description="0.0–1.0. Higher = more certain the change is needed.")


class SuggestUpdatesRequest(BaseModel):
    document_id: str = Field(..., description="document_id returned by POST /generate-document.")
    new_meeting_id: str = Field(..., description="Unique identifier for the new meeting being added.")
    new_meeting_content: str = Field(..., description="Raw transcript text of the new meeting.")


class SuggestUpdatesResponse(BaseModel):
    document_id: str
    new_meeting_id: str
    suggestions: List[Suggestion]
    total_count: int


# ── Gap Analysis schemas ──────────────────────────────────────────────────────

class GapOptionSchema(BaseModel):
    option_id: str = Field(..., description="'A', 'B', or 'C'.")
    text: str = Field(..., description="Concrete option text the user can select.")
    source: Literal["meeting_context", "industry_standard"]


class GapItemSchema(BaseModel):
    gap_id: str = Field(..., description="UUID — used when calling POST /resolve-gap.")
    section_title: str
    section_index: int
    gap_type: Literal[
        "missing_content",
        "incomplete_requirement",
        "undefined_metric",
        "missing_persona",
        "missing_rationale",
    ]
    gap_description: str = Field(..., description="One sentence describing the specific gap.")
    question: str = Field(..., description="Natural-language question posed to the user.")
    options: List[GapOptionSchema] = Field(..., description="2–3 selectable options.")


class GapAnalysisRequest(BaseModel):
    document_id: str = Field(..., description="document_id returned by POST /generate-document.")


class GapAnalysisResponse(BaseModel):
    document_id: str
    document_type: str
    gaps: List[GapItemSchema]
    total_count: int


class ResolveGapRequest(BaseModel):
    document_id: str = Field(..., description="document_id of the document being edited.")
    gap_id: str = Field(..., description="gap_id from the GapAnalysisResponse.")
    section_title: str
    section_index: int
    question: str = Field(..., description="The gap question (echo from GapAnalysisResponse).")
    gap_description: str = Field(..., description="The gap description (echo from GapAnalysisResponse).")
    selected_option_text: Optional[str] = Field(
        None,
        description="Text of the selected multiple-choice option. Provide this OR custom_response.",
    )
    custom_response: Optional[str] = Field(
        None,
        description="User's free-text answer. Provide this OR selected_option_text.",
    )


class ResolveGapResponse(BaseModel):
    suggestion: Suggestion = Field(
        ...,
        description=(
            "Inline suggestion generated from the gap resolution. "
            "Pass this to the editor's suggesting-mode pipeline exactly as the sync suggestions."
        ),
    )


# ── Project Analysis schemas ─────────────────────────────────────────────────

class ProjectAnalysisRequest(BaseModel):
    project_id: str = Field(..., description="Project identifier shared across all meetings.")
    document_type: Literal["srs", "prd", "sdd"] = Field(
        ..., description="Document type to assess readiness for."
    )
    meeting_sources: List[MeetingSourceInput] = Field(
        ..., description="Two or more meeting transcripts to analyze for conflicts."
    )


class ConflictItemSchema(BaseModel):
    topic: str
    conflict_description: str
    meeting_a_id: str
    meeting_a_position: str
    meeting_b_id: str
    meeting_b_position: str
    recommended_resolution: str


class AlignedTopicSchema(BaseModel):
    topic: str
    consensus_summary: str
    supporting_meetings: List[str]


class ProjectAnalysisResponse(BaseModel):
    project_id: str
    document_type: str
    overall_readiness: Literal["ready", "conflicts_detected", "insufficient_data"]
    analysis_summary: str
    conflicts: List[ConflictItemSchema] = Field(default_factory=list)
    aligned_topics: List[AlignedTopicSchema] = Field(default_factory=list)
    thin_coverage_areas: List[str] = Field(default_factory=list)
    meetings_analyzed: int


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

