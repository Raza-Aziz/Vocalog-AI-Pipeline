from typing import TypedDict, List, Optional, Dict, Literal


class Section(TypedDict):
    title: str
    content: str


class DocumentGenerationState(TypedDict):
    # ── Persistent identity ──────────────────────────────────────────────────
    thread_id: str          # LangGraph thread_id == document_id exposed in API
    project_id: str

    # ── Document strategy ────────────────────────────────────────────────────
    document_type: str      # "srs" | "prd" | "sdd" (or any registered strategy key)

    # ── Source material ──────────────────────────────────────────────────────
    meeting_minutes: Optional[str]

    # ── Document progress ────────────────────────────────────────────────────
    sections_outline: List[str]       # Ordered section headings from the strategy
    current_section_index: int        # Pointer to the section currently in draft

    # ── Current draft ────────────────────────────────────────────────────────
    current_section_content: str      # LLM output for the section being reviewed

    # ── Human-in-the-loop control ────────────────────────────────────────────
    # Set by the API before resuming the graph after an interrupt.
    # "approve"     → save current section, advance to next
    # "refine"      → regenerate with feedback_notes incorporated
    # "regenerate"  → regenerate without specific feedback (fresh attempt)
    pending_action: Optional[Literal["approve", "refine", "regenerate"]]
    feedback_notes: Optional[str]     # User's written feedback (used on "refine")

    # ── Refinement history ───────────────────────────────────────────────────
    # Maps section_index (str) → list of {draft, feedback} for audit trail
    refinement_history: Dict[str, List[Dict[str, str]]]

    # ── Final output ─────────────────────────────────────────────────────────
    final_document: List[Section]     # Approved sections in order
    is_complete: bool
