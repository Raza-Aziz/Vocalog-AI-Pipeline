from typing import TypedDict, List, Dict, Any


class SuggestionItem(TypedDict):
    suggestion_id: str
    section_title: str
    section_index: int
    original_text: str      # verbatim substring of section content — empty for "addition"
    suggested_text: str     # proposed replacement — empty for "deletion"
    suggestion_type: str    # "update" | "addition" | "deletion"
    rationale: str
    source_meeting_id: str
    confidence: float       # 0.0–1.0


class SuggestionState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    document_id: str
    new_meeting_id: str
    new_meeting_content: str    # raw transcript text of the new meeting

    # ── Loaded from checkpoint ───────────────────────────────────────────────
    project_id: str
    document_type: str
    sections_outline: List[str]
    final_document: List[Dict[str, str]]    # [{title, content}] — approved sections only
    document_found: bool

    # ── Output ───────────────────────────────────────────────────────────────
    suggestions: List[SuggestionItem]
