from typing import TypedDict, List, Dict, Any


class GapOption(TypedDict):
    option_id: str              # "A", "B", "C"
    text: str                   # the actual content of this option
    source: str                 # "meeting_context" | "industry_standard"


class GapItem(TypedDict):
    gap_id: str
    section_title: str
    section_index: int
    gap_type: str               # "missing_content" | "incomplete_requirement" | "undefined_metric" | "missing_persona" | "missing_rationale"
    gap_description: str        # e.g. "Non-Functional Requirements lacks measurable performance targets"
    question: str               # the natural-language question posed to the user
    options: List[GapOption]    # 2–3 multiple-choice options


class GapAnalysisState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    document_id: str

    # ── Loaded from checkpoint ───────────────────────────────────────────────
    project_id: str
    document_type: str
    sections_outline: List[str]
    final_document: List[Dict[str, str]]
    current_section_content: str
    current_section_index: int
    document_found: bool

    # ── Output ───────────────────────────────────────────────────────────────
    gaps: List[GapItem]
