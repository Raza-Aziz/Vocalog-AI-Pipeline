from typing import TypedDict, List, Dict, Any, Optional


class AssistantChunk(TypedDict):
    content: str
    meeting_id: str
    doc_type: str       # "transcript" | "mom"
    chunk_index: int
    speakers: List[str]
    score: float


class DocAssistantState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    document_id: str
    question: str

    # ── Loaded from the doc-gen checkpoint (populated by load_document_state) ─
    project_id: str
    document_type: str
    sections_outline: List[str]
    current_section_index: int
    current_section_content: str
    final_document: List[Dict[str, str]]            # approved sections: [{title, content}]
    refinement_history: Dict[str, List[Dict[str, str]]]  # section_idx → [{draft, feedback}]
    meeting_sources: List[Dict[str, Any]]           # [{meeting_id, content}]
    is_complete: bool
    document_found: bool                            # False when checkpoint lookup fails

    # ── Retrieved knowledge base results ─────────────────────────────────────
    retrieved_chunks: List[AssistantChunk]

    # ── Output ───────────────────────────────────────────────────────────────
    answer: str
    citations: List[Dict[str, Any]]
