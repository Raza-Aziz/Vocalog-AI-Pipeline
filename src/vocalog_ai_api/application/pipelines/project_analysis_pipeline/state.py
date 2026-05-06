from typing import TypedDict, List, Dict, Any


class ConflictItem(TypedDict):
    topic: str
    conflict_description: str      # plain-language explanation of the contradiction
    meeting_a_id: str
    meeting_a_position: str        # what meeting A says on this topic
    meeting_b_id: str
    meeting_b_position: str        # what meeting B says on this topic
    recommended_resolution: str    # suggested way to reconcile before generating the document


class AlignedTopic(TypedDict):
    topic: str
    consensus_summary: str         # what all relevant meetings agree on
    supporting_meetings: List[str] # which meeting IDs back this up


class ProjectAnalysisState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    project_id: str
    document_type: str                       # "srs" | "prd" | "sdd"
    meeting_sources: List[Dict[str, Any]]    # [{meeting_id, content}]

    # ── Output ───────────────────────────────────────────────────────────────
    conflicts: List[ConflictItem]
    aligned_topics: List[AlignedTopic]
    thin_coverage_areas: List[str]           # section topics with only one meeting mentioning them
    overall_readiness: str                   # "ready" | "conflicts_detected" | "insufficient_data"
    analysis_summary: str                    # one-paragraph human-readable narrative
