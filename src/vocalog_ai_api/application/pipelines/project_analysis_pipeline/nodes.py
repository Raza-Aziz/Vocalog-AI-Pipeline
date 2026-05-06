from typing import List, Literal

from pydantic import BaseModel, Field

from vocalog_ai_api.application.pipelines.project_analysis_pipeline.state import (
    ProjectAnalysisState, ConflictItem, AlignedTopic,
)
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import ingest_minutes
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import get_strategy
from vocalog_ai_api.infrastructure.vector_store.qdrant import query_by_meeting
from vocalog_ai_api.infrastructure.llm_providers.groq import llm

# ── Structured output schemas ─────────────────────────────────────────────────

class _ConflictItem(BaseModel):
    topic: str
    conflict_description: str = Field(
        description="Plain-language explanation of the contradiction between the two meetings."
    )
    meeting_a_id: str
    meeting_a_position: str = Field(
        description="What meeting A explicitly states on this topic."
    )
    meeting_b_id: str
    meeting_b_position: str = Field(
        description="What meeting B explicitly states on this topic, contradicting meeting A."
    )
    recommended_resolution: str = Field(
        description="Suggested way to reconcile the conflict before generating the document."
    )


class _AlignedTopic(BaseModel):
    topic: str
    consensus_summary: str = Field(
        description="What all relevant meetings consistently agree on."
    )
    supporting_meetings: List[str] = Field(
        description="Meeting IDs that all corroborate this position."
    )


class _TopicAnalysis(BaseModel):
    conflicts: List[_ConflictItem] = Field(default_factory=list)
    aligned_topics: List[_AlignedTopic] = Field(default_factory=list)
    thin_coverage: bool = Field(
        description="True if this topic is only mentioned in one meeting — insufficient to establish consensus."
    )


class _FinalSummary(BaseModel):
    overall_readiness: Literal["ready", "conflicts_detected", "insufficient_data"] = Field(
        description=(
            "'ready' = no conflicts, sufficient coverage across meetings. "
            "'conflicts_detected' = one or more contradictions need resolution before generating. "
            "'insufficient_data' = too few meetings or too little content to generate a reliable document."
        )
    )
    analysis_summary: str = Field(
        description="One concise paragraph summarising the state of the project knowledge base."
    )


_topic_llm = llm.with_structured_output(_TopicAnalysis)
_summary_llm = llm.with_structured_output(_FinalSummary)

_TOPIC_PROMPT = """\
You are analyzing multiple meeting transcripts to assess alignment on the topic: "{topic}"

The goal is to detect CONFLICTS (meetings that contradict each other) and ALIGNED positions \
(meetings that agree) on this specific topic, to decide if the project knowledge base is \
ready to generate a {doc_type} document.

{meeting_blocks}

TASK:
1. CONFLICTS: Identify any cases where two meetings state clearly contradictory positions \
   on this topic. Only report genuine factual contradictions — not different levels of detail.
2. ALIGNED: Identify positions all relevant meetings consistently share.
3. THIN COVERAGE: Set thin_coverage=true if this topic appears in only one meeting \
   (not enough for cross-verification).

Be specific — reference what each meeting says when reporting a conflict.\
"""

_SUMMARY_PROMPT = """\
You are writing a readiness assessment for generating a {doc_type} document from \
{meeting_count} meeting transcript(s).

Conflicts found ({conflict_count}):
{conflict_summary}

Aligned topics ({aligned_count}):
{aligned_summary}

Thin coverage areas:
{thin_coverage}

Based on the above, determine:
1. overall_readiness: 'ready', 'conflicts_detected', or 'insufficient_data'.
2. analysis_summary: One concise paragraph the user can read before deciding whether to \
   proceed with document generation or resolve conflicts first.\
"""


# ── Node 1: Ingest all meetings into the project knowledge base ────────────────

def ingest_all_meetings(state: ProjectAnalysisState) -> dict:
    """
    Ingests every meeting source into the shared project Qdrant namespace.
    Uses the same idempotent ingest_minutes function as the doc-gen pipeline
    so re-running analysis with updated transcripts safely replaces stale vectors.
    """
    project_id = state["project_id"]
    sources = state["meeting_sources"]

    for source in sources:
        ingest_minutes(
            project_id=project_id,
            meeting_id=source["meeting_id"],
            input_data=source["content"],
        )
        print(f"[ProjectAnalysis] Ingested meeting: {source['meeting_id']}")

    return {}


# ── Node 2: Detect conflicts across all meetings per strategy section ──────────

def detect_conflicts(state: ProjectAnalysisState) -> dict:
    """
    For each section in the document strategy, retrieves the most relevant excerpts
    from every meeting independently (scoped by meeting_id) and asks the LLM to
    identify agreements, contradictions, and thin-coverage areas.

    Strategy sections act as topic probes — this avoids O(n²) meeting-pair comparison
    and instead lets the LLM reason about all perspectives on a topic simultaneously.
    """
    project_id = state["project_id"]
    doc_type = state["document_type"]
    sources = state["meeting_sources"]

    if len(sources) < 2:
        # Single meeting — no conflicts possible, but coverage may be thin
        strategy = get_strategy(doc_type)
        return {
            "conflicts": [],
            "aligned_topics": [],
            "thin_coverage_areas": strategy.sections,
            "overall_readiness": "insufficient_data",
            "analysis_summary": (
                f"Only one meeting was provided. A single source is insufficient to verify "
                f"consistency across the {doc_type.upper()} scope. "
                f"Consider adding more meetings before generating the document."
            ),
        }

    strategy = get_strategy(doc_type)
    meeting_ids = [s["meeting_id"] for s in sources]

    all_conflicts: List[ConflictItem] = []
    all_aligned: List[AlignedTopic] = []
    thin_areas: List[str] = []

    for section_title in strategy.sections:
        # Retrieve top excerpts from EACH meeting independently for this section topic
        meeting_blocks_parts: List[str] = []
        meetings_with_content: List[str] = []

        for meeting_id in meeting_ids:
            chunks = query_by_meeting(
                query_text=section_title,
                meeting_id=meeting_id,
                doc_type="transcript",
                limit=3,
                enable_reranking=False,  # speed — reranking not critical for conflict detection
            )
            if chunks:
                content = "\n".join(c["content"] for c in chunks)
                meeting_blocks_parts.append(f"[Meeting: {meeting_id}]\n{content}")
                meetings_with_content.append(meeting_id)

        # Skip if fewer than 2 meetings have relevant content for this topic
        if len(meetings_with_content) < 2:
            thin_areas.append(section_title)
            continue

        meeting_blocks = "\n\n".join(meeting_blocks_parts)
        prompt = _TOPIC_PROMPT.format(
            topic=section_title,
            doc_type=doc_type.upper(),
            meeting_blocks=meeting_blocks,
        )

        try:
            result: _TopicAnalysis = _topic_llm.invoke(prompt)
        except Exception as e:
            print(f"[ProjectAnalysis] LLM call failed for topic '{section_title}': {e}")
            continue

        if result.thin_coverage:
            thin_areas.append(section_title)

        for c in result.conflicts:
            all_conflicts.append(
                ConflictItem(
                    topic=section_title,
                    conflict_description=c.conflict_description,
                    meeting_a_id=c.meeting_a_id,
                    meeting_a_position=c.meeting_a_position,
                    meeting_b_id=c.meeting_b_id,
                    meeting_b_position=c.meeting_b_position,
                    recommended_resolution=c.recommended_resolution,
                )
            )
        for a in result.aligned_topics:
            all_aligned.append(
                AlignedTopic(
                    topic=a.topic,
                    consensus_summary=a.consensus_summary,
                    supporting_meetings=a.supporting_meetings,
                )
            )

    # ── Generate overall readiness summary ────────────────────────────────────
    conflict_summary = (
        "\n".join(f"- [{c['topic']}] {c['conflict_description']}" for c in all_conflicts)
        or "None"
    )
    aligned_summary = (
        "\n".join(f"- [{a['topic']}] {a['consensus_summary']}" for a in all_aligned)
        or "None"
    )
    thin_summary = "\n".join(f"- {t}" for t in thin_areas) or "None"

    summary_prompt = _SUMMARY_PROMPT.format(
        doc_type=doc_type.upper(),
        meeting_count=len(sources),
        conflict_count=len(all_conflicts),
        conflict_summary=conflict_summary,
        aligned_count=len(all_aligned),
        aligned_summary=aligned_summary,
        thin_coverage=thin_summary,
    )

    try:
        summary: _FinalSummary = _summary_llm.invoke(summary_prompt)
        readiness = summary.overall_readiness
        narrative = summary.analysis_summary
    except Exception as e:
        print(f"[ProjectAnalysis] Summary LLM call failed: {e}")
        readiness = "conflicts_detected" if all_conflicts else "ready"
        narrative = f"Analysis complete: {len(all_conflicts)} conflict(s) found across {len(sources)} meeting(s)."

    print(
        f"[ProjectAnalysis] Done — {len(all_conflicts)} conflicts, "
        f"{len(all_aligned)} aligned topics, {len(thin_areas)} thin areas. "
        f"Readiness: {readiness}"
    )

    return {
        "conflicts": all_conflicts,
        "aligned_topics": all_aligned,
        "thin_coverage_areas": thin_areas,
        "overall_readiness": readiness,
        "analysis_summary": narrative,
    }
