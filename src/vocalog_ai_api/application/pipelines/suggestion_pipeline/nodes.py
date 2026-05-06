import json
import re
import uuid
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from vocalog_ai_api.application.pipelines.suggestion_pipeline.state import SuggestionState, SuggestionItem
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import ingest_minutes
from vocalog_ai_api.infrastructure.vector_store.qdrant import query_by_meeting
from vocalog_ai_api.infrastructure.llm_providers.groq import llm


# ── Pydantic models for validation only (not passed to with_structured_output) ─

class _LLMSuggestion(BaseModel):
    original_text: str
    suggested_text: str
    suggestion_type: Literal["update", "addition", "deletion"]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class _SectionAnalysis(BaseModel):
    suggestions: List[_LLMSuggestion] = Field(default_factory=list)


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences gracefully."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── Prompts ───────────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = (
    "You are a precise document synchronization assistant. "
    'You MUST respond with ONLY a valid JSON object with a single key "suggestions" '
    "containing a list of suggestion objects. Each suggestion object must have: "
    "original_text (string), suggested_text (string), "
    "suggestion_type ('update'|'addition'|'deletion'), rationale (string), confidence (float 0-1). "
    'If nothing needs changing return {"suggestions": []}. No markdown, no explanation — JSON only.'
)

_ANALYSIS_HUMAN = """\
Analyze this document section against the new meeting discussion and return suggestions.

EXISTING DOCUMENT SECTION: "{section_title}"
───────────────────────────────────────────
{section_content}
───────────────────────────────────────────

NEW MEETING DISCUSSION:
───────────────────────────────────────────
{new_meeting_context}
───────────────────────────────────────────

RULES:
1. original_text MUST be a verbatim, character-for-character copy from the EXISTING SECTION.
2. Only suggest changes directly supported by the new meeting discussion.
3. No stylistic or formatting changes — factual/substantive updates only.
4. For 'addition': original_text is the last sentence of the paragraph to insert after.
5. Confidence: 0.9+ = clear contradiction or new fact; 0.7–0.9 = strong implication; 0.5–0.7 = possible. Omit below 0.5.\
"""


# ── Node 1: Load document state ────────────────────────────────────────────────

def load_document_state(state: SuggestionState) -> dict:
    from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import doc_gen_graph

    config = {"configurable": {"thread_id": state["document_id"]}}
    checkpoint = doc_gen_graph.get_state(config)

    if checkpoint is None or not checkpoint.values:
        return {
            "document_found": False,
            "project_id": "",
            "document_type": "",
            "sections_outline": [],
            "final_document": [],
        }

    doc = checkpoint.values
    return {
        "document_found": True,
        "project_id": doc.get("project_id", ""),
        "document_type": doc.get("document_type", ""),
        "sections_outline": doc.get("sections_outline", []),
        "final_document": list(doc.get("final_document", [])),
    }


# ── Node 2: Ingest new meeting into the project knowledge base ─────────────────

def ingest_new_meeting(state: SuggestionState) -> dict:
    """
    Ingests the new meeting transcript into the project-level Qdrant knowledge base
    so future document generation and retrieval can use it, then also ingests it
    under the Q&A namespace so /meeting-qa works for this meeting too.
    """
    if not state.get("document_found") or not state.get("project_id"):
        return {}

    project_id = state["project_id"]
    meeting_id = state["new_meeting_id"]
    content = state["new_meeting_content"]

    # Project-level ingestion (for doc-gen RAG)
    ingest_minutes(project_id=project_id, meeting_id=meeting_id, input_data=content)

    # Q&A-scoped ingestion (for /meeting-qa)
    from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import (
        ingest_transcript_for_qa,
    )
    ingest_transcript_for_qa(meeting_id=meeting_id, transcript_text=content)

    return {}


# ── Node 3: Analyze each approved section against the new meeting ──────────────

def analyze_sections(state: SuggestionState) -> dict:
    """
    For every approved section, retrieves the most relevant excerpts from the new
    meeting transcript (scoped by meeting_id) and asks the LLM — using structured
    output — to produce surgical inline suggestions.

    Only sections with at least one suggestion above the 0.5 confidence threshold
    produce output.  Skips sections with no approved content.
    """
    if not state.get("document_found") or not state.get("project_id"):
        return {"suggestions": []}

    final_document = state.get("final_document", [])
    new_meeting_id = state["new_meeting_id"]
    sections_outline = state.get("sections_outline", [])
    all_suggestions: List[SuggestionItem] = []

    for section in final_document:
        section_title = section.get("title", "")
        section_content = section.get("content", "")

        if not section_content.strip():
            continue

        # Determine the section index for citation purposes
        try:
            section_index = sections_outline.index(section_title)
        except ValueError:
            section_index = -1

        # Retrieve the top relevant chunks from the NEW meeting for this section
        new_meeting_chunks = query_by_meeting(
            query_text=section_title,
            meeting_id=new_meeting_id,
            doc_type="transcript",
            limit=4,
            enable_reranking=True,
        )
        if not new_meeting_chunks:
            continue  # new meeting has nothing relevant for this section

        new_meeting_context = "\n\n".join(
            f"[Excerpt {i+1}]{' [Speakers: ' + ', '.join(r['metadata'].get('speakers', [])) + ']' if r['metadata'].get('speakers') else ''}\n{r['content']}"
            for i, r in enumerate(new_meeting_chunks)
        )

        messages = [
            SystemMessage(content=_ANALYSIS_SYSTEM),
            HumanMessage(content=_ANALYSIS_HUMAN.format(
                section_title=section_title,
                section_content=section_content[:2500],
                new_meeting_context=new_meeting_context[:800],
            )),
        ]

        try:
            response = llm.invoke(messages)
            parsed = _extract_json(response.content)
            result = _SectionAnalysis.model_validate(parsed)
        except Exception as e:
            print(f"[SuggestionPipeline] LLM call failed for section '{section_title}': {e}")
            continue

        for llm_sug in result.suggestions:
            if llm_sug.confidence < 0.5:
                continue

            # Validate that original_text is actually in the section (for "update"/"deletion")
            if llm_sug.suggestion_type in ("update", "deletion"):
                if llm_sug.original_text and llm_sug.original_text not in section_content:
                    print(
                        f"[SuggestionPipeline] Skipping suggestion — original_text not found verbatim in section '{section_title}'"
                    )
                    continue

            all_suggestions.append(
                SuggestionItem(
                    suggestion_id=str(uuid.uuid4()),
                    section_title=section_title,
                    section_index=section_index,
                    original_text=llm_sug.original_text,
                    suggested_text=llm_sug.suggested_text,
                    suggestion_type=llm_sug.suggestion_type,
                    rationale=llm_sug.rationale,
                    source_meeting_id=new_meeting_id,
                    confidence=llm_sug.confidence,
                )
            )

    # Sort by confidence descending so highest-priority changes surface first
    all_suggestions.sort(key=lambda s: s["confidence"], reverse=True)
    print(f"[SuggestionPipeline] Generated {len(all_suggestions)} suggestions from meeting '{new_meeting_id}'")
    return {"suggestions": all_suggestions}
