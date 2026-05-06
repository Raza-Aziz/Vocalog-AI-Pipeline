import uuid
from typing import List, Literal

from pydantic import BaseModel, Field

from vocalog_ai_api.application.pipelines.suggestion_pipeline.state import SuggestionState, SuggestionItem
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import ingest_minutes
from vocalog_ai_api.infrastructure.vector_store.qdrant import query_by_meeting
from vocalog_ai_api.infrastructure.llm_providers.groq import llm

# ── Structured output schemas (Pydantic, used only with_structured_output) ────

class _LLMSuggestion(BaseModel):
    original_text: str = Field(
        description=(
            "EXACT verbatim substring of the existing section content to be modified. "
            "Must be copy-pasted character-for-character from the section text. "
            "For 'addition' type, use the last sentence of the relevant paragraph as an anchor."
        )
    )
    suggested_text: str = Field(
        description=(
            "The proposed replacement or new text. "
            "Empty string for 'deletion' type."
        )
    )
    suggestion_type: Literal["update", "addition", "deletion"] = Field(
        description=(
            "'update' = replace original_text with suggested_text. "
            "'addition' = insert suggested_text after the anchor in original_text. "
            "'deletion' = remove original_text entirely."
        )
    )
    rationale: str = Field(
        description="One or two sentences explaining why this change is necessary based on the new meeting."
    )
    confidence: float = Field(
        description="0.9–1.0: clearly required. 0.7–0.9: likely needed. 0.5–0.7: possibly needed.",
        ge=0.0,
        le=1.0,
    )


class _SectionAnalysis(BaseModel):
    suggestions: List[_LLMSuggestion] = Field(
        default_factory=list,
        description="List of surgical suggestions. Empty list if nothing needs changing.",
    )


_structured_llm = llm.with_structured_output(_SectionAnalysis)

_ANALYSIS_PROMPT = """\
You are a precise document synchronization assistant. Your job is to identify whether \
new meeting discussion makes any part of an existing document section stale, contradicted, \
or incomplete — and to generate surgical inline suggestions.

EXISTING DOCUMENT SECTION: "{section_title}"
───────────────────────────────────────────
{section_content}
───────────────────────────────────────────

NEW MEETING DISCUSSION (excerpts most relevant to this section):
───────────────────────────────────────────
{new_meeting_context}
───────────────────────────────────────────

RULES — read carefully before generating suggestions:
1. original_text MUST be a verbatim, character-for-character copy of text from the \
   EXISTING SECTION above. Never paraphrase or rewrite it.
2. Only suggest changes that are directly, clearly supported by the new meeting discussion.
3. Do NOT suggest stylistic, formatting, or structural changes — only factual/substantive updates.
4. For 'addition': original_text is the last sentence of the paragraph after which to insert.
5. If nothing in this section needs changing, return an empty suggestions list.
6. Confidence guide: 0.9+ = unambiguous contradiction or new fact; 0.7–0.9 = strong implication; \
   0.5–0.7 = possible relevance. Do not include suggestions below 0.5.\
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

        prompt = _ANALYSIS_PROMPT.format(
            section_title=section_title,
            section_content=section_content,
            new_meeting_context=new_meeting_context,
        )

        try:
            result: _SectionAnalysis = _structured_llm.invoke(prompt)
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
