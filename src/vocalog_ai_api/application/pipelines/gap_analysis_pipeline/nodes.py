import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from vocalog_ai_api.application.pipelines.gap_analysis_pipeline.state import (
    GapAnalysisState, GapItem, GapOption,
)
from vocalog_ai_api.application.pipelines.gap_analysis_pipeline.requirements import get_section_requirements
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.hybrid_retriever import HybridRetriever
from langchain_core.messages import SystemMessage, HumanMessage
from vocalog_ai_api.infrastructure.llm_providers.groq import llm

# ── Structured output schemas ─────────────────────────────────────────────────

class _GapOption(BaseModel):
    option_id: str = Field(description="Single uppercase letter: 'A', 'B', or 'C'.")
    text: str = Field(description="The concrete option text the user can select.")
    source: Literal["meeting_context", "industry_standard"] = Field(
        description=(
            "'meeting_context' if derived from the provided meeting excerpts. "
            "'industry_standard' if derived from general best practice for this document type."
        )
    )


class _GapItem(BaseModel):
    gap_type: Literal[
        "missing_content",
        "incomplete_requirement",
        "undefined_metric",
        "missing_persona",
        "missing_rationale",
    ]
    gap_description: str = Field(
        description="One sentence describing what is missing or under-specified."
    )
    question: str = Field(
        description="A direct natural-language question to ask the user to resolve this gap."
    )
    options: List[_GapOption] = Field(
        description="Exactly 2 options the user can choose from.",
        min_length=2,
        max_length=2,
    )


class _SectionGapAnalysis(BaseModel):
    gaps: List[_GapItem] = Field(
        default_factory=list,
        description="All gaps found in this section. Empty list if section is complete.",
    )


_structured_llm = llm.with_structured_output(_SectionGapAnalysis, method="json_mode")

_GAP_SYSTEM = (
    "You are a document quality analyst. "
    "You MUST respond with ONLY a valid JSON object — no markdown, no explanation, no table. "
    'The JSON must have exactly one key "gaps" whose value is a list of gap objects. '
    "Each gap object must have: gap_type (string), gap_description (string), "
    "question (string), options (list of exactly 2 objects each with option_id, text, source). "
    'If there are no gaps return {"gaps": []}.'
)

_GAP_HUMAN = """\
Audit this {doc_type} section for completeness.

SECTION: "{section_title}"
───────────────────────────────────────────
{section_content}
───────────────────────────────────────────

COMPLETENESS REQUIREMENTS:
{requirements_list}

MEETING CONTEXT (use to derive grounded options):
───────────────────────────────────────────
{meeting_context}
───────────────────────────────────────────

Report at most 5 genuine gaps (criteria that are absent, vague, or unmeasurable).
For each gap provide exactly 2 options — prefer meeting_context sources, fall back to industry_standard.
Each option must be a concrete usable value, not a placeholder.\
"""

_RESOLVE_PROMPT = """\
You are a precise document editor. The user has answered a gap question and you must \
generate an inline suggestion to resolve it within the existing document section.

SECTION: "{section_title}"
───────────────────────────────────────────
{section_content}
───────────────────────────────────────────

GAP THAT WAS IDENTIFIED:
{gap_description}

QUESTION ASKED:
{question}

USER'S ANSWER:
{selected_text}

TASK:
Generate a surgical inline suggestion to incorporate the user's answer into the section.

Rules:
1. original_text MUST be a verbatim substring of the EXISTING SECTION content that \
   identifies where the change should happen. Use the most relevant surrounding sentence.
   For a pure addition (new content with no replacement), use the last sentence of the \
   most relevant paragraph as an anchor.
2. suggested_text must incorporate the user's answer naturally into the document's writing style.
3. suggestion_type: 'update' to replace text, 'addition' to insert after anchor.
4. rationale: one sentence explaining how this resolves the gap.\
"""


class _ResolveSuggestion(BaseModel):
    original_text: str = Field(
        description="Exact verbatim substring from the section content. Used as the edit anchor."
    )
    suggested_text: str = Field(
        description="The new text to insert or replace with."
    )
    suggestion_type: Literal["update", "addition"] = Field(
        description="'update' = replace original_text. 'addition' = insert after original_text."
    )
    rationale: str


_resolve_llm = llm.with_structured_output(_ResolveSuggestion)


# ── Node 1: Load document state ────────────────────────────────────────────────

def load_document_state(state: GapAnalysisState) -> dict:
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
            "current_section_content": "",
            "current_section_index": 0,
        }

    doc = checkpoint.values
    return {
        "document_found": True,
        "project_id": doc.get("project_id", ""),
        "document_type": doc.get("document_type", ""),
        "sections_outline": doc.get("sections_outline", []),
        "final_document": list(doc.get("final_document", [])),
        "current_section_content": doc.get("current_section_content", ""),
        "current_section_index": doc.get("current_section_index", 0),
    }


# ── Node 2: Analyze each section for gaps ────────────────────────────────────

def analyze_gaps(state: GapAnalysisState) -> dict:
    """
    For every approved section (and the active draft if present), compares content
    against the strategy's per-section completeness requirements. For each gap,
    retrieves relevant meeting context to ground the suggested options in real
    project data rather than generic placeholders.
    """
    if not state.get("document_found") or not state.get("project_id"):
        return {"gaps": []}

    doc_type = state.get("document_type", "")
    project_id = state.get("project_id", "")
    sections_outline = state.get("sections_outline", [])
    final_document = state.get("final_document", [])
    current_draft = state.get("current_section_content", "")
    current_idx = state.get("current_section_index", 0)

    retriever = HybridRetriever(project_id=project_id, doc_type=None, recall_k=15, final_k=4)

    all_gaps: List[GapItem] = []

    # Sections to analyze: all approved + active draft (if any content exists)
    sections_to_check = list(final_document)
    if current_draft.strip() and current_idx < len(sections_outline):
        sections_to_check.append({
            "title": sections_outline[current_idx],
            "content": current_draft,
        })

    for section in sections_to_check:
        section_title = section.get("title", "")
        section_content = section.get("content", "")

        if not section_content.strip():
            continue

        requirements = get_section_requirements(doc_type, section_title)
        if not requirements:
            continue  # no requirements defined for this section/type

        try:
            section_index = sections_outline.index(section_title)
        except ValueError:
            section_index = -1

        # Retrieve meeting context to ground the options in real project data
        meeting_chunks = retriever.retrieve(section_title, expand=False)
        meeting_context = (
            "\n\n".join(f"[Excerpt {i+1}]\n{r['content']}" for i, r in enumerate(meeting_chunks))
            if meeting_chunks
            else "No meeting context available — use industry standards for options."
        )

        requirements_list = "\n".join(f"- {req}" for req in requirements)

        # Truncate to keep total prompt within Groq's token budget and prevent JSON truncation
        messages = [
            SystemMessage(content=_GAP_SYSTEM),
            HumanMessage(content=_GAP_HUMAN.format(
                doc_type=doc_type.upper(),
                section_title=section_title,
                section_content=section_content[:2500],
                requirements_list=requirements_list,
                meeting_context=meeting_context[:800],
            )),
        ]

        try:
            result: _SectionGapAnalysis = _structured_llm.invoke(messages)
        except Exception as e:
            print(f"[GapAnalysis] LLM call failed for section '{section_title}': {e}")
            continue

        for gap in result.gaps:
            options: List[GapOption] = [
                GapOption(option_id=o.option_id, text=o.text, source=o.source)
                for o in gap.options
            ]
            all_gaps.append(
                GapItem(
                    gap_id=str(uuid.uuid4()),
                    section_title=section_title,
                    section_index=section_index,
                    gap_type=gap.gap_type,
                    gap_description=gap.gap_description,
                    question=gap.question,
                    options=options,
                )
            )

    print(f"[GapAnalysis] Found {len(all_gaps)} gaps across {len(sections_to_check)} sections")
    return {"gaps": all_gaps}


# ── Stateless helper: resolve a gap into a Suggestion ────────────────────────

def resolve_gap_to_suggestion(
    document_id: str,
    section_title: str,
    section_index: int,
    section_content: str,
    gap_id: str,
    gap_description: str,
    question: str,
    selected_text: str,
) -> Optional[dict]:
    """
    Called directly by the API (not a graph node) to convert a user's gap answer
    into a precise inline suggestion using the same schema as the sync system.
    Returns None if the LLM call fails.
    """
    prompt = _RESOLVE_PROMPT.format(
        section_title=section_title,
        section_content=section_content,
        gap_description=gap_description,
        question=question,
        selected_text=selected_text,
    )

    try:
        result: _ResolveSuggestion = _resolve_llm.invoke(prompt)
    except Exception as e:
        print(f"[GapAnalysis] resolve_gap LLM call failed: {e}")
        return None

    # Validate verbatim constraint for "update" type
    if result.suggestion_type == "update" and result.original_text not in section_content:
        print("[GapAnalysis] resolve_gap: original_text not found verbatim, falling back to addition")
        suggestion_type = "addition"
        # Use last 100 chars of content as anchor
        original_text = section_content[-100:].strip()
    else:
        suggestion_type = result.suggestion_type
        original_text = result.original_text

    return {
        "suggestion_id": str(uuid.uuid4()),
        "section_title": section_title,
        "section_index": section_index,
        "original_text": original_text,
        "suggested_text": result.suggested_text,
        "suggestion_type": suggestion_type,
        "rationale": result.rationale,
        "source_meeting_id": None,
        "source_gap_id": gap_id,
        "confidence": 1.0,  # user-confirmed resolution is always high confidence
    }
