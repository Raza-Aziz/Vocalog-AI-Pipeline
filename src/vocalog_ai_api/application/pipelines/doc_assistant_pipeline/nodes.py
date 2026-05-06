from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from vocalog_ai_api.application.pipelines.doc_assistant_pipeline.state import DocAssistantState, AssistantChunk
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.hybrid_retriever import HybridRetriever
from vocalog_ai_api.infrastructure.llm_providers.groq import llm

_RETRIEVAL_RECALL_K = 25
_RETRIEVAL_FINAL_K = 8   # after multi-probe dedup and rerank

_SYSTEM_PROMPT = """\
You are a Document Generation Assistant — an intelligent companion helping users audit, \
validate, and understand their technical documentation as it is being built.

You have access to two sources of truth:

1. DOCUMENT STATE — a live snapshot of the document under construction:
   - Approved sections (finalised content)
   - The active section draft currently under human review
   - The full refinement/feedback history (what the user asked to change and why)

2. MEETING CONTEXT — hybrid-retrieved excerpts from all meeting transcripts and formal \
   minutes that were used to generate the document.

Your responsibilities:
- Explain WHY a specific requirement or design decision exists by tracing it back to the \
  meeting discussion or formal minutes where it was first raised.
- Answer WHERE a piece of content came from ("What meeting decided the auth approach?").
- Clarify HOW the document evolved ("Why was this section refined?").
- Cross-reference the approved sections with raw meeting dialogue to surface supporting \
  or contradicting evidence.
- Identify if a participant explicitly denied, approved, or committed to something that \
  appears (or should appear) in the document.

Citation rules — use these labels inline in your answer:
  [SECTION-N]   → approved document section
  [DRAFT]       → the active section currently under review
  [FEEDBACK-N]  → a specific refinement cycle recorded in the history
  [MEETING-N]   → a retrieved meeting transcript or minutes chunk

If the context is insufficient to answer, say so — never fabricate. \
Output the entire response in well-structured markdown.\
"""


# ── Node 1: Load document state from SQLite checkpoint ────────────────────────

def load_document_state(state: DocAssistantState) -> dict:
    """
    Reads the live DocumentGenerationState from the SQLite checkpoint keyed by
    document_id (= LangGraph thread_id).  Populates all doc-state fields so
    subsequent nodes have full situational awareness.
    """
    # Import here to avoid circular imports (doc_gen_graph imports from this package tree)
    from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import doc_gen_graph

    document_id = state["document_id"]
    config = {"configurable": {"thread_id": document_id}}

    checkpoint = doc_gen_graph.get_state(config)
    if checkpoint is None or not checkpoint.values:
        return {
            "document_found": False,
            "project_id": "",
            "document_type": "",
            "sections_outline": [],
            "current_section_index": 0,
            "current_section_content": "",
            "final_document": [],
            "refinement_history": {},
            "meeting_sources": [],
            "is_complete": False,
        }

    doc = checkpoint.values
    return {
        "document_found": True,
        "project_id": doc.get("project_id", ""),
        "document_type": doc.get("document_type", ""),
        "sections_outline": doc.get("sections_outline", []),
        "current_section_index": doc.get("current_section_index", 0),
        "current_section_content": doc.get("current_section_content", ""),
        "final_document": list(doc.get("final_document", [])),
        "refinement_history": dict(doc.get("refinement_history", {})),
        "meeting_sources": list(doc.get("meeting_sources", [])),
        "is_complete": bool(doc.get("is_complete", False)),
    }


# ── Node 2: Multi-probe hybrid retrieval across the project knowledge base ────

def retrieve_project_context(state: DocAssistantState) -> dict:
    """
    Generates multiple query probes from the user question + document context
    and runs each through HybridRetriever (Vector → BM25 → RRF → CrossEncoder).

    Probe strategy:
      1. Raw user question — primary intent
      2. Active section title + question — anchors retrieval to what's being drafted
      3. Question + first 150 chars of current draft — surface supporting evidence
         for refinement/rationale questions

    Results are deduplicated by Qdrant point ID, re-sorted by score, and capped
    at _RETRIEVAL_FINAL_K so the LLM context stays focused.
    """
    if not state.get("document_found") or not state.get("project_id"):
        return {"retrieved_chunks": []}

    project_id = state["project_id"]
    question = state["question"]
    sections_outline = state.get("sections_outline", [])
    current_idx = state.get("current_section_index", 0)
    current_draft = state.get("current_section_content", "")

    probes: List[str] = [question]

    if sections_outline and current_idx < len(sections_outline):
        active_section = sections_outline[current_idx]
        probes.append(f"{active_section}: {question}")

    if current_draft and len(current_draft) > 50:
        probes.append(f"{question} {current_draft[:150]}")

    retriever = HybridRetriever(
        project_id=project_id,
        doc_type=None,      # no doc_type filter — retrieve from transcript AND mom
        recall_k=_RETRIEVAL_RECALL_K,
        final_k=_RETRIEVAL_FINAL_K,
    )

    seen_ids: set = set()
    all_chunks: List[AssistantChunk] = []

    for probe in probes:
        results = retriever.retrieve(probe, expand=True)
        for r in results:
            point_id = str(r.get("id", ""))
            if point_id in seen_ids:
                continue
            seen_ids.add(point_id)
            all_chunks.append(
                AssistantChunk(
                    content=r["content"],
                    meeting_id=r["metadata"].get("meeting_id", ""),
                    doc_type=r["metadata"].get("doc_type", ""),
                    chunk_index=r["metadata"].get("chunk_index", 0),
                    speakers=r["metadata"].get("speakers", []),
                    score=float(r.get("rerank_score", r.get("score", 0.0))),
                )
            )

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    return {"retrieved_chunks": all_chunks[:_RETRIEVAL_FINAL_K]}


# ── Node 3: Build context and generate grounded answer ────────────────────────

def generate_answer(state: DocAssistantState) -> dict:
    """
    Assembles a layered context block (document state + meeting evidence) and
    prompts the LLM to produce a citation-rich, markdown-formatted answer.

    Context layers (in order):
      1. Document status summary
      2. Approved sections (truncated to 600 chars each)
      3. Active draft (truncated to 800 chars)
      4. Refinement / feedback history
      5. Retrieved meeting chunks
    """
    if not state.get("document_found"):
        return {
            "answer": (
                "**Document not found.**\n\n"
                f"No document with ID `{state['document_id']}` exists in the system. "
                "Please verify the `document_id` returned by `POST /generate-document`."
            ),
            "citations": [],
        }

    question = state["question"]
    doc_type = state.get("document_type", "").upper()
    sections_outline: List[str] = state.get("sections_outline", [])
    current_idx: int = state.get("current_section_index", 0)
    current_draft: str = state.get("current_section_content", "")
    final_document: List[Dict] = state.get("final_document", [])
    refinement_history: Dict = state.get("refinement_history", {})
    meeting_sources: List[Dict] = state.get("meeting_sources", [])
    retrieved_chunks: List[AssistantChunk] = state.get("retrieved_chunks", [])
    is_complete: bool = state.get("is_complete", False)

    approved_count = len(final_document)
    total_sections = len(sections_outline)
    status_label = "completed" if is_complete else "in progress"
    active_title = (
        sections_outline[current_idx]
        if sections_outline and current_idx < len(sections_outline)
        else "—"
    )
    meeting_ids = [s.get("meeting_id", "") for s in meeting_sources if s.get("meeting_id")]

    # ── Block 1: Document Status ─────────────────────────────────────────────
    context_parts: List[str] = [
        "=== DOCUMENT STATUS ===",
        (
            f"Type: {doc_type} | Status: {status_label} "
            f"| Progress: {approved_count}/{total_sections} sections approved"
        ),
        f"Active Section: \"{active_title}\"",
        f"Source Meetings: {', '.join(meeting_ids) if meeting_ids else 'none recorded'}",
        f"Section Outline: {' → '.join(sections_outline) if sections_outline else 'not yet generated'}",
    ]

    # ── Block 2: Approved Sections ───────────────────────────────────────────
    if final_document:
        context_parts.append("\n=== APPROVED SECTIONS ===")
        for i, section in enumerate(final_document, start=1):
            body = section.get("content", "")
            excerpt = body[:600] + ("…" if len(body) > 600 else "")
            context_parts.append(f"[SECTION-{i}: {section.get('title', '')}]\n{excerpt}")

    # ── Block 3: Active Draft ────────────────────────────────────────────────
    if current_draft and not is_complete:
        body = current_draft[:800] + ("…" if len(current_draft) > 800 else "")
        context_parts.append(f"\n=== CURRENT DRAFT (pending review) ===\n[DRAFT: {active_title}]\n{body}")

    # ── Block 4: Refinement / Feedback History ────────────────────────────────
    if refinement_history:
        context_parts.append("\n=== REFINEMENT HISTORY ===")
        feedback_counter = 1
        for section_idx_str, entries in refinement_history.items():
            idx = int(section_idx_str)
            sec_title = sections_outline[idx] if idx < len(sections_outline) else f"Section {idx}"
            for entry in entries:
                prior = entry.get("draft", "")[:200]
                feedback = entry.get("feedback", "")
                context_parts.append(
                    f"[FEEDBACK-{feedback_counter}] Section: \"{sec_title}\"\n"
                    f"  Prior draft excerpt: {prior}…\n"
                    f"  User feedback: {feedback}"
                )
                feedback_counter += 1

    # ── Block 5: Retrieved Meeting Context ───────────────────────────────────
    if retrieved_chunks:
        context_parts.append("\n=== RELEVANT MEETING CONTEXT ===")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            speaker_label = (
                f" [Speakers: {', '.join(chunk['speakers'])}]" if chunk["speakers"] else ""
            )
            source_type = chunk["doc_type"].upper()
            context_parts.append(
                f"[MEETING-{i} | {source_type} | meeting_id={chunk['meeting_id']}]{speaker_label}\n"
                f"{chunk['content']}"
            )

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)

    # ── Citations list ────────────────────────────────────────────────────────
    citations: List[Dict[str, Any]] = []

    for i, section in enumerate(final_document, start=1):
        body = section.get("content", "")
        citations.append({
            "label": f"SECTION-{i}",
            "source": "document",
            "doc_type": "approved_section",
            "section_title": section.get("title", ""),
            "meeting_id": None,
            "chunk_index": i - 1,
            "speakers": [],
            "excerpt": body[:200] + "…" if len(body) > 200 else body,
        })

    if current_draft and not is_complete:
        citations.append({
            "label": "DRAFT",
            "source": "document",
            "doc_type": "active_draft",
            "section_title": active_title,
            "meeting_id": None,
            "chunk_index": current_idx,
            "speakers": [],
            "excerpt": current_draft[:200] + "…" if len(current_draft) > 200 else current_draft,
        })

    for i, chunk in enumerate(retrieved_chunks, start=1):
        citations.append({
            "label": f"MEETING-{i}",
            "source": "meeting",
            "doc_type": chunk["doc_type"],
            "section_title": None,
            "meeting_id": chunk["meeting_id"],
            "chunk_index": chunk["chunk_index"],
            "speakers": chunk["speakers"],
            "excerpt": chunk["content"][:200] + "…" if len(chunk["content"]) > 200 else chunk["content"],
        })

    return {"answer": response.content, "citations": citations}
