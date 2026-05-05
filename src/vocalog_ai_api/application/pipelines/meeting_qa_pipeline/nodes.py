from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from vocalog_ai_api.application.pipelines.meeting_qa_pipeline.state import MeetingQAState, RetrievedChunk
from vocalog_ai_api.infrastructure.llm_providers.groq import llm
from vocalog_ai_api.infrastructure.vector_store.qdrant import query_by_meeting

_TRANSCRIPT_LIMIT = 5
_MOM_LIMIT = 4

_SYSTEM_PROMPT = """\
You are a precise meeting analysis assistant. Answer the user's question using ONLY \
the context provided below — never fabricate or infer beyond it.

Guidelines:
- Cross-reference the formal Minutes (MOM) with raw Transcript dialogue for nuanced answers.
- When quoting a speaker, attribute by name if available (e.g. "Alice explicitly said...").
- Distinguish between what was formally recorded in the Minutes vs what was said in discussion.
- If someone denied a request, committed to a fix, or raised a concern, call that out directly.
- If the context is insufficient to answer, state that clearly rather than guessing.
- Structure the response with a direct answer first, then supporting evidence.
- Cite sources inline as [MOM-N] or [TRANSCRIPT-N] matching the labels in the context.\
- Give the whole output in proper markdown
"""


def _to_retrieved_chunk(r: Dict[str, Any], doc_type: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=r["content"],
        doc_type=doc_type,
        chunk_index=r["metadata"].get("chunk_index", 0),
        speakers=r["metadata"].get("speakers", []),
        score=float(r.get("rerank_score", r.get("score", 0.0))),
    )


def retrieve_context(state: MeetingQAState) -> dict:
    """
    Queries Qdrant for both transcript and MoM chunks scoped strictly to
    the requested meeting_id. Uses CrossEncoder reranking per source.
    """
    meeting_id = state["meeting_id"]
    question = state["question"]

    raw_transcript = query_by_meeting(
        query_text=question,
        meeting_id=meeting_id,
        doc_type="transcript",
        limit=_TRANSCRIPT_LIMIT,
        enable_reranking=True,
    )
    raw_mom = query_by_meeting(
        query_text=question,
        meeting_id=meeting_id,
        doc_type="mom",
        limit=_MOM_LIMIT,
        enable_reranking=True,
    )

    return {
        "transcript_chunks": [_to_retrieved_chunk(r, "transcript") for r in raw_transcript],
        "mom_chunks": [_to_retrieved_chunk(r, "mom") for r in raw_mom],
    }


def generate_answer(state: MeetingQAState) -> dict:
    """
    Builds a labelled context block from retrieved chunks and prompts the LLM
    for a grounded, citation-rich answer. Returns answer text and a citations list.
    """
    question = state["question"]
    transcript_chunks: List[RetrievedChunk] = state.get("transcript_chunks", [])
    mom_chunks: List[RetrievedChunk] = state.get("mom_chunks", [])

    if not transcript_chunks and not mom_chunks:
        return {
            "answer": (
                "No relevant content found for this meeting. "
                "The meeting may not have been ingested yet, or the question does not relate "
                "to any topics discussed in this session."
            ),
            "citations": [],
        }

    context_parts: List[str] = []

    if mom_chunks:
        context_parts.append("=== FORMAL MINUTES (MOM) ===")
        for i, chunk in enumerate(mom_chunks, start=1):
            context_parts.append(f"[MOM-{i}]\n{chunk['content']}")

    if transcript_chunks:
        context_parts.append("\n=== RAW DISCUSSION TRANSCRIPT ===")
        for i, chunk in enumerate(transcript_chunks, start=1):
            speaker_label = (
                f" [Speakers: {', '.join(chunk['speakers'])}]" if chunk["speakers"] else ""
            )
            context_parts.append(f"[TRANSCRIPT-{i}]{speaker_label}\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)

    citations: List[Dict[str, Any]] = []
    for i, chunk in enumerate(mom_chunks, start=1):
        citations.append({
            "label": f"MOM-{i}",
            "source": "minutes",
            "doc_type": "mom",
            "chunk_index": chunk["chunk_index"],
            "speakers": chunk["speakers"],
            "excerpt": chunk["content"][:250] + "…" if len(chunk["content"]) > 250 else chunk["content"],
        })
    for i, chunk in enumerate(transcript_chunks, start=1):
        citations.append({
            "label": f"TRANSCRIPT-{i}",
            "source": "transcript",
            "doc_type": "transcript",
            "chunk_index": chunk["chunk_index"],
            "speakers": chunk["speakers"],
            "excerpt": chunk["content"][:250] + "…" if len(chunk["content"]) > 250 else chunk["content"],
        })

    return {"answer": response.content, "citations": citations}
