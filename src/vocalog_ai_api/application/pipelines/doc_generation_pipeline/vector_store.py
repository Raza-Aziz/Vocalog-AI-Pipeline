import uuid
import hashlib
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

from vocalog_ai_api.infrastructure.vector_store.qdrant import (
    ensure_collection_exists,
    delete_meeting_vectors,
    upsert_vectors,
    embed_documents,
    query_knowledge_base,
    embed_text,
    VectorPayload,
)


def _chunk_id(project_id: str, meeting_id: str, chunk_index: int) -> str:
    """Deterministic UUID derived from project, meeting, and chunk position.

    Ensures re-ingesting the same meeting upserts in place rather than creating
    new points, while new meetings always append to the project knowledge base.
    """
    raw = f"{project_id}:{meeting_id}:{chunk_index}".encode()
    return str(uuid.UUID(bytes=hashlib.sha256(raw).digest()[:16]))


# TODO: Change Splitter to Semantic Chunking by langchain_experimental
def ingest_minutes(project_id: str, meeting_id: str, input_data: str | dict):
    """
    Chunks, vectorizes, and upserts a single meeting's transcript into the
    project-level knowledge base.

    Idempotency: existing vectors for this (project_id, meeting_id) pair are
    purged before ingestion so re-submitting corrected source material stays in
    sync without duplicating chunks or touching other meetings.

    Args:
        project_id: Project identifier — scopes retrieval across all meetings.
        meeting_id: Unique identifier for this specific meeting.
        input_data: Raw transcript string OR structured dict with 'text' + 'words'.
    """
    ensure_collection_exists()

    # Targeted deletion: removes only this meeting's chunks, leaves all others intact.
    delete_meeting_vectors(project_id=project_id, meeting_id=meeting_id, doc_type="transcript")

    # Process input
    if isinstance(input_data, dict) and "text" in input_data:
        full_text = input_data["text"]
        word_metadata = input_data.get("words", [])
    elif isinstance(input_data, str):
        full_text = input_data
        word_metadata = []
    else:
        raise ValueError("input_data must be a transcript string or a dict with a 'text' field.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_text(full_text)

    if not chunks:
        raise ValueError(f"No chunks produced from meeting {meeting_id} — source material may be empty.")

    print(f"Embedding {len(chunks)} chunks for project={project_id}, meeting={meeting_id}...")
    vectors = embed_documents(chunks)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_speakers: List[str] = []
        if word_metadata:
            unique_speakers = {w.get("speaker_id") for w in word_metadata if w.get("speaker_id")}
            chunk_speakers = [spk for spk in unique_speakers if spk in chunk]

        payload = VectorPayload(
            project_id=project_id,
            session_id=project_id,  # backward compat for non-doc-gen pipelines
            meeting_id=meeting_id,
            doc_type="transcript",
            chunk_type="speaker_turn",
            content=chunk,
            section_id=None,
            speakers=chunk_speakers,
            timestamp=None,
            chunk_index=i,
        )

        points.append(
            models.PointStruct(
                id=_chunk_id(project_id, meeting_id, i),
                vector=vector,
                payload=payload,
            )
        )

    upsert_vectors(points)
    print(f"--- Vector Store: Upserted {len(points)} chunks for project={project_id}, meeting={meeting_id} ---")


def retrieve_context(project_id: str, query: str, limit: int = 3) -> str:
    """
    Retrieves context relevant to 'query' across all meetings in the project.
    Uses HybridRetriever: Vector + BM25 + RRF Fusion + Reranking.
    """
    from vocalog_ai_api.application.pipelines.doc_generation_pipeline.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        project_id=project_id,
        doc_type="transcript",
        recall_k=20,
        final_k=limit,
    )

    results = retriever.retrieve(query, expand=True)

    if not results:
        return "No relevant context found in meeting minutes."

    return "\n\n---\n\n".join(r["content"] for r in results)
