import uuid
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

# Import Centralized Infrastructure
from vocalog_ai_api.infrastructure.vector_store.qdrant import (
    ensure_collection_exists,
    delete_session_vectors,
    upsert_vectors,
    embed_documents,
    query_knowledge_base,
    embed_text, # Used in retrieval if needed, though query_knowledge_base handles it
    VectorPayload
)

# TODO: Change Splitter to Semantic Chunking by langchain_experimental
def ingest_minutes(session_id: str, text: str):
    """
    Chunks, vectorizes, and stores in the Unified Knowledge Base.
    ENFORCES IDEMPOTENCY: Deletes old 'transcript' chunks for this session first.
    """
    ensure_collection_exists()

    # 1. IDEMPOTENCY: Delete existing transcript info for this session
    # This ensures re-running generation doesn't duplicate data
    delete_session_vectors(session_id=session_id, doc_type="transcript")

    # 2. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        print("Warning: No chunks created from text.")
        return

    # 3. Create Embeddings (Batch)
    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_documents(chunks)

    # 4. Prepare Points with Unified Schema
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        
        # Construct strict payload
        payload = VectorPayload(
            session_id=session_id,
            meeting_id=session_id, # Assuming 1:1 for now
            doc_type="transcript",
            chunk_type="srs_section", # Defaulting to general section for now, ideally 'transcript_chunk'
            content=chunk,
            section_id=None,
            speaker=None,
            timestamp=None,
            chunk_index=i
        )
        
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        )

    # 5. Upload to Centralized Knowledge Base
    upsert_vectors(points)
    print(f"--- Vector Store: Indexed {len(points)} chunks for session {session_id} ---")


def retrieve_context(session_id: str, query: str, limit: int = 3) -> str:
    """
    Searches for text relevant to 'query' within the specific 'session_id'.
    """
    # 1. Search using Centralized Logic
    results = query_knowledge_base(
        query_text=query,
        session_id=session_id,
        doc_type="transcript", # We primarily want transcripts for context? Or maybe None for all? Defaulting to transcript for compatibility
        limit=limit
    )

    # 2. Format results
    if not results:
        return "No relevant context found in meeting minutes."

    # query_knowledge_base returns dicts with 'content' key
    context_blocks = [r["content"] for r in results]
    return "\n\n---\n\n".join(context_blocks)
