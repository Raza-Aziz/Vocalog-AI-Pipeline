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
def ingest_minutes(session_id: str, input_data: str | dict):
    """
    Chunks, vectorizes, and stores in the Unified Knowledge Base.
    ENFORCES IDEMPOTENCY: Deletes old 'transcript' chunks for this session first.
    
    Args:
        session_id: The session UUID.
        input_data: Can be raw text string OR a JSON-like dict with word-level metadata.
    """
    ensure_collection_exists()

    # 1. IDEMPOTENCY: Delete existing transcript info for this session
    # This ensures re-running generation doesn't duplicate data
    delete_session_vectors(session_id=session_id, doc_type="transcript")

    # 2. Process Input (Text vs JSON)
    full_text = ""
    word_metadata = [] # To assist with speaker extraction if available
    
    if isinstance(input_data, dict) and "text" in input_data:
        # User provided the JSON format with 'text' and 'words'
        full_text = input_data["text"]
        word_metadata = input_data.get("words", [])
    elif isinstance(input_data, str):
        full_text = input_data
    else:
        print("Error: Input data must be string or dict with 'text' field.")
        return

    # 3. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        print("Warning: No chunks created from text.")
        return

    # 4. Create Embeddings (Batch)
    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_documents(chunks)

    # 5. Prepare Points with Unified Schema
    points = []
    
    # Simple strategy: If we have word metadata, we try to map chunks back to speakers.
    # Since splitting is by character, this is an approximation.
    # For a robust solution, we'd split the WORD LIST directly, not the raw text.
    # For now, we'll do a basic regex-like check: "Does this chunk contain text attributed to Speaker X?"
    
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        
        # Extract unique speakers present in this chunk
        # If we have word metadata, checking intersection is safer
        chunk_speakers = []
        if word_metadata:
             # Heuristic: If a word's text is in the chunk, add its speaker.
             # Note: This is O(N*M) but N(words) and M(chunk size) are small enough for now.
             # Optimization: use start/end char indices if available (JSON has times, not char indices, but order is preserved).
             
             # Faster approach: valid speakers in this session
             unique_speakers = set(w.get("speaker_id") for w in word_metadata if w.get("speaker_id"))
             for spk in unique_speakers:
                 # Check if the speaker's name appears as a label (e.g. "Raza:") inside the chunk
                 # OR if we want to be strict, we check if words attributed to this speaker are in the chunk.
                 # Given the text format "Raza: We need...", finding "Raza" is likely part of the text.
                 if spk in chunk: # formatting usually implies "Speaker:" is part of text
                     chunk_speakers.append(spk)
        
        if not chunk_speakers: 
             # Fallback or if no metadata provided
             chunk_speakers = []

        # Construct strict payload
        payload = VectorPayload(
            session_id=session_id,
            meeting_id=session_id, # Assuming 1:1 for now
            doc_type="transcript",
            chunk_type="speaker_turn", 
            content=chunk,
            section_id=None,
            speakers=chunk_speakers,
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
