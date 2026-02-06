import sys
import os
import uuid
import time
from typing import List

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vocalog_ai_api.infrastructure.vector_store.qdrant import (
    ensure_collection_exists,
    delete_session_vectors,
    upsert_vectors,
    query_knowledge_base,
    get_qdrant_client,
    VOCALOG_MAIN_COLLECTION,
    VectorPayload
)
from qdrant_client.http import models

def test_vector_store_workflow():
    print("--- Starting Vector Store Verification ---")
    
    # 1. Setup
    session_id = str(uuid.uuid4())
    print(f"Testing with Session ID: {session_id}")
    
    ensure_collection_exists()
    client = get_qdrant_client()
    
    # 2. Ingest Data (Version 1)
    print("\n[Step 1] Ingesting Version 1...")
    
    # Simulate ingest logic manually to isolate infrastructure
    delete_session_vectors(session_id, "transcript")
    
    payload_v1 = VectorPayload(
        session_id=session_id,
        meeting_id=session_id,
        doc_type="transcript",
        chunk_type="test_chunk",
        content="This is the first version of the transcript.",
        section_id="intro",
        speaker="User",
        timestamp=0.0,
        chunk_index=0
    )
    
    # create a dummy vector (dim 384)
    dummy_vector = [0.1] * 384
    
    upsert_vectors([
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=dummy_vector,
            payload=payload_v1
        )
    ])
    
    # Allow some time for Qdrant to index (usually near instant for 1 point but good to be safe)
    time.sleep(1)
    
    # 3. Query Version 1
    results = query_knowledge_base("dummy query", session_id, "transcript")
    print(f"Query Results V1: {len(results)} matches")
    assert len(results) == 1
    assert results[0]['content'] == "This is the first version of the transcript."
    
    # 4. Ingest Data (Version 2) - Idempotency Check
    print("\n[Step 2] Ingesting Version 2 (Idempotency Check)...")
    
    # The key is calling delete_session_vectors first
    delete_session_vectors(session_id, "transcript")
    
    payload_v2 = VectorPayload(
        session_id=session_id,
        meeting_id=session_id,
        doc_type="transcript",
        chunk_type="test_chunk",
        content="This is the UPDATED version of the transcript.",
        section_id="intro",
        speaker="User",
        timestamp=0.0,
        chunk_index=0
    )
    
    upsert_vectors([
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=dummy_vector,
            payload=payload_v2
        )
    ])
    
    time.sleep(1)
    
    # 5. Query Version 2
    results = query_knowledge_base("dummy query", session_id, "transcript")
    print(f"Query Results V2: {len(results)} matches")
    
    # If idempotency works, we should STILL have only 1 result (the new one)
    if len(results) != 1:
        print(f"FAILURE: Expected 1 result, got {len(results)}")
        for r in results:
            print(f" - {r['content']}")
    else:
        print("SUCCESS: Count is correct.")
        
    assert len(results) == 1
    assert results[0]['content'] == "This is the UPDATED version of the transcript."
    
    print("\n--- Verification Passed! ---")

if __name__ == "__main__":
    test_vector_store_workflow()
