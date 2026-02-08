import sys
import os
import uuid
import time
from typing import List

# Add src to path
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

def test_independent_retrieval():
    print("--- Starting Independent Retrieval Verification ---")
    
    # 1. Setup
    session_id = str(uuid.uuid4())
    print(f"Testing with Session ID: {session_id}")
    
    ensure_collection_exists()
    
    # Clean up any potential stale data for this UUID (unlikely but good practice)
    delete_session_vectors(session_id) 

    # 2. Prepare Data: 1 Transcript Chunk, 1 Document Section
    # We maintain the SAME content topic to ensure the vector similarity would match both,
    # proving that the Metadata Filter is what separates them.
    
    transcript_text = "The architect mentioned that microservices are the way to go for scalability."
    section_text = "Executive Summary: The team decided to adopt a microservices architecture."
    
    # Fake vector (dim 384) - in real life these would be close, but we want to prove filtering works
    dummy_vector = [0.1] * 384 
    
    print("\n[Step 1] Ingesting Data...")
    
    # Ingest Transcript
    payload_transcript = VectorPayload(
        session_id=session_id,
        meeting_id=session_id,
        doc_type="transcript",      # <--- KEY TAG
        chunk_type="speaker_turn",
        content=transcript_text,
        section_id=None,
        speaker="Architect",
        timestamp=10.5,
        chunk_index=0
    )
    
    # Ingest Generated Section
    payload_section = VectorPayload(
        session_id=session_id,
        meeting_id=session_id,
        doc_type="srs_section",     # <--- KEY TAG
        chunk_type="section",
        content=section_text,
        section_id="Summary",
        speaker=None,
        timestamp=None,
        chunk_index=0
    )
    
    upsert_vectors([
        models.PointStruct(id=str(uuid.uuid4()), vector=dummy_vector, payload=payload_transcript),
        models.PointStruct(id=str(uuid.uuid4()), vector=dummy_vector, payload=payload_section)
    ])
    
    time.sleep(1) # Allow for indexing
    
    # 3. Test Retrieval: Ask for TRANSCRIPT only
    print("\n[Step 2] Querying for TRANSCRIPT only...")
    results_transcript = query_knowledge_base(
        query_text="microservices architecture", # Theoretically matches both
        session_id=session_id,
        doc_type="transcript" # <--- FILTER
    )
    
    print(f"Transcript Results: {len(results_transcript)}")
    for r in results_transcript:
        print(f" - Found: {r['content']} (Type: {r['metadata']['doc_type']})")
        
    # Assertion: Should only find the transcript
    if len(results_transcript) == 1 and results_transcript[0]['metadata']['doc_type'] == "transcript":
        print("SUCCESS: Retrieved only transcript.")
    else:
        print("FAILURE: Transcript retrieval returned unexpected results.")

        
    # 4. Test Retrieval: Ask for SECTIONS only
    print("\n[Step 3] Querying for SECTIONS only...")
    results_section = query_knowledge_base(
        query_text="microservices architecture",
        session_id=session_id,
        doc_type="srs_section" # <--- FILTER
    )
    
    print(f"Section Results: {len(results_section)}")
    for r in results_section:
        print(f" - Found: {r['content']} (Type: {r['metadata']['doc_type']})")

    # Assertion: Should only find the section
    if len(results_section) == 1 and results_section[0]['metadata']['doc_type'] == "srs_section":
        print("SUCCESS: Retrieved only section.")
    else:
        print("FAILURE: Section retrieval returned unexpected results.")

    # 5. Test Retrieval: Ask for EVERYTHING (No doc_type filter)
    print("\n[Step 4] Querying without filter (Should get both)...")
    results_all = query_knowledge_base(
        query_text="microservices architecture",
        session_id=session_id,
        doc_type=None # <--- NO FILTER
    )
    print(f"Total Results: {len(results_all)}")
    if len(results_all) == 2:
        print("SUCCESS: Retrieved both when filter is removed.")
    else:
        print("FAILURE: Expected 2 results.")

    print("\n--- Independent Retrieval Verification Passed! ---")

if __name__ == "__main__":
    test_independent_retrieval()
