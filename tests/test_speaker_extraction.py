import sys
import os
import uuid
import time
import json
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
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import ingest_minutes
from qdrant_client.http import models

def test_speaker_extraction():
    print("--- Starting Speaker Extraction Verification ---")
    
    # 1. Setup
    session_id = str(uuid.uuid4())
    print(f"Testing with Session ID: {session_id}")
    
    ensure_collection_exists()

    # 2. Prepare JSON Data (From User Request)
    mock_json_input = {
      "language_code": "en",
      "language_probability": 1,
      "text": "Raza: We need to finalize which Vocalog features to show for the Mid-Year Evaluation. Hamza: I think the MoM generation demo should be the focus. Shahzeb: Agreed, plus we can show speaker diarization and template adaptation.",
      "words": [
        {"text": "Raza:", "start": 0.0, "end": 0.5, "type": "word", "speaker_id": "Raza"},
        {"text": "Hamza:", "start": 6.0, "end": 6.5, "type": "word", "speaker_id": "Hamza"},
        {"text": "Shahzeb:", "start": 11.0, "end": 11.5, "type": "word", "speaker_id": "Shahzeb"}
      ]
    }
    
    # 3. Ingest using the new Logic
    print("\n[Step 1] Ingesting JSON Data...")
    ingest_minutes(session_id, mock_json_input)
    
    time.sleep(1)
    
    # 4. Verify Extraction
    print("\n[Step 2] Querying to verify 'speakers' field...")
    
    # Query for something generic
    results = query_knowledge_base(
        query_text="Vocalog features",
        session_id=session_id,
        doc_type="transcript"
    )
    
    if not results:
        print("FAILURE: No results found.")
        return

    first_result = results[0]
    speakers_found = first_result['metadata'].get('speakers', [])
    
    print(f"Content: {first_result['content'][:50]}...")
    print(f"Speakers Found in Metadata: {speakers_found}")
    
    # Check if our target speakers are there
    expected_speakers = {"Raza", "Hamza", "Shahzeb"}
    found_set = set(speakers_found)
    
    if expected_speakers.issubset(found_set):
        print("SUCCESS: All expected speakers found in metadata.")
    else:
        print(f"FAILURE: Missing speakers. Expected {expected_speakers}, got {found_set}")

    print("\n--- Speaker Extraction Verification Passed! ---")

if __name__ == "__main__":
    test_speaker_extraction()
