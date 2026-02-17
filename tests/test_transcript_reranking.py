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
    query_knowledge_base,
)
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import ingest_minutes

def test_transcript_reranking():
    print("--- Starting Transcript Reranking Verification ---")
    
    # 1. Setup
    session_id = str(uuid.uuid4())
    print(f"Testing with Session ID: {session_id}")
    
    ensure_collection_exists()

    # 2. Load and Ingest Transcripts
    t1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "transcript1.json"))
    t2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "transcript2.json"))
    
    with open(t1_path, "r") as f:
        t1_data = json.load(f)
    
    with open(t2_path, "r") as f:
        t2_data = json.load(f)
        
    print("\n[Step 1] Ingesting Transcript 1 (Jira/Slack Focus)...")
    ingest_minutes(session_id, t1_data)
    
    print("[Step 2] Ingesting Transcript 2 (Evaluation/Demo Focus)...")
    # Note: ingest_minutes deletes PREVIOUS transcript data for the SAME session.
    # To test retrieving from BOTH, they need to be treated as segments of the SAME session 
    # OR we need to disable the "delete_session_vectors" logic for this specific test 
    # OR we just append the text.
    
    # Actually, `ingest_minutes` calls `delete_session_vectors(session_id, "transcript")`.
    # If we call it twice for the SAME session, the second call WIPES the first.
    # CORRECT APPROACH FOR TEST: Combine them into one input or modify ingestion to append?
    # For now, let's simulate them being in the SAME meeting but maybe ingested together?
    # OR, we verify that `ingest_minutes` chunks them.
    
    # HACK for Test: We want both in the DB.
    # Option A: Use different session_ids (but then we can't search across both easily if we filter by session).
    # Option B: Manually ingest skipping the delete step.
    # Option C: Concatenate them.
    
    # Let's Concatenate for this test to simulate a longer meeting
    combined_text = t1_data["text"] + "\n\n" + t2_data["text"]
    # We lose strict word-mapping for speaker extraction on the combined string if we just cat text,
    # but `ingest_minutes` supports dict.
    # Let's just ingest T1, then T2? NO, T2 deletes T1.
    
    # Let's use `ingest_minutes` on T1, then MANUALLY add T2 chunks just to bypass deletion?
    # Or better: `ingest_minutes` should perhaps NOT delete if we say "append"? 
    # The current requirement is "Idempotent Ingestion", meaning "Replace".
    
    # OK, for this test, I will combine the TEXT and mock the words list if needed, 
    # or just test them one by one? No, we want to RERANK between them.
    
    print("Manual Upsert to ensure 2 distinct chunks (simulating 2 diff meetings or long pause)...")
    
    # We want T1 and T2 to be separate chunks to compare their scores.
    # ingest_minutes might merge them if they are short.
    
    from vocalog_ai_api.infrastructure.vector_store.qdrant import upsert_vectors, embed_documents, VectorPayload
    from qdrant_client.http import models
    
    # 1. Clear old
    delete_session_vectors(session_id)
    
    # 2. Prepare 2 chunks
    texts = [t1_data["text"], t2_data["text"]]
    vectors = embed_documents(texts)
    
    points = []
    for i, (txt, vec) in enumerate(zip(texts, vectors)):
        # Naive payload
        payload = VectorPayload(
            session_id=session_id,
            meeting_id=session_id,
            doc_type="transcript",
            chunk_type="speaker_turn",
            content=txt,
            section_id=None,
            speakers=[], # skipping extraction for this specific test
            timestamp=float(i),
            chunk_index=i
        )
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
        
    upsert_vectors(points) 
    
    time.sleep(1)
    
    # 3. Test Single Broad Query
    # We want to see how the reranker scores chunks from T1 vs T2 for the same query.
    query = "What features and integrations are we showing?"
    print(f"\n[Query]: '{query}'")
    
    # Fetch enough results to likely get chunks from both
    results = query_knowledge_base(query, session_id, "transcript", limit=10, enable_reranking=True)
    
    if not results:
        print("FAILURE: No results.")
        return

    print(f"\n{'Score':<10} | {'Content Snippet'}")
    print("-" * 60)
    
    for r in results:
        score = r.get("rerank_score", r.get("score", 0)) # Fallback if reranking fails/disabled
        # Try to identify origin based on unique keywords
        origin = "Unknown"
        if "Jira" in r["content"] or "Slack" in r["content"]:
            origin = "[T1 - Jira/Slack]"
        elif "Mid-Year" in r["content"] or "MoM" in r["content"]:
            origin = "[T2 - Eval/MoM]"
            
        snippet = r["content"][:40].replace("\n", " ") + "..."
        print(f"{score:>10.4f} | {origin} {snippet}")

if __name__ == "__main__":
    test_transcript_reranking()
