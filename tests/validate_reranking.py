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
    embed_documents,
    VectorPayload
)
from qdrant_client.http import models

def validate_reranking():
    print("--- Starting Reranking Validation ---")
    
    # 1. Setup
    session_id = str(uuid.uuid4())
    print(f"Testing with Session ID: {session_id}")
    
    ensure_collection_exists()
    delete_session_vectors(session_id)

    # 2. Prepare Tricky Data
    # We want sentences that are semantically close in vector space but distinct for specific queries.
    texts = [
        "The apple is a sweet, edible fruit produced by an apple tree.", # Index 0
        "Apple Inc. shares hit a record high after the new iPhone release.", # Index 1
        "You can make delicious apple pie using granny smith apples.", # Index 2
        "Big technology companies like Google and Microsoft are competitors.", # Index 3 (No keyword, but contextually related to tech)
        "Oranges and bananas are also popular fruits." # Index 4
    ]
    
    print(f"\n[Step 1] Ingesting {len(texts)} chunks...")
    vectors = embed_documents(texts)
    
    points = []
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        payload = VectorPayload(
            session_id=session_id,
            meeting_id=session_id,
            doc_type="transcript",
            chunk_type="test",
            content=text,
            section_id=None,
            speakers=[],
            timestamp=None,
            chunk_index=i
        )
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
        
    upsert_vectors(points)
    time.sleep(1) # Indexing wait
    
    # 3. Test Query: "Apple stock market"
    # Expected: 
    # - Vector Search might pull fruit stuff because of the word "Apple".
    # - Reranker should push "Apple Inc." (Index 1) to #1 and drop fruit stuff.
    
    query = "latest financial report for Apple"
    print(f"\n[Step 2] Querying: '{query}'")
    
    # Run WITHOUT Reranking (Simulated by asking for top 5 directly)
    print("\n--- Raw Vector Search (Top 3) ---")
    raw_results = query_knowledge_base(query, session_id, "transcript", limit=3, enable_reranking=False)
    for r in raw_results:
        print(f"[{r['score']:.4f}] {r['content'][:60]}...")
        
    # Run WITH Reranking
    print("\n--- Reranked Search (Top 3) ---")
    reranked_results = query_knowledge_base(query, session_id, "transcript", limit=3, enable_reranking=True)
    for r in reranked_results:
        # Note: score here is the CrossEncoder score (logits), might be negative or > 1
        print(f"[{r.get('rerank_score', 0):.4f}] {r['content'][:60]}...")
        
    # Validation Logic
    # The "Apple Inc" sentence should be higher ranked in the Reranked list than in Raw list (or at least #1)
    
    if not reranked_results:
        print("FAILURE: No results returned.")
        return

    top_reranked = reranked_results[0]['content']
    if "Apple Inc" in top_reranked:
        print("\nSUCCESS: Reranker correctly identified the Tech context as #1!")
    else:
        print(f"\nWARNING: Top result was '{top_reranked}'. Expected Apple Inc.")

if __name__ == "__main__":
    validate_reranking()
