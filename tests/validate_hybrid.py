"""
Validation script for Module 5: Hybrid Retrieval.
Tests that BM25 keyword matching boosts results that pure vector search might miss.
"""
import sys
import os
import uuid
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vocalog_ai_api.infrastructure.vector_store.qdrant import (
    ensure_collection_exists,
    delete_session_vectors,
    upsert_vectors,
    embed_documents,
    query_knowledge_base,
    VectorPayload,
)
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.hybrid_retriever import (
    HybridRetriever,
)
from qdrant_client.http import models


def validate_hybrid():
    print("=" * 60)
    print("MODULE 5: Hybrid Retrieval Validation")
    print("=" * 60)

    session_id = str(uuid.uuid4())
    print(f"\nSession: {session_id}")

    ensure_collection_exists()
    delete_session_vectors(session_id)

    # --- Ingest test data ---
    texts = [
        "The timeline for Project X-12 has been delayed by two weeks due to vendor issues.",
        "We discussed the new onboarding flow for enterprise customers during the sprint review.",
        "The apple pie recipe requires fresh green apples and cinnamon.",
        "Hybrid cars combine electric motors with traditional combustion engines for efficiency.",
        "Raza mentioned that the Jira integration should push tasks automatically after MoM generation.",
    ]

    print(f"\nIngesting {len(texts)} chunks...")
    vectors = embed_documents(texts)

    points = []
    for i, (txt, vec) in enumerate(zip(texts, vectors)):
        payload = VectorPayload(
            session_id=session_id,
            meeting_id=session_id,
            doc_type="transcript",
            chunk_type="test",
            content=txt,
            section_id=None,
            speakers=[],
            timestamp=None,
            chunk_index=i,
        )
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    upsert_vectors(points)
    time.sleep(1)

    # --- Test 1: Keyword-heavy query ---
    query = "status of X-12"
    print(f"\n{'─' * 60}")
    print(f"TEST 1 — Keyword Query: '{query}'")
    print(f"{'─' * 60}")

    # Vector-only (no reranking, no BM25)
    vector_only = query_knowledge_base(query, session_id, "transcript", limit=3, enable_reranking=False)
    print("\n  Vector-only results:")
    for i, r in enumerate(vector_only):
        print(f"    #{i+1} [cosine: {r['score']:.4f}] {r['content'][:60]}...")

    # Hybrid
    retriever = HybridRetriever(session_id=session_id, doc_type="transcript", recall_k=10, final_k=3)
    hybrid = retriever.retrieve(query)
    print("\n  Hybrid (Vector + BM25 + RRF + Rerank) results:")
    for i, r in enumerate(hybrid):
        rrf = r.get("rrf_score", "N/A")
        rerank = r.get("rerank_score", "N/A")
        print(f"    #{i+1} [rrf: {rrf:.4f}, rerank: {rerank:.4f}] {r['content'][:60]}...")

    # Check
    if hybrid and "X-12" in hybrid[0]["content"]:
        print("\n  ✅ SUCCESS: Hybrid search found the keyword match as #1!")
    else:
        print("\n  ⚠️  WARNING: X-12 not top result. Check BM25 weighting.")

    # --- Test 2: Semantic-heavy query ---
    query2 = "how do we handle new client setup"
    print(f"\n{'─' * 60}")
    print(f"TEST 2 — Semantic Query: '{query2}'")
    print(f"{'─' * 60}")

    hybrid2 = retriever.retrieve(query2)
    print("\n  Hybrid results:")
    for i, r in enumerate(hybrid2):
        rrf = r.get("rrf_score", "N/A")
        rerank = r.get("rerank_score", "N/A")
        print(f"    #{i+1} [rrf: {rrf:.4f}, rerank: {rerank:.4f}] {r['content'][:60]}...")

    if hybrid2 and "onboarding" in hybrid2[0]["content"]:
        print("\n  ✅ SUCCESS: Semantic match still works in hybrid mode!")
    else:
        print("\n  ⚠️  WARNING: Onboarding chunk not top result.")

    print(f"\n{'=' * 60}")
    print("Validation Complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    validate_hybrid()
