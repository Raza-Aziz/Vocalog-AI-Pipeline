"""
Hybrid Retriever validation using real transcript1.json and transcript2.json.
Ingests both as separate chunks in the same session, then queries with:
  - Keyword-exact queries (where BM25 should help)
  - Semantic queries (where vector search shines) 
Shows scores from both Vector-only and Hybrid pipelines side by side.
"""
import sys
import os
import json
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


def run_test():
    print("=" * 65)
    print("  HYBRID RETRIEVER TEST — Real Transcripts")
    print("=" * 65)

    session_id = str(uuid.uuid4())
    print(f"\n  Session: {session_id}")

    # --- Load Transcripts ---
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, "transcript1.json"), "r") as f:
        t1 = json.load(f)
    with open(os.path.join(script_dir, "transcript2.json"), "r") as f:
        t2 = json.load(f)

    texts = [t1["text"], t2["text"]]
    labels = ["T1 (Jira/Slack)", "T2 (Mid-Year Eval)"]

    # --- Ingest as 2 separate chunks ---
    ensure_collection_exists()
    delete_session_vectors(session_id)

    print(f"\n  Ingesting {len(texts)} transcript chunks...")
    vectors = embed_documents(texts)

    points = []
    for i, (txt, vec) in enumerate(zip(texts, vectors)):
        payload = VectorPayload(
            session_id=session_id,
            meeting_id=session_id,
            doc_type="transcript",
            chunk_type="speaker_turn",
            content=txt,
            section_id=None,
            speakers=[],
            timestamp=None,
            chunk_index=i,
        )
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    upsert_vectors(points)
    time.sleep(1)

    # --- Queries ---
    queries = [
        # (query, description, expected keyword in top result)
        ("Jira", "Exact keyword — only T1 mentions Jira", "Jira"),
        ("Slack notifications with deadlines", "Keyword + semantic — T1 topic", "Slack"),
        ("Mid-Year Evaluation features", "Keyword + semantic — T2 topic", "Mid-Year"),
        ("MoM generation demo", "Exact keyword — only T2 mentions MoM", "MoM"),
        ("What should we demonstrate?", "Pure semantic — both transcripts relevant", None),
    ]

    retriever = HybridRetriever(
        session_id=session_id,
        doc_type="transcript",
        recall_k=10,
        final_k=2,
    )

    for query, desc, expected in queries:
        print(f"\n{'─' * 65}")
        print(f"  Query: \"{query}\"")
        print(f"  ({desc})")
        print(f"{'─' * 65}")

        # Vector-only
        vec_results = query_knowledge_base(query, session_id, "transcript", limit=2, enable_reranking=False)
        print(f"\n  Vector-only:")
        for i, r in enumerate(vec_results):
            label = labels[0] if "Jira" in r["content"] or "Slack" in r["content"] else labels[1]
            print(f"    #{i+1} [{label}] cosine={r['score']:.4f}  \"{r['content'][:50]}...\"")

        # Hybrid
        hybrid_results = retriever.retrieve(query, expand=False)
        print(f"\n  Hybrid (BM25 + RRF + Rerank):")
        for i, r in enumerate(hybrid_results):
            label = labels[0] if "Jira" in r["content"] or "Slack" in r["content"] else labels[1]
            rrf = r.get("rrf_score", 0)
            rerank = r.get("rerank_score", 0)
            print(f"    #{i+1} [{label}] rrf={rrf:.4f}  rerank={rerank:.4f}  \"{r['content'][:50]}...\"")

        if expected and hybrid_results:
            if expected in hybrid_results[0]["content"]:
                print(f"\n  ✅ Correct: Top result contains \"{expected}\"")
            else:
                print(f"\n  ⚠️  Top result missing \"{expected}\"")

    print(f"\n{'=' * 65}")
    print("  Done.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run_test()
