"""
Module 5: Hybrid Retriever
Combines Vector Search (Qdrant) + Keyword Search (BM25) via Reciprocal Rank Fusion.
Optionally expands queries using LLM-generated probes for better recall.
"""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from vocalog_ai_api.infrastructure.vector_store.qdrant import (
    query_knowledge_base,
    rerank_documents,
)


class HybridRetriever:
    """
    Orchestrates: Vector Search → BM25 → Fusion → Rerank.
    
    Sits ON TOP of the existing infrastructure. Does NOT modify qdrant.py or ingestion.
    BM25 runs in-memory over the candidate set returned by vector search.
    
    Args:
        session_id: The session to scope retrieval to.
        doc_type: Optional filter (e.g., "transcript").
        vector_weight: RRF weight for vector results (default 0.6).
        bm25_weight: RRF weight for BM25 results (default 0.4).
        recall_k: How many candidates to fetch from Qdrant for BM25 scoring (default 20).
        final_k: How many results to return after fusion + rerank (default 5).
    """

    def __init__(
        self,
        session_id: str,
        doc_type: Optional[str] = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        recall_k: int = 20,
        final_k: int = 5,
    ):
        self.session_id = session_id
        self.doc_type = doc_type
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.recall_k = recall_k
        self.final_k = final_k

    # ------------------------------------------------------------------ #
    # Stage 1: Vector Search (delegates to existing infra)
    # ------------------------------------------------------------------ #
    def _vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Calls the existing query_knowledge_base for dense vector retrieval.
        Returns list of dicts with 'content', 'metadata', 'score', 'id'.
        """
        return query_knowledge_base(
            query_text=query,
            session_id=self.session_id,
            doc_type=self.doc_type,
            limit=k,
            enable_reranking=False,  # We handle reranking ourselves after fusion
        )

    # ------------------------------------------------------------------ #
    # Stage 2: BM25 Keyword Search (in-memory over candidates)
    # ------------------------------------------------------------------ #
    def _bm25_search(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scores each candidate's content against the query using BM25.
        Returns the same candidates list sorted by BM25 score (descending),
        with 'bm25_score' attached to each doc.
        """
        if not candidates:
            return []

        # Tokenize: simple whitespace + lowercase
        corpus = [doc["content"].lower().split() for doc in candidates]
        query_tokens = query.lower().split()

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        # Attach scores
        for doc, score in zip(candidates, scores):
            doc["bm25_score"] = float(score)

        # Sort by BM25 score descending
        return sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)

    # ------------------------------------------------------------------ #
    # Stage 3: Reciprocal Rank Fusion (RRF)
    # ------------------------------------------------------------------ #
    def _fuse_rrf(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,  # RRF constant (standard default)
    ) -> List[Dict[str, Any]]:
        """
        Merges two ranked lists using Reciprocal Rank Fusion.
        
        RRF Score = Σ (weight / (k + rank))
        where rank is 1-indexed position in each list.
        
        Returns deduplicated results sorted by fused score.
        """
        # Build a map: doc_id -> {doc, rrf_score}
        fused: Dict[str, Dict[str, Any]] = {}

        # Score from vector results
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = str(doc["id"])
            if doc_id not in fused:
                fused[doc_id] = {**doc, "rrf_score": 0.0}
            fused[doc_id]["rrf_score"] += self.vector_weight / (k + rank)

        # Score from BM25 results
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = str(doc["id"])
            if doc_id not in fused:
                fused[doc_id] = {**doc, "rrf_score": 0.0}
            fused[doc_id]["rrf_score"] += self.bm25_weight / (k + rank)

        # Sort by fused score descending
        sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results

    # ------------------------------------------------------------------ #
    # Stage 4 (5.2): Query Expansion — LLM Probes
    # ------------------------------------------------------------------ #
    def _expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Generates query variants for multi-probe retrieval.
        
        Strategy: Domain-specific keyword augmentation (no LLM call needed for now).
        This keeps it fast and deterministic. Can be upgraded to LLM-based later.
        
        Returns a list of expanded queries (original query is always first).
        """
        expansions = [query]  # Always include the original

        # Domain-specific expansions for meeting/SRS context
        domain_keywords = {
            "requirements": ["specification", "feature", "user story", "use case"],
            "architecture": ["design", "system structure", "component", "module"],
            "timeline": ["schedule", "deadline", "milestone", "sprint"],
            "integration": ["API", "connector", "webhook", "plugin"],
            "feedback": ["review", "comment", "suggestion", "improvement"],
            "action item": ["task", "todo", "follow-up", "assignment"],
            "discussion": ["meeting", "conversation", "decision", "agreement"],
        }

        query_lower = query.lower()

        # Check if any domain keyword group matches
        for trigger, synonyms in domain_keywords.items():
            if trigger in query_lower:
                # Add 1-2 variants using synonyms
                for synonym in synonyms[:2]:
                    variant = query_lower.replace(trigger, synonym)
                    if variant != query_lower:
                        expansions.append(variant)

        # Cap at max_expansions
        return expansions[:max_expansions + 1]  # +1 because original is always included

    # ------------------------------------------------------------------ #
    # Main Entry Point
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, expand: bool = False) -> List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline:
        1. (Optional) Expand query into multiple probes
        2. For each probe: Vector Search → BM25 Score → RRF Fusion
        3. Merge all probes (deduplicate by ID, keep max RRF score)
        4. Rerank final candidates with CrossEncoder
        
        Args:
            query: The search query.
            expand: If True, generate query variants for multi-probe retrieval.
            
        Returns:
            List of documents sorted by relevance.
        """
        queries = self._expand_query(query) if expand else [query]

        # Accumulate all candidates across probes
        all_fused: Dict[str, Dict[str, Any]] = {}

        for probe_query in queries:
            # Step 1: Vector search
            vector_results = self._vector_search(probe_query, k=self.recall_k)

            if not vector_results:
                continue

            # Step 2: BM25 over the same candidates
            # We make a copy so BM25 scoring doesn't mutate the vector_results order
            bm25_candidates = [dict(doc) for doc in vector_results]
            bm25_results = self._bm25_search(probe_query, bm25_candidates)

            # Step 3: Fuse
            fused_results = self._fuse_rrf(vector_results, bm25_results)

            # Merge across probes: keep the max RRF score per doc
            for doc in fused_results:
                doc_id = str(doc["id"])
                if doc_id not in all_fused or doc["rrf_score"] > all_fused[doc_id]["rrf_score"]:
                    all_fused[doc_id] = doc

        # Collect and sort
        merged = sorted(all_fused.values(), key=lambda x: x["rrf_score"], reverse=True)

        if not merged:
            return []

        # Step 4: Rerank the top candidates
        rerank_candidates = merged[: self.recall_k]  # Don't rerank more than recall_k
        final_docs = rerank_documents(query, rerank_candidates, top_k=self.final_k)

        return final_docs
