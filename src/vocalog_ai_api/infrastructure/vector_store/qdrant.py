import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL_ENDPOINT", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKING_MODEL_NAME = os.getenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Reranking Configuration
ENABLE_RERANKING = True
RECALL_K = 20  # How many to fetch from Qdrant
RERANK_K = 5   # How many to keep after reranking

# Unified Collection Name
VOCALOG_MAIN_COLLECTION = "vocalog_main"
VECTOR_SIZE = 384  # Dimension for all-MiniLM-L6-v2

# --- Type Definitions ---
class VectorPayload(TypedDict):
    """Schema Contract for Vocalog Knowledge Base"""
    session_id: str         # UUID: The primary grouping key (tenant)
    meeting_id: str         # UUID: Could be same as session_id or differ if multiple sessions map to one meeting
    doc_type: str           # "transcript", "mom", "srs_section", "feedback"
    chunk_type: str         # "speaker_turn", "topic", "section"
    content: str            # The actual text content
    section_id: Optional[str]   # Logical section name (e.g., "Architecture", "Discussion")
    speakers: List[str]         # List of speakers found in this chunk
    timestamp: Optional[float]  # For audio alignment
    chunk_index: int            # For ordering

# --- Singleton Client ---
# Using a global variable for the client to ensure singleton pattern across imports
_client_instance: Optional[QdrantClient] = None
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None
_reranker_instance: Optional[CrossEncoder] = None

def get_qdrant_client() -> QdrantClient:
    global _client_instance
    if _client_instance is None:
        if QDRANT_API_KEY:
            _client_instance = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client_instance = QdrantClient(url=QDRANT_URL)
    return _client_instance

def get_embeddings_model() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_instance

def get_reranker_model() -> CrossEncoder:
    """Singleton for the CrossEncoder model (approx 300MB RAM)."""
    global _reranker_instance
    if _reranker_instance is None:
        print(f"Loading Reranker Model: {RERANKING_MODEL_NAME}")
        _reranker_instance = CrossEncoder(RERANKING_MODEL_NAME)
    return _reranker_instance


# --- Core Operations ---

def ensure_collection_exists():
    """
    Ensures the 'vocalog_main' collection exists with proper configuration.
    Creates Payload Indexes for high-performance filtering.
    """
    client = get_qdrant_client()
    
    if not client.collection_exists(VOCALOG_MAIN_COLLECTION):
        print(f"Creating Unified Collection: {VOCALOG_MAIN_COLLECTION}")
        client.create_collection(
            collection_name=VOCALOG_MAIN_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

    # Use session_id as the tenant key for physical grouping/sharding optimization
    # (Note: In simple local Qdrant, this just ensures an index exists)
    _create_index_if_not_exists(client, "session_id", models.PayloadSchemaType.KEYWORD)
    _create_index_if_not_exists(client, "doc_type", models.PayloadSchemaType.KEYWORD)
    _create_index_if_not_exists(client, "chunk_type", models.PayloadSchemaType.KEYWORD)
    _create_index_if_not_exists(client, "speakers", models.PayloadSchemaType.KEYWORD)


def _create_index_if_not_exists(client: QdrantClient, field_name: str, schema_type: models.PayloadSchemaType):
    """Helper to safely create payload indexes."""
    try:
        client.create_payload_index(
            collection_name=VOCALOG_MAIN_COLLECTION,
            field_name=field_name,
            field_schema=schema_type
        )
    except Exception as e:
        # Ignore if index already exists (Qdrant might raise 409 or similar)
        # Printing for debug but not raising
        pass


def session_vectors_exist(session_id: str, doc_type: Optional[str] = None) -> bool:
    """
    Returns True if at least one vector already exists for the given session_id
    (and optionally doc_type). Used for skip-if-exists idempotency checks.
    """
    client = get_qdrant_client()
    ensure_collection_exists()

    must_filters = [
        models.FieldCondition(
            key="session_id",
            match=models.MatchValue(value=session_id)
        )
    ]

    if doc_type:
        must_filters.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type)
            )
        )

    result = client.count(
        collection_name=VOCALOG_MAIN_COLLECTION,
        count_filter=models.Filter(must=must_filters),
        exact=False,  # approximate is fast and sufficient for existence checks
    )
    return result.count > 0


def delete_session_vectors(session_id: str, doc_type: Optional[str] = None):
    """
    Idempotency: Clears existing vectors for a session (and optionally a specific doc_type)
    BEFORE ingesting new ones. This prevents duplication.
    """
    client = get_qdrant_client()
    ensure_collection_exists() # Good practice to check existence
    
    must_filters = [
        models.FieldCondition(
            key="session_id",
            match=models.MatchValue(value=session_id)
        )
    ]
    
    if doc_type:
        must_filters.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type)
            )
        )
        
    print(f"Clearing old vectors for session={session_id}, doc_type={doc_type or 'ALL'}")
    client.delete(
        collection_name=VOCALOG_MAIN_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=must_filters)
        )
    )


def upsert_vectors(points: List[models.PointStruct]):
    """
    Centralized upsert method.
    """
    client = get_qdrant_client()
    client.upsert(
        collection_name=VOCALOG_MAIN_COLLECTION,
        points=points
    )


def embed_text(text: str) -> List[float]:
    """
    Centralized embedding generation.
    """
    model = get_embeddings_model()
    return model.embed_query(text)

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Batch embedding generation.
    """
    model = get_embeddings_model()
    return model.embed_documents(texts)


def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Re-scores a list of documents against the query using CrossEncoder.
    Returns the top-k documents sorted by relevance score.
    """
    if not documents:
        return []

    reranker = get_reranker_model()
    
    # CrossEncoder expects pairs of [Query, Document Text]
    pairs = [[query, doc.get("content", "")] for doc in documents]
    
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        print(f"Reranking Failed: {e}. Returning original order.")
        return documents[:top_k]

    # Attach scores to documents
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    # Sort by score descending
    # Higher score = more relevant for CrossEncoder (usually logits or sigmoid)
    sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    
    return sorted_docs[:top_k]


# --- Retrieval ---

def query_knowledge_base(
    query_text: str,
    session_id: str,
    doc_type: Optional[str] = None,
    limit: int = 5,
    enable_reranking: bool = False
) -> List[Dict[str, Any]]:
    """
    Unified retrieval function with optional 2-stage reranking.
    """
    client = get_qdrant_client()
    ensure_collection_exists() # JIT check
    
    query_vector = embed_text(query_text)
    
    must_filters = [
        models.FieldCondition(
            key="session_id",
            match=models.MatchValue(value=session_id)
        )
    ]
    
    if doc_type:
        must_filters.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type)
            )
        )
    
    # Stage 1: Recall
    # If reranking is ON, we fetch MORE candidates (RECALL_K)
    # If OFF, we just fetch the user requested limit
    fetch_limit = RECALL_K if enable_reranking else limit
        
    results = client.query_points(
        collection_name=VOCALOG_MAIN_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(must=must_filters),
        limit=fetch_limit
    ).points
    
    # Convert Qdrant PointStructs to simple dicts
    candidate_docs = [
        {
            "content": r.payload.get("content", ""),
            "metadata": r.payload,
            "score": r.score, # Vector similarity score
            "id": r.id
        }
        for r in results
    ]
    
    # Stage 2: Rerank (Optional)
    if enable_reranking and candidate_docs:
        final_docs = rerank_documents(query_text, candidate_docs, top_k=limit) # Return only requested 'limit'
    else:
        final_docs = candidate_docs[:limit]
    
    return final_docs

