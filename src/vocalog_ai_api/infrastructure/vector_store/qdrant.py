import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL_ENDPOINT", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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
    speaker: Optional[str]      # Speaker name if available
    timestamp: Optional[float]  # For audio alignment
    chunk_index: int            # For ordering

# --- Singleton Client ---
# Using a global variable for the client to ensure singleton pattern across imports
_client_instance: Optional[QdrantClient] = None
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None

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


# --- Retrieval ---

def query_knowledge_base(
    query_text: str,
    session_id: str,
    doc_type: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Unified retrieval function.
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
        
    results = client.query_points(
        collection_name=VOCALOG_MAIN_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(must=must_filters),
        limit=limit
    ).points
    
    return [
        {
            "content": r.payload.get("content", ""),
            "metadata": r.payload,
            "score": r.score
        }
        for r in results
    ]
