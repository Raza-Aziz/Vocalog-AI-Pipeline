import os
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL_ENDPOINT", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Collection Names
MEETINGS_COLLECTION = "meetings_vectors"
DOCUMENT_SECTIONS_COLLECTION = "document_sections_vectors"
VECTOR_SIZE = 384  # Dimension for all-MiniLM-L6-v2

# --- Initialization ---
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)

# Initialize Embedding Model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _ensure_collection_exists(collection_name: str):
    """Ensures the collection exists before attempting operations."""
    if not client.collection_exists(collection_name):
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

def _get_embedding(text: str) -> List[float]:
    """Generate embedding for text using HuggingFace model."""
    return embeddings.embed_query(text)


def retrieve_meeting_context(project_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant meeting minutes context using the new query_points API.
    """
    _ensure_collection_exists(MEETINGS_COLLECTION)
    query_vector = _get_embedding(query)
    
    search_filter = Filter(
        must=[
            FieldCondition(
                key="project_id",
                match=MatchValue(value=project_id)
            ),
            FieldCondition(
                key="document_type",
                match=MatchValue(value="meeting_minutes")
            )
        ]
    )
    
    # FIXED: Uses query_points instead of search
    results = client.query_points(
        collection_name=MEETINGS_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit
    ).points
    
    return [
        {
            "text": r.payload.get("text", ""),
            "meeting_id": r.payload.get("meeting_id"),
            "score": r.score
        }
        for r in results
    ]


def store_meeting_chunk(project_id: str, meeting_id: str, text: str):
    """
    Store a meeting minutes chunk in Qdrant.
    """
    _ensure_collection_exists(MEETINGS_COLLECTION)
    embedding = _get_embedding(text)
    
    # Generate unique ID based on content hash to prevent duplicates
    text_hash = hashlib.md5(text.encode()).hexdigest()
    point_id = str(hashlib.uuid.uuid5(hashlib.uuid.NAMESPACE_DNS, f"{meeting_id}-{text_hash}"))
    
    client.upsert(
        collection_name=MEETINGS_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "project_id": project_id,
                    "meeting_id": meeting_id,
                    "text": text,
                    "document_type": "meeting_minutes"
                }
            )
        ]
    )
    print(f"Stored chunk for meeting {meeting_id}")

# Legacy adapter to support calls from older nodes.py code if needed
def retrieve_context(session_id: str, query: str, limit: int = 3) -> str:
    """Adapter for backward compatibility with nodes.py"""
    # Assuming session_id acts as project_id for now
    results = retrieve_meeting_context(project_id=session_id, query=query, limit=limit)
    if not results:
        return "No relevant context found."
    return "\n\n---\n\n".join([r["text"] for r in results])