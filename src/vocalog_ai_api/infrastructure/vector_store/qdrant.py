from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)

# Initialize embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Collection names
MEETINGS_COLLECTION = "meetings_vectors"
DOCUMENT_SECTIONS_COLLECTION = "document_sections_vectors"


def _get_embedding(text: str) -> list:
    """Generate embedding for text using HuggingFace model."""
    return embeddings.embed_query(text)


def retrieve_meeting_context(project_id: str, query: str, limit: int = 5) -> list:
    """
    Retrieve relevant meeting minutes context from Qdrant for a project.
    
    Args:
        project_id: The project identifier
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of context dictionaries with text and meeting_id
    """
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
    
    results = client.search(
        collection_name=MEETINGS_COLLECTION,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit
    )
    
    return [
        {
            "text": r.payload.get("text", ""),
            "meeting_id": r.payload.get("meeting_id"),
            "score": r.score
        }
        for r in results
    ]


def retrieve_section_context(project_id: str, document_id: str, query: str, limit: int = 5) -> list:
    """
    Retrieve previously approved document sections from Qdrant for context.
    
    Args:
        project_id: The project identifier
        document_id: The document identifier
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of context dictionaries with text and section_id
    """
    query_vector = _get_embedding(query)
    
    search_filter = Filter(
        must=[
            FieldCondition(
                key="project_id",
                match=MatchValue(value=project_id)
            ),
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id)
            ),
            FieldCondition(
                key="document_type",
                match=MatchValue(value="document_section")
            )
        ]
    )
    
    results = client.search(
        collection_name=DOCUMENT_SECTIONS_COLLECTION,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit
    )
    
    return [
        {
            "text": r.payload.get("text", ""),
            "section_id": r.payload.get("section_id"),
            "score": r.score
        }
        for r in results
    ]


def store_meeting_chunk(
    project_id: str,
    meeting_id: str,
    text: str,
    embedding: list = None
):
    """
    Store a meeting minutes chunk in Qdrant.
    
    Args:
        project_id: The project identifier
        meeting_id: The meeting identifier
        text: The text content to store
        embedding: Optional pre-computed embedding (if None, will compute)
    """
    if embedding is None:
        embedding = _get_embedding(text)
    
    # Generate unique ID
    text_hash = hashlib.md5(text.encode()).hexdigest()
    point_id = f"{meeting_id}-{text_hash}"
    
    client.upsert(
        collection_name=MEETINGS_COLLECTION,
        points=[
            {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "project_id": project_id,
                    "meeting_id": meeting_id,
                    "text": text,
                    "document_type": "meeting_minutes"
                }
            }
        ]
    )


def store_document_section(
    project_id: str,
    document_id: str,
    section_id: str,
    text: str,
    embedding: list = None
):
    """
    Store an approved document section in Qdrant.
    
    Args:
        project_id: The project identifier
        document_id: The document identifier
        section_id: The section identifier
        text: The section text content
        embedding: Optional pre-computed embedding (if None, will compute)
    """
    if embedding is None:
        embedding = _get_embedding(text)
    
    # Generate unique ID
    text_hash = hashlib.md5(text.encode()).hexdigest()
    point_id = f"{document_id}-{section_id}-{text_hash}"
    
    client.upsert(
        collection_name=DOCUMENT_SECTIONS_COLLECTION,
        points=[
            {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "project_id": project_id,
                    "document_id": document_id,
                    "section_id": section_id,
                    "text": text,
                    "document_type": "document_section"
                }
            }
        ]
    )


# Legacy function for backward compatibility
def retrieve_project_context(project_id: str, query: str, limit: int = 5) -> list:
    """Legacy function - retrieves meeting context."""
    return retrieve_meeting_context(project_id, query, limit)
