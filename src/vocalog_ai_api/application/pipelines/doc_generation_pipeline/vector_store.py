import os
import uuid
from typing import List
from dotenv import load_dotenv

# Qdrant & Sentence Transformers imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 

# Environment variables for Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL_ENDPOINT")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_KEY:
    raise ValueError("QDRANT_URL_ENDPOINT or QDRANT_API_KEY is missing from .env")

# Initialize Client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_KEY
)

# Initialize Embedding Model
try:
    embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")

def ensure_collection_exists():
    """
    Checks if collection exists, creates it if not.
    CRITICAL: Ensures a Payload Index exists for 'session_id' to allow filtering.
    """
    # 1. Create Collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            )
        )
    
    # 2. Create Payload Index (FIXES THE 400 ERROR)
    # We try to create this every time. Qdrant is smart enough to ignore it if it already exists.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="session_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

def ingest_minutes(session_id: str, text: str):
    """
    Chunks, vectorizes, and stores in Qdrant with 'session_id' payload.
    """
    ensure_collection_exists()

    # 1. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        print("Warning: No chunks created from text.")
        return

    # 2. Create Embeddings
    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL_NAME}...")
    vectors = embeddings_model.encode(chunks, convert_to_numpy=True).tolist() 

    # 3. Prepare Points
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "session_id": session_id,
                "content": chunk,
                "chunk_index": i
            }
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    # 4. Upload to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"--- Vector Store: Indexed {len(points)} chunks for session {session_id} ---")

def retrieve_context(session_id: str, query: str, limit: int = 3) -> str:
    """
    Searches for text relevant to 'query' within the specific 'session_id'.
    """
    # Ensure collection and INDEX exist before searching
    ensure_collection_exists()

    # 1. Embed the query
    query_vector = embeddings_model.encode(query, convert_to_numpy=True).tolist()

    # 2. Search (Using query_points)
    # The Payload Index created above allows this filter to work
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            ]
        ),
        limit=limit
    ).points

    # 3. Format results
    if not search_result:
        return "No relevant context found in meeting minutes."

    context_blocks = [hit.payload["content"] for hit in search_result]
    return "\n\n---\n\n".join(context_blocks)