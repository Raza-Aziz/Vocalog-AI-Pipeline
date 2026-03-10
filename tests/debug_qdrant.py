import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL_ENDPOINT")
key = os.getenv("QDRANT_API_KEY")

print(f"Connecting to: {url}")
client = QdrantClient(url=url, api_key=key)

try:
    collections = client.get_collections()
    print("Successfully connected!")
    print(f"Collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"Connection failed: {e}")
