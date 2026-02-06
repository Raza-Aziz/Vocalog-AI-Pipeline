import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load env from the same location as the app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
load_dotenv()

def check_connection():
    url = os.getenv("QDRANT_URL_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"--- Qdrant Connection Check ---")
    print(f"URL: {url}")
    if api_key:
        print(f"API Key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 5 else ''}")
    else:
        print("API Key: Not set")
        
    if not url:
        print("ERROR: QDRANT_URL_ENDPOINT is not set.")
        return

    try:
        print("Attempting to connect...")
        client = QdrantClient(url=url, api_key=api_key)
        
        # Simple health check involves fetching collections
        cols = client.get_collections()
        print("SUCCESS: Connected to Qdrant.")
        print(f"Collections found: {[c.name for c in cols.collections]}")
        
    except Exception as e:
        print(f"\nCONNECTION FAILED: {e}")
        print("\nPossible causes:")
        print("1. Qdrant Cloud cluster is paused (Free Tier pauses after inactivity).")
        print("2. URL is incorrect.")
        print("3. Network firewall/proxy issues.")

if __name__ == "__main__":
    check_connection()
