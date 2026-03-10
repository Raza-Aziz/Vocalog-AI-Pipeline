import sys
import os
import asyncio
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Force MOCK_MODE for nodes to speed up API tests if needed, 
# but we want to test the logic in main.py
from vocalog_ai_api.api.main import app

client = TestClient(app)

def test_api_refinement_flow():
    print("="*60)
    print("API: HITL Refinement Validation")
    print("="*60)

    # 1. Start Generation
    payload = {
        "meeting_minutes": "Raza: Let's use Qdrant. Hamza: And FastAPI.",
        "project_id": "api-test"
    }
    response = client.post("/generate-document", json=payload)
    assert response.status_code == 200
    data = response.json()
    doc_id = data["document_id"]
    print(f"\n[Step 1] Document Created: {doc_id}")
    print(f"Initial Content: {data['content'][:50]}...")

    # 2. Provide Refinement Feedback
    feedback_payload = {
        "document_id": doc_id,
        "action": "refine",
        "feedback_notes": "Make it very short."
    }
    print(f"\n[Step 2] Sending 'refine' action...")
    response = client.post("/provide-feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()
    
    print(f"Refined Content Preview: {data['content'][:50]}...")
    print(f"Refinement Count: {data['refinement_count']}")
    
    assert data["refinement_count"] == 1
    # Check if content changed from initial (though it could theoretically be similar, usually it changes)
    # We'll just verify refinement_count is 1 as that proves the hitl loop worked in main.py
    print("✅ Success: Refinement count incremented.")

    # 3. Approve
    approve_payload = {
        "document_id": doc_id,
        "action": "approve"
    }
    print(f"\n[Step 3] Sending 'approve' action...")
    response = client.post("/provide-feedback", json=approve_payload)
    assert response.status_code == 200
    data = response.json()
    
    print(f"Next Section Title: {data['section_title']}")
    print(f"New Refinement Count (should be 0): {data['refinement_count']}")
    
    assert data["refinement_count"] == 0
    print("✅ Success: Transitioned to next section and reset refinement count.")

    print("="*60)
    print("API Validation Complete.")
    print("="*60)

if __name__ == "__main__":
    test_api_refinement_flow()
