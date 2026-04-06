import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vocalog_ai_api.api.main import app

client = TestClient(app)

def test_action_extraction():
    print("="*60)
    print("API: Action Extraction Validation")
    print("="*60)

    # 1. Provide a dummy transcript with clear action items
    transcript = """
    Raza: Welcome to the weekly sync. Let's discuss the new features.
    Hamza: I've looked at the MCP integration. I will build the Slack client by tomorrow.
    Raza: Great. Saad, can you please review the GitHub PRs for the authentication module by Friday?
    Saad: Sure thing. Also, we need someone to draft the release email.
    Raza: I'll handle drafting the release email over the weekend, let's target Gmail for that.
    """

    payload = {
        "transcript": transcript
    }
    
    print("\n[Step 1] Sending extraction request...")
    response = client.post("/action-items/extract", json=payload)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Details: {response.text}"
    data = response.json()
    
    actions = data.get("actions", [])
    print(f"\nExtracted {len(actions)} actions:")
    for i, act in enumerate(actions, 1):
        print(f"  {i}. Assignee: {act['assignee']}")
        print(f"     Task: {act['task_description']}")
        print(f"     Due: {act['due_date']}")
        print(f"     Platform: {act['target_platform']}")
        print("-" * 30)
        
    assert len(actions) >= 3, "Expected at least 3 actions to be extracted."
    print("\n[SUCCESS]: Action extraction logic and endpoint are working correctly.")
    print("="*60)

if __name__ == "__main__":
    test_action_extraction()
