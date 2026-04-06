import sys
import os
import asyncio
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vocalog_ai_api.api.main import app

client = TestClient(app)

def test_slack_mcp_routing():
    print("="*60)
    print("API: MCP Slack Execution Validation")
    print("="*60)

    # Note: For this test to ACTUALLY post to Slack, 
    # SLACK_BOT_TOKEN and SLACK_TEAM_ID must be set in your environment.
    # Otherwise, the MCP server will start but throw an auth error,
    # which we catch gracefully in the action_executor.
    
    payload = {
        "channel_id": "general", 
        "actions": [
            {
                "assignee": "Raza",
                "task_description": "Review the newly created MCP client connection.",
                "due_date": "End of Day",
                "target_platform": "slack"
            },
            {
                "assignee": "Hamza",
                "task_description": "Draft the final Demo script.",
                "due_date": "Tomorrow",
                "target_platform": "slack"
            }
        ]
    }
    
    print("\n[Step 1] Sending execution request to /action-items/execute/slack...")
    response = client.post("/action-items/execute/slack", json=payload)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    print("\nAPI Response:")
    print(f"Message: {data.get('message')}")
    
    exec_result = data.get("execution_result", {})
    print(f"Status: {exec_result.get('status')}")
    
    if exec_result.get('status') == "completed":
        print(f"Successfully processed {exec_result.get('processed_count')} slack actions via MCP.")
        for res in exec_result.get('results', []):
            print(f"  - [{res.get('status')}] {res.get('task')}")
    else:
        print(f"\nExecution skipped or errored (usually due to missing SLACK environ keys):")
        print(f"Details: {exec_result.get('message')}")
        
    print("\n[SUCCESS]: MCP Routing API endpoint runs without crashing.")
    print("="*60)

if __name__ == "__main__":
    test_slack_mcp_routing()
