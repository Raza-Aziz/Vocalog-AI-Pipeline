import asyncio
from typing import List
from vocalog_ai_api.application.pipelines.action_items_pipeline.schema import ActionItem
from vocalog_ai_api.infrastructure.mcp.client import get_slack_mcp_session

async def execute_slack_actions(actions: List[ActionItem], channel_id: str) -> dict:
    """
    Routes extracted action items to Slack via the MCP client.
    
    Args:
        actions: List of ActionItem objects to execute.
        channel_id: The specific Slack channel or user ID to post to.
        
    Returns:
        dict: Summary of the execution results.
    """
    
    # Filter for slack-targeted actions if preferred, though default is slack.
    slack_actions = [a for a in actions if a.target_platform == "slack"]
    
    if not slack_actions:
        return {"status": "skipped", "message": "No Slack-targeted actions found."}
        
    results = []
    
    # Connect to the Slack MCP Server
    try:
        async with get_slack_mcp_session() as session:
            for action in slack_actions:
                # Format the message
                message_text = (
                    f"*:bell: New Action Item Assigned*\n"
                    f"*Assignee*: {action.assignee}\n"
                    f"*Task*: {action.task_description}\n"
                    f"*Due Date*: {action.due_date if action.due_date else 'Not Specified'}"
                )
                
                print(f"Routing to Slack via MCP -> {action.task_description[:30]}...")
                
                # Call the custom MCP Tool
                try:
                    response = await session.call_tool(
                        name="send_slack_message",
                        arguments={
                            "channel_id": channel_id,
                            "text": message_text
                        }
                    )
                    results.append({"task": action.task_description, "status": "success", "response": str(response)})
                except Exception as e:
                    results.append({"task": action.task_description, "status": "failed", "error": str(e)})
                    print(f"Failed to post to Slack: {e}")
                    
    except Exception as e:
        # Catch broad connection errors (e.g. npx not installed, missing SLACK_APP_TOKEN)
        print(f"MCP Connection Error: {e}")
        return {"status": "error", "message": f"Could not connect to Slack MCP Server: {e}"}
        
    return {
        "status": "completed",
        "processed_count": len(slack_actions),
        "results": results
    }
