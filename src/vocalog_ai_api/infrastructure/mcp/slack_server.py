import os
from mcp.server.fastmcp import FastMCP
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Initialize FastMCP server
mcp = FastMCP("Slack Integration")

def get_slack_client() -> WebClient:
    """Helper to initialize the Slack client with the bot token."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise ValueError("SLACK_BOT_TOKEN environment variable is not set.")
    return WebClient(token=token)

@mcp.tool()
def send_slack_message(channel_id: str, text: str) -> str:
    """
    Sends a message to a specific Slack channel or user ID.
    
    Args:
        channel_id: The ID of the channel (e.g., C12345) or user (e.g., U12345).
        text: The markdown formatted text message to send.
    """
    try:
        client = get_slack_client()
        response = client.chat_postMessage(
            channel=channel_id,
            text=text,
            mrkdwn=True
        )
        return f"Successfully posted message to {channel_id}. Message TS: {response['ts']}"
    except SlackApiError as e:
        error_msg = f"Slack API Error: {e.response['error']}"
        print(error_msg)
        return error_msg
    except Exception as e:
        return f"Failed to send message: {str(e)}"

if __name__ == "__main__":
    # Start the server using standard input/output when run directly
    mcp.run(transport="stdio")
