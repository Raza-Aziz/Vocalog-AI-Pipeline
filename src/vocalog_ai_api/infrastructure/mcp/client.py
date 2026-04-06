import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@asynccontextmanager
async def get_slack_mcp_session() -> AsyncGenerator[ClientSession, None]:
    """
    Creates an MCP Stdio connection to the @zencoderai/slack-mcp-server.
    Requires SLACK_BOT_TOKEN (and potentially SLACK_TEAM_ID) in the environment.
    """
    
    # Ensure env variables are passed to the subprocess
    env = os.environ.copy()
    
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "slack_server.py"))
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", server_path],
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP connection
            await session.initialize()
            yield session
