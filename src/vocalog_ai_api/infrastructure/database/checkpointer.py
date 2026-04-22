"""
LangGraph SQLite checkpointer for Vocalog.

Provides `get_checkpointer()` which returns a SqliteSaver bound to the
local vocalog_local.db.  All LangGraph graph state (MoM, action items,
doc generation) is persisted here, keyed by thread_id.

Thread ID convention (enforce at call-site):
    "{user_id}:{meeting_id}"   — for meeting-bound pipelines
    "{user_id}:{session_uuid}" — for ad-hoc sessions

The checkpointer tables (checkpoints, checkpoint_blobs, checkpoint_writes,
checkpoint_migrations) are managed automatically by LangGraph and do NOT
clash with the Vocalog business tables in the same DB file.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from vocalog_ai_api.infrastructure.database.connection import get_conn

def get_checkpointer() -> SqliteSaver:
    """
    Return a SqliteSaver using the shared application connection.
    """
    return SqliteSaver(get_conn())


# Module-level singleton — created once when the module is first imported.
# Graphs that import this at module level (graph.py files) will share it.
checkpointer: SqliteSaver = get_checkpointer()
