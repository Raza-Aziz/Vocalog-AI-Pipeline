from typing import TypedDict, Optional, List
from vocalog_ai_api.application.pipelines.action_items_pipeline.schema import ActionItem

class ActionItemsState(TypedDict):
    session_id: str
    transcript: str
    extracted_actions: Optional[List[ActionItem]]
    # Store results from MCP after execution
    execution_results: Optional[List[dict]]
