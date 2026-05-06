from pydantic import BaseModel, Field
from typing import Optional, Literal

class ActionItem(BaseModel):
    assignee: str = Field(description="The name or identifier of the person assigned to the task.")
    task_description: str = Field(description="A clear description of the action item or task to be completed.")
    due_date: Optional[str] = Field(None, description="The deadline for the task if explicitly mentioned, e.g. 'Friday', '2026-05-10', 'end of sprint'.")
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Urgency of the task. 'high' = blocking or time-critical, 'medium' = important but not urgent, 'low' = nice-to-have or long-term.",
    )
    target_platform: Literal["slack", "github", "gmail", "unknown"] = Field(
        default="unknown",
        description="Classification of the intended platform. Routing is handled by the backend service."
    )

class ActionExtractionResult(BaseModel):
    actions: list[ActionItem] = Field(default_factory=list, description="List of extracted action items.")
