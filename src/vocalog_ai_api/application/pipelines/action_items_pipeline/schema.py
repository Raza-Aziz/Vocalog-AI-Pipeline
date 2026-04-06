from pydantic import BaseModel, Field
from typing import Optional, Literal

class ActionItem(BaseModel):
    assignee: str = Field(description="The name or identifier of the person assigned to the task.")
    task_description: str = Field(description="A clear description of the action item or task to be completed.")
    due_date: Optional[str] = Field(None, description="The deadline for the task, if mentioned.")
    target_platform: Literal["slack", "github", "gmail", "unknown"] = Field(
        default="slack", 
        description="The platform where this action should be routed. Default to slack if not specified."
    )

class ActionExtractionResult(BaseModel):
    actions: list[ActionItem] = Field(default_factory=list, description="List of extracted action items.")
