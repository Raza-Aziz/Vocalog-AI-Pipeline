from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field

class MeetingInfo(BaseModel):
  title: str = Field(description='Generate a descriptive title/topic for the meeting based entirely on the context and transcript, as normal meeting info might be insufficient.')
  date: str = Field(description='Date of the meeting, in YYYY-MM-DD')
  time: str = Field(description='Meeting start and end time')
  venue_or_platform: str = Field(description="Meeting location (Physical) or online platform")

class Attendees(BaseModel):
    present: List[str] = Field(..., description="List of attendees present")
    absent: Optional[List[str]] = Field(default_factory=list, description="List of attendees absent")

class DiscussionSummaryItem(BaseModel):
    agenda_item: str = Field(..., description="Agenda topic")
    summary: str = Field(..., description="Concise discussion summary for this agenda item")

class ActionItem(BaseModel):
    task: str = Field(..., description="Task description")
    assignee: str = Field(..., description="Person responsible for the task")
    deadline: str = Field(..., description="Task deadline (YYYY-MM-DD or natural text)")

class MinutesOfMeeting(BaseModel):
  meeting_info: MeetingInfo
  attendees: Attendees
  agenda: List[str]
  discussion_summary: List[DiscussionSummaryItem]
  action_items: List[ActionItem]
