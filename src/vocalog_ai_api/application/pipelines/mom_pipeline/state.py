from typing import TypedDict, Optional
from vocalog_ai_api.application.pipelines.mom_pipeline.schema import MinutesOfMeeting

# ----State----
class MoMGraphState(TypedDict):
    raw_transcript: dict
    clean_transcript: Optional[str]
    meeting_context: Optional[dict]
    mom: Optional[MinutesOfMeeting]
    mom_markdown: Optional[str]