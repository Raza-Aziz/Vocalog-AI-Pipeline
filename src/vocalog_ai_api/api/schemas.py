from pydantic import BaseModel

class TranscriptInput(BaseModel):
    raw_transcript: dict

class MoMResponse(BaseModel):
    markdown: str