from typing import TypedDict, List, Dict, Any


class RetrievedChunk(TypedDict):
    content: str
    doc_type: str       # "transcript" | "mom"
    chunk_index: int
    speakers: List[str]
    score: float


class MeetingQAState(TypedDict):
    meeting_id: str
    question: str
    transcript_chunks: List[RetrievedChunk]
    mom_chunks: List[RetrievedChunk]
    answer: str
    citations: List[Dict[str, Any]]
