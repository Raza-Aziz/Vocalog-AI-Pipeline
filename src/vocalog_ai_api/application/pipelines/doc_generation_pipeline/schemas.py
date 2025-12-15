# schemas/document.py
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict
from datetime import datetime


class CreateDocumentRequest(BaseModel):
    project_id: str
    user_id: str
    doc_type: Literal["srs", "mom", "tender", "boq"]
    template_id: Optional[str] = None


class CreateDocumentResponse(BaseModel):
    document_id: str
    status: str


class SectionResponse(BaseModel):
    section_id: str
    title: str
    draft_text: str
    revision: int
    status: str

# schemas/hitl.py
class HITLFeedbackRequest(BaseModel):
    document_id: str
    section_id: str
    user_id: str
    accept: bool
    edits: Optional[str] = None
    comment: Optional[str] = None
    revision: int
    action: Literal["accept", "edit", "refine"]
