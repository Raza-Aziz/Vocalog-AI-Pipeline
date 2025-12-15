from pydantic import BaseModel
from typing import List, Dict, Optional

# schemas/graph_state.py
class GraphState(BaseModel):
    document_id: str
    project_id: str
    user_id: str

    sections: List[str]
    current_section_index: int

    section_drafts: Dict[str, str]
    finalized_sections: Dict[str, str]

    hitl_feedback: Optional[Dict] = None
