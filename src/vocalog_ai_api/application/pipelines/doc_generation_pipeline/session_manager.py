import uuid
from typing import Dict, Optional
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState

class SessionManager:
    def __init__(self):
        # In-memory store: {document_id: State}
        self._store: Dict[str, DocumentGenerationState] = {}

    def create_session(self, meeting_minutes: str, project_id: str) -> str:
        doc_id = str(uuid.uuid4())
        
        # Initialize default state
        initial_state: DocumentGenerationState = {
            "session_id": doc_id,
            "project_id": project_id,
            "meeting_minutes": meeting_minutes,
            "sections_outline": [],
            "current_section_index": 0,
            "current_section_content": "",
            "feedback_action": None,
            "feedback_notes": None,
            "final_document": [],
            "is_complete": False
        }
        
        self._store[doc_id] = initial_state
        return doc_id

    def get_session(self, doc_id: str) -> Optional[DocumentGenerationState]:
        return self._store.get(doc_id)

    def update_session(self, doc_id: str, new_state: DocumentGenerationState):
        if doc_id in self._store:
            self._store[doc_id] = new_state

    def delete_session(self, doc_id: str):
        if doc_id in self._store:
            del self._store[doc_id]

# Singleton instance for the app
session_manager = SessionManager()