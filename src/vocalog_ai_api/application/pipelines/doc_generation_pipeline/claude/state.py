"""
State definition for document generation pipeline
"""
from typing import TypedDict, Dict, List, Optional
from enum import Enum


class SectionType(str, Enum):
    """Available SRS document sections"""
    INTRODUCTION = "introduction"
    SYSTEM_OVERVIEW = "system_overview"
    FUNCTIONAL_REQUIREMENTS = "functional_requirements"
    NON_FUNCTIONAL_REQUIREMENTS = "non_functional_requirements"
    USER_REQUIREMENTS = "user_requirements"
    SYSTEM_FEATURES = "system_features"


class DocumentState(TypedDict):
    """State for document generation workflow"""
    session_id: str
    project_name: str
    meeting_minutes: str
    document_type: str
    current_section: SectionType
    section_content: str
    approved_sections: Dict[str, str]
    revision_count: int
    user_feedback: Optional[str]
    all_sections: List[SectionType]
    section_index: int
    error: Optional[str]