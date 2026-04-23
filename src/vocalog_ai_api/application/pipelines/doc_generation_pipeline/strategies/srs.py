from typing import List
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.base import DocumentStrategy


class SRSStrategy(DocumentStrategy):
    """
    Strategy for Software Requirements Specification documents.
    Persona: Expert Technical Writer focused on precise, testable requirements.
    """

    @property
    def document_type(self) -> str:
        return "srs"

    @property
    def display_name(self) -> str:
        return "Software Requirements Specification (SRS)"

    @property
    def sections(self) -> List[str]:
        return [
            "1. Introduction",
            "2. Overall Description",
            "3. System Features",
            "4. Functional Requirements",
            "5. Non-Functional Requirements",
            "6. External Interface Requirements",
            "7. Appendices",
        ]

    @property
    def persona(self) -> str:
        return (
            "an expert Technical Writer and Systems Analyst specialising in IEEE 830-style "
            "Software Requirements Specifications. Your output must be precise, unambiguous, "
            "and written so that developers can implement each requirement independently. "
            "Every functional requirement must be testable and uniquely identifiable (FR-XXX)."
        )

    @property
    def generation_instructions(self) -> str:
        return (
            "Instructions:\n"
            "1. Use clear, professional language free of ambiguity.\n"
            "2. Format all output in Markdown with appropriate heading levels (###).\n"
            "3. Number every functional requirement as FR-001, FR-002, …\n"
            "4. Number every non-functional requirement as NFR-P-001 (Performance), "
            "NFR-S-001 (Security), NFR-R-001 (Reliability), etc.\n"
            "5. Do NOT include the section heading itself — only the section body.\n"
            "6. If the provided context lacks sufficient information for a sub-section, "
            "write a clearly marked placeholder: '[TBD — awaiting stakeholder input]'.\n"
            "7. Do not introduce facts not present in the reference context."
        )

    @property
    def refinement_instructions(self) -> str:
        return (
            "Refinement Instructions:\n"
            "1. Address every point raised in the feedback without omitting existing content.\n"
            "2. Maintain IEEE 830 requirement numbering continuity.\n"
            "3. Keep Markdown formatting consistent with the rest of the document.\n"
            "4. Do NOT include the section heading — only the revised body.\n"
            "5. Do not introduce facts absent from the reference context."
        )

    def get_section_focus(self, section_title: str) -> str:
        focus_map = {
            "1. Introduction": (
                "Cover: purpose of this document, product scope, definitions/acronyms, "
                "references, and document overview."
            ),
            "2. Overall Description": (
                "Cover: product perspective (system context), major product functions, "
                "user classes & characteristics, operating environment, design constraints, "
                "and assumptions/dependencies."
            ),
            "3. System Features": (
                "For each major feature provide: Feature ID (FEAT-XXX), description, "
                "priority (Critical/High/Medium/Low), user value, and dependencies."
            ),
            "4. Functional Requirements": (
                "List every discrete functional requirement numbered FR-001, FR-002, … "
                "Each must state: ID, description, priority, inputs, outputs, and dependencies."
            ),
            "5. Non-Functional Requirements": (
                "Cover Performance (NFR-P), Security (NFR-S), Usability (NFR-U), "
                "Reliability (NFR-R), and Scalability (NFR-SC). Each requirement must be measurable."
            ),
            "6. External Interface Requirements": (
                "Cover: User Interfaces, Hardware Interfaces, Software Interfaces, "
                "and Communications Interfaces."
            ),
            "7. Appendices": (
                "Include: Glossary, Analysis Models (data-flow / sequence diagrams described "
                "in text), and any supplementary information."
            ),
        }
        return focus_map.get(section_title, "")
