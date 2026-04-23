from abc import ABC, abstractmethod
from typing import List


class DocumentStrategy(ABC):
    """
    Abstract base class for document generation strategies.

    Each concrete strategy encapsulates the structural definition (headings,
    section count) and the contextual LLM persona for one document type.
    To add a new document type, subclass this and register it in the registry.
    """

    @property
    @abstractmethod
    def document_type(self) -> str:
        """Short identifier used in state and API (e.g. 'srs', 'prd', 'sdd')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable document title (e.g. 'Software Requirements Specification')."""

    @property
    @abstractmethod
    def sections(self) -> List[str]:
        """Ordered list of section headings that form the document outline."""

    @property
    @abstractmethod
    def persona(self) -> str:
        """
        Role description injected as the LLM system persona.
        Should describe the expert role AND the focus area for this document type.
        """

    @property
    @abstractmethod
    def generation_instructions(self) -> str:
        """
        High-level generation instructions appended to every initial draft prompt.
        Should reflect the unique quality bar and conventions of this document type.
        """

    @property
    @abstractmethod
    def refinement_instructions(self) -> str:
        """
        High-level refinement instructions appended to every feedback/refine prompt.
        Should remind the LLM of this document's standards when incorporating feedback.
        """

    def get_section_focus(self, section_title: str) -> str:
        """
        Optional per-section guidance. Subclasses may override to return
        section-specific instructions. Default returns an empty string.
        """
        return ""

    def build_initial_prompt(self, section_title: str, context: str) -> str:
        """Compose the full initial-draft prompt for a section."""
        section_focus = self.get_section_focus(section_title)
        focus_block = f"\n\nSection Focus:\n{section_focus}" if section_focus else ""

        return (
            f"You are {self.persona}.\n\n"
            f"Task: Write the '{section_title}' section of a {self.display_name}.\n"
            f"{focus_block}\n\n"
            f"Reference Context (use ONLY the information below — do not hallucinate facts):\n"
            f"{context}\n\n"
            f"{self.generation_instructions}"
        )

    def build_refinement_prompt(
        self, section_title: str, current_draft: str, feedback: str, context: str
    ) -> str:
        """Compose the full refinement prompt incorporating user feedback."""
        return (
            f"You are {self.persona}.\n\n"
            f"Task: Revise the '{section_title}' section of a {self.display_name} "
            f"based on the reviewer's feedback below.\n\n"
            f"Current Draft:\n{current_draft}\n\n"
            f"Reviewer Feedback:\n{feedback}\n\n"
            f"Reference Context (for factual accuracy):\n{context}\n\n"
            f"{self.refinement_instructions}"
        )
