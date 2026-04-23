from typing import List
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.base import DocumentStrategy


class SDDStrategy(DocumentStrategy):
    """
    Strategy for Software Design Documents.
    Persona: Lead Software Architect focused on technical design and system integrity.
    """

    @property
    def document_type(self) -> str:
        return "sdd"

    @property
    def display_name(self) -> str:
        return "Software Design Document (SDD)"

    @property
    def sections(self) -> List[str]:
        return [
            "1. System Overview",
            "2. Architecture Design",
            "3. Component Specifications",
            "4. Data Models & Schemas",
            "5. API Contracts",
            "6. Security Design",
            "7. Deployment & Infrastructure",
            "8. Technical Constraints & Trade-offs",
        ]

    @property
    def persona(self) -> str:
        return (
            "a Lead Software Architect with deep expertise in distributed systems, API design, "
            "and production-grade software engineering. Your writing is precise, technically "
            "rigorous, and implementation-ready. Every design decision must be justified by "
            "its technical constraints, scalability implications, or security requirements. "
            "Favour clarity over jargon; a mid-level engineer reading this document must be "
            "able to implement each component without ambiguity."
        )

    @property
    def generation_instructions(self) -> str:
        return (
            "Instructions:\n"
            "1. Write for an engineering audience — be technically precise.\n"
            "2. Format output in Markdown. Use ### for sub-headings, code blocks for schemas/APIs.\n"
            "3. Every design decision must include a brief rationale ('Why this approach').\n"
            "4. Describe interfaces, data contracts, and component boundaries explicitly.\n"
            "5. Do NOT include the section heading itself — only the section body.\n"
            "6. Mark unknowns explicitly: '[Architecture Decision Record (ADR) required]'.\n"
            "7. Do not speculate on technology choices absent from the provided context."
        )

    @property
    def refinement_instructions(self) -> str:
        return (
            "Refinement Instructions:\n"
            "1. Address all feedback while preserving technical rigour and decision rationale.\n"
            "2. Ensure any new interfaces or schemas remain internally consistent.\n"
            "3. If feedback introduces a scope change, flag it: "
            "'[Scope Change — verify with PM before implementing]'.\n"
            "4. Do NOT include the section heading — only the revised body.\n"
            "5. Retain code block formatting for all schemas and API contracts."
        )

    def get_section_focus(self, section_title: str) -> str:
        focus_map = {
            "1. System Overview": (
                "Describe the system context: business purpose, major actors, high-level "
                "data flow, and system boundaries. Include a textual description of a "
                "context diagram if one can be inferred from the context."
            ),
            "2. Architecture Design": (
                "Describe the chosen architectural style (e.g., microservices, event-driven, "
                "layered). List major components and their responsibilities. Explain the "
                "communication patterns between components (sync/async, REST/gRPC/message queue). "
                "Justify the architecture choice against technical constraints."
            ),
            "3. Component Specifications": (
                "For each major component provide: name, responsibility, interfaces it exposes, "
                "interfaces it consumes, internal state management, and failure modes. "
                "Use a table or structured list for clarity."
            ),
            "4. Data Models & Schemas": (
                "Define all persistent entities with field names, types, constraints, and "
                "relationships. Use a table or code block (e.g., SQL DDL or JSON Schema). "
                "Describe indexing strategy and any denormalization decisions."
            ),
            "5. API Contracts": (
                "For each API endpoint: HTTP method + path, request schema (with field types "
                "and validation rules), success response schema, error response codes, and "
                "authentication/authorisation requirements. Use Markdown code blocks."
            ),
            "6. Security Design": (
                "Cover: authentication mechanism, authorisation model (RBAC/ABAC), data "
                "encryption (at rest and in transit), input validation strategy, secrets "
                "management, and any relevant compliance requirements."
            ),
            "7. Deployment & Infrastructure": (
                "Describe: hosting environment, containerisation strategy (Docker/K8s), "
                "CI/CD pipeline, environment tiers (dev/staging/prod), scaling strategy, "
                "and monitoring/alerting setup."
            ),
            "8. Technical Constraints & Trade-offs": (
                "List known technical constraints (language, platform, budget). "
                "For each major design trade-off considered, describe the options evaluated "
                "and the rationale for the chosen approach."
            ),
        }
        return focus_map.get(section_title, "")
