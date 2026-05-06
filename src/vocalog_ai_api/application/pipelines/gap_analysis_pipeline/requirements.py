"""
Per-section completeness requirements for each document strategy.

Each entry is a list of plain-language criteria the section must satisfy
to be considered "complete." The gap analysis LLM uses these as a checklist
to identify what's missing or under-specified in the draft content.
"""

from typing import Dict, List

SECTION_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
    "srs": {
        "1. Introduction": [
            "States the purpose and intended audience of the document",
            "Defines the product scope with clear boundaries",
            "Includes a glossary or definitions section for domain-specific terms",
            "Lists any referenced standards or external documents",
        ],
        "2. Overall Description": [
            "Describes the system context and how it fits into a larger ecosystem",
            "Identifies and characterises all user classes",
            "Specifies the operating environment (OS, browser, hardware)",
            "Lists design constraints and dependencies",
            "Documents assumptions that must hold for requirements to be valid",
        ],
        "3. System Features": [
            "Each feature has a unique ID (FEAT-XXX)",
            "Each feature has an assigned priority (Critical/High/Medium/Low)",
            "Feature descriptions explain user value, not just technical function",
            "Dependencies between features are explicitly documented",
        ],
        "4. Functional Requirements": [
            "Every requirement has a unique, sequential ID (FR-001, FR-002, …)",
            "Each requirement is independently testable and verifiable",
            "Each requirement specifies the actor, action, and expected system response",
            "Priority is assigned to each requirement",
            "Dependencies and pre-conditions are documented where relevant",
        ],
        "5. Non-Functional Requirements": [
            "Performance: response time targets are measurable (e.g., < 200 ms at p99)",
            "Scalability: concurrent user or throughput targets are defined",
            "Security: authentication, authorisation, and encryption requirements exist",
            "Reliability: uptime/availability SLA is specified (e.g., 99.9%)",
            "Usability: accessibility or task-completion time targets are stated",
        ],
        "6. External Interface Requirements": [
            "User interface requirements are described (layout, standards, or accessibility)",
            "All third-party software integrations are identified",
            "Data exchange formats and protocols are specified",
            "Hardware interface constraints are noted if applicable",
        ],
        "7. Appendices": [
            "Glossary covers all domain-specific terms used in the document",
            "Supplementary diagrams or models are described or referenced",
        ],
    },
    "prd": {
        "1. Executive Summary": [
            "States the core problem being solved in one paragraph",
            "Describes the proposed solution at a high level",
            "Identifies primary target users",
            "Mentions expected business or user impact",
        ],
        "2. Product Vision & Goals": [
            "Articulates a one-to-two sentence long-term product vision",
            "Lists 3–5 concrete, measurable product goals",
            "Includes a clear problem statement",
            "Each goal has a measurable outcome or success criterion",
        ],
        "3. Target Users & Personas": [
            "Defines at least two named user personas",
            "Each persona includes role, key pain points, and goals",
            "Explains how the product addresses each persona's pain points",
            "User research or qualitative data backs persona definitions",
        ],
        "4. Feature Specifications": [
            "Each feature has a user story in 'As a … I want … so that …' format",
            "Acceptance criteria are defined for each feature",
            "Priority is assigned (P0/P1/P2) with justification",
            "In-scope vs. out-of-scope boundaries are explicit",
        ],
        "5. Success Metrics & KPIs": [
            "At least two primary KPIs are defined with target values",
            "Each KPI specifies a current baseline and target",
            "Measurement methodology is described (how/when metrics are collected)",
            "Review cadence is stated (weekly, monthly, per-release)",
        ],
        "6. Constraints & Assumptions": [
            "Technical constraints are listed (platform, legacy system, budget)",
            "Business constraints are listed (regulatory, compliance, timeline)",
            "All assumptions are enumerable and verifiable",
            "Risks associated with invalid assumptions are noted",
        ],
        "7. Release Timeline & Milestones": [
            "At least two release phases or milestones are defined",
            "Target dates or relative timeframes are stated per milestone",
            "Features included per phase are specified",
            "Key dependencies and risks per milestone are documented",
        ],
    },
    "sdd": {
        "1. System Overview": [
            "Describes the business purpose and system boundaries",
            "Identifies all major actors and external systems",
            "Explains high-level data flow through the system",
            "References or describes a context diagram",
        ],
        "2. Architecture Design": [
            "Names and justifies the chosen architectural style (e.g., microservices, monolith)",
            "Lists all major components with their responsibilities",
            "Describes communication patterns (sync/async, REST/gRPC/queue)",
            "Explains trade-offs of the chosen architecture",
        ],
        "3. Component Specifications": [
            "Each major component has a name, responsibility, and interface description",
            "Interfaces consumed and exposed per component are listed",
            "Internal state management strategy is described",
            "Failure modes and fallback behaviour are noted per component",
        ],
        "4. Data Models & Schemas": [
            "All persistent entities are defined with field names, types, and constraints",
            "Relationships between entities are described",
            "Indexing strategy is documented",
            "Data migration or versioning strategy is mentioned",
        ],
        "5. API Contracts": [
            "Every API endpoint specifies HTTP method, path, and authentication",
            "Request schemas include field names, types, and validation rules",
            "Success and error response schemas are defined",
            "Rate limiting or throttling policies are documented if applicable",
        ],
        "6. Security Design": [
            "Authentication mechanism is specified (e.g., JWT, OAuth2, API keys)",
            "Authorisation model is described (RBAC, ABAC, or other)",
            "Data encryption strategy covers both at-rest and in-transit",
            "Input validation and sanitisation strategy is stated",
            "Secrets management approach is documented",
        ],
        "7. Deployment & Infrastructure": [
            "Hosting environment is specified (cloud provider, on-prem)",
            "Containerisation or packaging strategy is described",
            "CI/CD pipeline stages are outlined",
            "Scaling strategy (horizontal/vertical, auto-scaling triggers) is defined",
            "Monitoring and alerting setup is described",
        ],
        "8. Technical Constraints & Trade-offs": [
            "Known platform or language constraints are listed",
            "At least two major design trade-offs are documented with rationale",
            "Rejected architectural alternatives are mentioned with reasons",
        ],
    },
}


def get_section_requirements(document_type: str, section_title: str) -> List[str]:
    """Returns completeness criteria for the given strategy + section, or [] if not found."""
    return SECTION_REQUIREMENTS.get(document_type, {}).get(section_title, [])


def get_all_sections_with_requirements(document_type: str) -> Dict[str, List[str]]:
    """Returns the full requirements map for a strategy type."""
    return SECTION_REQUIREMENTS.get(document_type, {})
