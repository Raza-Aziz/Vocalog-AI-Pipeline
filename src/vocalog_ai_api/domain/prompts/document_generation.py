"""
Prompts for document generation, specifically for SRS (Software Requirements Specification) documents.
"""
"""
Prompts for document generation
Located in: src/vocalog_ai_api/domain/prompts/document_generation.py
"""
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.claude import SectionType


def get_section_prompt(section_type: SectionType) -> str:
    """
    Get the system prompt for a specific section type
    
    Args:
        section_type: Type of section to generate
        
    Returns:
        System prompt string for that section
    """
    prompts = {
        SectionType.INTRODUCTION: """Generate the Introduction section for an SRS document. Include:

1. **Purpose**: Clear statement of the document's purpose and intended audience
2. **Scope**: Brief overview of the software product and its key capabilities
3. **Definitions and Acronyms**: Important terms and their definitions
4. **References**: Any referenced documents or standards
5. **Overview**: Brief description of the rest of the document

Keep it concise (2-3 paragraphs per subsection) and professional. Use markdown formatting with appropriate headers (###).""",

        SectionType.SYSTEM_OVERVIEW: """Generate the System Overview section. Include:

1. **System Context**: How the system fits into the larger environment
2. **High-Level Architecture**: Major components and their relationships
3. **System Boundaries**: What is inside vs outside the system scope
4. **Key Technologies**: Main technologies and platforms used
5. **System Constraints**: Any technical or business constraints

Focus on the big picture. Use diagrams descriptions where helpful (in markdown).""",

        SectionType.FUNCTIONAL_REQUIREMENTS: """Generate the Functional Requirements section. Include:

1. Detailed functional requirements numbered as FR-001, FR-002, etc.
2. Each requirement should specify:
   - **Requirement ID**: FR-XXX
   - **Description**: Clear, testable requirement statement
   - **Priority**: High/Medium/Low
   - **Input**: Expected inputs
   - **Output**: Expected outputs
   - **Dependencies**: Related requirements

Format as a numbered list with subsections. Be specific and measurable. Aim for 8-12 key requirements.""",

        SectionType.NON_FUNCTIONAL_REQUIREMENTS: """Generate the Non-Functional Requirements section. Include requirements for:

1. **Performance Requirements** (NFR-P-XXX):
   - Response times, throughput, capacity
   
2. **Security Requirements** (NFR-S-XXX):
   - Authentication, authorization, data protection
   
3. **Usability Requirements** (NFR-U-XXX):
   - User interface standards, accessibility
   
4. **Reliability Requirements** (NFR-R-XXX):
   - Availability, fault tolerance, recovery
   
5. **Scalability Requirements** (NFR-SC-XXX):
   - Growth capacity, horizontal/vertical scaling

Each requirement should be specific and measurable. Use appropriate numbering.""",

        SectionType.USER_REQUIREMENTS: """Generate the User Requirements section. Include:

1. **User Personas**: Key user types and their characteristics
2. **User Stories**: Format as "As a [user], I want [goal] so that [benefit]"
3. **User Interface Requirements**: 
   - Navigation patterns
   - Key screens/views
   - Interaction patterns
4. **Accessibility Requirements**: Standards compliance (WCAG, etc.)
5. **User Documentation Needs**: Help, training, onboarding

Focus on the user's perspective and needs. Include 5-8 key user stories.""",

        SectionType.SYSTEM_FEATURES: """Generate the System Features section. Include:

1. **Feature List**: Major features with detailed descriptions
2. For each feature:
   - **Feature ID**: FEAT-XXX
   - **Description**: What the feature does
   - **Priority**: Critical/High/Medium/Low
   - **User Value**: Why this feature matters
   - **Dependencies**: Related features or requirements
   - **Technical Notes**: Implementation considerations

Organize features logically (by module, user journey, etc.). Include 6-10 major features."""
    }
    
    return prompts.get(
        section_type,
        "Generate this section based on the meeting minutes and maintain consistency with other sections."
    )

def build_section_prompt(
    section_id: str,
    section_title: str,
    meeting_context: list,
    previous_sections: dict,
    initial_context: str = None
) -> str:
    """
    Build a prompt for generating a specific SRS section.
    
    Args:
        section_id: The section identifier (e.g., "introduction", "system_features")
        section_title: The section title (e.g., "1. Introduction")
        meeting_context: List of context snippets from meeting minutes
        previous_sections: Dictionary of previously finalized sections (section_id -> content)
        initial_context: Optional initial context/requirements
        
    Returns:
        Formatted prompt string for the LLM
    """
    
    # Format meeting context
    meeting_context_text = ""
    if meeting_context:
        meeting_context_text = "\n\n## Relevant Meeting Minutes Context:\n\n"
        for i, ctx in enumerate(meeting_context, 1):
            meeting_context_text += f"### Context {i}:\n{ctx.get('text', '')}\n\n"
    
    # Format previous sections
    previous_sections_text = ""
    if previous_sections:
        previous_sections_text = "\n\n## Previously Generated Sections:\n\n"
        for prev_section_id, prev_content in previous_sections.items():
            previous_sections_text += f"### {prev_section_id}:\n{prev_content}\n\n"
    
    # Initial context
    initial_context_text = ""
    if initial_context:
        initial_context_text = f"\n\n## Initial Requirements/Context:\n{initial_context}\n"
    
    # Section-specific instructions
    section_instructions = _get_section_instructions(section_id)
    
    prompt = f"""You are a technical documentation specialist generating a Software Requirements Specification (SRS) document.

Your task is to generate the "{section_title}" section of an SRS document based on the provided context from meeting minutes and previously generated sections.

{section_instructions}
{initial_context_text}
{meeting_context_text}
{previous_sections_text}

## Instructions:
1. Generate ONLY the content for the "{section_title}" section.
2. Use clear, professional, and technical language appropriate for an SRS document.
3. Format the content in Markdown.
4. Be specific and detailed based on the context provided.
5. If the context doesn't provide enough information for a particular subsection, indicate that it needs to be specified.
6. Ensure consistency with previously generated sections.
7. Do NOT include the section title/heading in your output - only the content.
8. Do NOT repeat information that was already covered in previous sections unless necessary for clarity.

Generate the section content now:"""
    
    return prompt


def _get_section_instructions(section_id: str) -> str:
    """Get section-specific instructions based on section ID."""
    
    instructions_map = {
        "introduction": """
## Section: 1. Introduction

This section should include:
- 1.1 Purpose: Describe the purpose of the SRS document
- 1.2 Scope: Define the scope of the software product
- 1.3 Definitions, Acronyms, and Abbreviations: List terms and abbreviations used
- 1.4 References: List any referenced documents
- 1.5 Overview: Provide an overview of the document structure
""",
        "overall_description": """
## Section: 2. Overall Description

This section should include:
- 2.1 Product Perspective: Describe how the product relates to other systems
- 2.2 Product Functions: High-level summary of major functions
- 2.3 User Classes and Characteristics: Describe different user types
- 2.4 Operating Environment: Hardware and software platforms
- 2.5 Design and Implementation Constraints: Constraints affecting design
- 2.6 User Documentation: Documentation requirements
- 2.7 Assumptions and Dependencies: Assumptions and external dependencies
""",
        "system_features": """
## Section: 3. System Features

This section should include detailed functional requirements organized by feature:
- For each major feature, provide:
  - 3.X Feature Name
  - 3.X.1 Description and Priority
  - 3.X.2 Action/Item Description
  - 3.X.3 Input/Output
  - 3.X.4 Processing
  - 3.X.5 User Interface Requirements (if applicable)
""",
        "external_interfaces": """
## Section: 4. External Interface Requirements

This section should include:
- 4.1 User Interfaces: Description of UI requirements
- 4.2 Hardware Interfaces: Hardware requirements and interfaces
- 4.3 Software Interfaces: Interfaces with other software systems
- 4.4 Communications Interfaces: Network and communication requirements
""",
        "non_functional": """
## Section: 5. Non-functional Requirements

This section should include:
- 5.1 Performance Requirements: Response time, throughput, resource usage
- 5.2 Safety Requirements: Safety-critical aspects
- 5.3 Security Requirements: Security and privacy requirements
- 5.4 Software Quality Attributes: Reliability, maintainability, portability
- 5.5 Business Rules: Business logic and rules
""",
        "appendices": """
## Section: 6. Appendices

This section should include:
- Appendix A: Glossary (if not covered in Introduction)
- Appendix B: Analysis Models (diagrams, data flow, etc.)
- Appendix C: Issue List (if applicable)
- Any other supplementary information
"""
    }
    
    return instructions_map.get(section_id, "Generate comprehensive content for this section based on the provided context.")


def build_refinement_prompt(
    section_id: str,
    section_title: str,
    current_draft: str,
    feedback_text: str,
    meeting_context: list,
    previous_sections: dict
) -> str:
    """
    Build a prompt for refining a section based on user feedback.
    
    Args:
        section_id: The section identifier
        section_title: The section title
        current_draft: Current draft of the section
        feedback_text: User's feedback for refinement
        meeting_context: Context from meeting minutes
        previous_sections: Previously finalized sections
        
    Returns:
        Formatted prompt string for refinement
    """
    
    meeting_context_text = ""
    if meeting_context:
        meeting_context_text = "\n\n## Relevant Meeting Minutes Context:\n\n"
        for i, ctx in enumerate(meeting_context, 1):
            meeting_context_text += f"### Context {i}:\n{ctx.get('text', '')}\n\n"
    
    previous_sections_text = ""
    if previous_sections:
        previous_sections_text = "\n\n## Previously Generated Sections:\n\n"
        for prev_section_id, prev_content in previous_sections.items():
            previous_sections_text += f"### {prev_section_id}:\n{prev_content}\n\n"
    
    prompt = f"""You are refining the "{section_title}" section of an SRS document based on user feedback.

## Current Draft:
{current_draft}

## User Feedback:
{feedback_text}
{meeting_context_text}
{previous_sections_text}

## Instructions:
1. Revise the section content based on the user's feedback.
2. Maintain consistency with previously generated sections.
3. Keep the same Markdown formatting.
4. Do NOT include the section title/heading - only the content.
5. Ensure the revised content addresses all points in the feedback.

Generate the refined section content:"""
    
    return prompt
