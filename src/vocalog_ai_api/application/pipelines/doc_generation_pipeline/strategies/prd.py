from typing import List
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.base import DocumentStrategy


class PRDStrategy(DocumentStrategy):
    """
    Strategy for Product Requirements Documents.
    Persona: Senior Product Manager focused on user value and business outcomes.
    """

    @property
    def document_type(self) -> str:
        return "prd"

    @property
    def display_name(self) -> str:
        return "Product Requirements Document (PRD)"

    @property
    def sections(self) -> List[str]:
        return [
            "1. Executive Summary",
            "2. Product Vision & Goals",
            "3. Target Users & Personas",
            "4. Feature Specifications",
            "5. Success Metrics & KPIs",
            "6. Constraints & Assumptions",
            "7. Release Timeline & Milestones",
        ]

    @property
    def persona(self) -> str:
        return (
            "a Senior Product Manager with a track record of shipping B2B SaaS products. "
            "Your writing centres on user value, business outcomes, and measurable success. "
            "Every section must answer 'Why does this matter to the user?' before describing "
            "'What will be built?' Avoid implementation detail — leave that to engineering."
        )

    @property
    def generation_instructions(self) -> str:
        return (
            "Instructions:\n"
            "1. Lead with user value and business rationale in every section.\n"
            "2. Format output in Markdown. Use ### for sub-headings.\n"
            "3. Write user stories in the format: "
            "'As a [persona], I want [goal] so that [benefit]'.\n"
            "4. Define success metrics as specific, measurable KPIs (e.g., '< 2 s load time').\n"
            "5. Do NOT include the section heading itself — only the section body.\n"
            "6. Where the context is insufficient, mark gaps explicitly: "
            "'[Open Question — needs PM/stakeholder decision]'.\n"
            "7. Avoid technical implementation detail unless directly relevant to scope."
        )

    @property
    def refinement_instructions(self) -> str:
        return (
            "Refinement Instructions:\n"
            "1. Address every point in the feedback while preserving the user-value focus.\n"
            "2. Ensure any new KPIs remain specific and measurable.\n"
            "3. Keep the PRD tone: strategic and outcome-oriented, not implementation-focused.\n"
            "4. Do NOT include the section heading — only the revised body.\n"
            "5. Retain all existing user stories unless the feedback explicitly removes one."
        )

    def get_section_focus(self, section_title: str) -> str:
        focus_map = {
            "1. Executive Summary": (
                "Two-to-three paragraph overview: the problem being solved, the product solution, "
                "primary target users, and expected business impact."
            ),
            "2. Product Vision & Goals": (
                "State the long-term product vision (1–2 sentences), then list 3–5 concrete "
                "product goals with measurable outcomes. Include the problem statement and "
                "how this product addresses it."
            ),
            "3. Target Users & Personas": (
                "Define 2–4 user personas. For each: name, role, key pain points, goals, "
                "and how this product helps them. Include a 'day-in-the-life' sketch if context allows."
            ),
            "4. Feature Specifications": (
                "For each major feature provide: Feature Name, user story, acceptance criteria, "
                "priority (P0/P1/P2), and scope boundary (in-scope vs. out-of-scope)."
            ),
            "5. Success Metrics & KPIs": (
                "Define primary and secondary KPIs. For each metric: name, current baseline, "
                "target value, measurement method, and review cadence."
            ),
            "6. Constraints & Assumptions": (
                "List technical, business, regulatory, or resource constraints. "
                "Enumerate assumptions that must hold for the product plan to be valid."
            ),
            "7. Release Timeline & Milestones": (
                "Outline phased delivery: milestones, target dates, features included per phase, "
                "and key dependencies or risks per milestone."
            ),
        }
        return focus_map.get(section_title, "")
