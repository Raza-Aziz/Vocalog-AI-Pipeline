"""
Test suite 1 — Cross-Strategy Outline Validation & Prompt Snapshot Tests

Covers:
  • Each strategy exposes the correct, non-overlapping section headings.
  • Headings that are semantically exclusive to one strategy do NOT appear in others.
  • get_strategy() registry resolves to the expected class for each key.
  • The init node populates sections_outline from the active strategy (not hardcoded).
  • The LLM prompt built for each strategy contains the correct persona string
    (snapshot test — intercepts the actual HumanMessage sent to llm.invoke).
  • A spy on get_strategy() confirms the correct class is resolved during generation.
"""

import uuid
from unittest.mock import patch, call

import pytest

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import (
    get_strategy,
    list_supported_types,
)
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.srs import SRSStrategy
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.prd import PRDStrategy
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies.sdd import SDDStrategy
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes import initialize_document

from tests.test_doc_generation.conftest import (
    build_initial_state,
    make_llm_response,
    MOCK_LLM_PATCH,
    MOCK_INGEST_PATCH,
    MOCK_RETRIEVE_PATCH,
    MOCK_CONTEXT,
    SAMPLE_MINUTES,
)


# ── 1. Strategy registry ──────────────────────────────────────────────────────

class TestStrategyRegistry:
    def test_all_three_types_registered(self):
        types = list_supported_types()
        assert "srs" in types
        assert "prd" in types
        assert "sdd" in types

    def test_get_strategy_srs_returns_srs_instance(self):
        assert isinstance(get_strategy("srs"), SRSStrategy)

    def test_get_strategy_prd_returns_prd_instance(self):
        assert isinstance(get_strategy("prd"), PRDStrategy)

    def test_get_strategy_sdd_returns_sdd_instance(self):
        assert isinstance(get_strategy("sdd"), SDDStrategy)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown document type"):
            get_strategy("marketing_plan")

    def test_lookup_is_case_insensitive(self):
        assert isinstance(get_strategy("SRS"), SRSStrategy)
        assert isinstance(get_strategy("PRD"), PRDStrategy)
        assert isinstance(get_strategy("SDD"), SDDStrategy)


# ── 2. Section heading correctness ────────────────────────────────────────────

class TestSRSSections:
    def setup_method(self):
        self.strategy = get_strategy("srs")

    def test_srs_has_expected_headings(self):
        sections = self.strategy.sections
        assert any("Introduction" in s for s in sections)
        assert any("Functional Requirements" in s for s in sections)
        assert any("Non-Functional Requirements" in s for s in sections)

    def test_srs_does_not_contain_prd_headings(self):
        sections = self.strategy.sections
        # These are PRD-exclusive concepts
        assert not any("Product Vision" in s for s in sections)
        assert not any("User Stories" in s for s in sections)
        assert not any("KPI" in s for s in sections)

    def test_srs_does_not_contain_sdd_headings(self):
        sections = self.strategy.sections
        # These are SDD-exclusive concepts
        assert not any("Architecture Design" in s for s in sections)
        assert not any("API Contracts" in s for s in sections)
        assert not any("Deployment" in s for s in sections)

    def test_srs_sections_are_ordered_list(self):
        sections = self.strategy.sections
        assert isinstance(sections, list)
        assert len(sections) >= 5
        # Headings should be numbered sequentially
        for i, heading in enumerate(sections, start=1):
            assert heading.startswith(f"{i}."), (
                f"Expected heading {i} to start with '{i}.', got: '{heading}'"
            )


class TestPRDSections:
    def setup_method(self):
        self.strategy = get_strategy("prd")

    def test_prd_has_expected_headings(self):
        sections = self.strategy.sections
        assert any("Executive Summary" in s for s in sections)
        assert any("Product Vision" in s for s in sections)
        assert any("Target Users" in s or "Personas" in s for s in sections)
        assert any("Success Metrics" in s or "KPI" in s for s in sections)

    def test_prd_does_not_contain_srs_headings(self):
        sections = self.strategy.sections
        assert not any("Functional Requirements" in s for s in sections)
        assert not any("Non-Functional Requirements" in s for s in sections)
        assert not any("External Interface" in s for s in sections)

    def test_prd_does_not_contain_sdd_headings(self):
        sections = self.strategy.sections
        assert not any("Architecture Design" in s for s in sections)
        assert not any("Data Models" in s for s in sections)
        assert not any("Deployment" in s for s in sections)


class TestSDDSections:
    def setup_method(self):
        self.strategy = get_strategy("sdd")

    def test_sdd_has_expected_headings(self):
        sections = self.strategy.sections
        assert any("System Overview" in s for s in sections)
        assert any("Architecture Design" in s for s in sections)
        assert any("API Contracts" in s for s in sections)
        assert any("Deployment" in s for s in sections)

    def test_sdd_does_not_contain_srs_headings(self):
        sections = self.strategy.sections
        assert not any("Functional Requirements" in s for s in sections)
        assert not any("Non-Functional Requirements" in s for s in sections)

    def test_sdd_does_not_contain_prd_headings(self):
        sections = self.strategy.sections
        assert not any("Product Vision" in s for s in sections)
        assert not any("Executive Summary" in s for s in sections)
        assert not any("KPI" in s for s in sections)

    def test_sdd_has_more_sections_than_srs(self):
        # SDD is the most detailed document type in this implementation
        assert len(get_strategy("sdd").sections) >= len(get_strategy("srs").sections)


# ── 3. Init node populates outline from strategy (not hardcoded) ──────────────

class TestInitNodeOutline:
    """
    Calls initialize_document() directly (bypassing the graph) to verify
    that the sections_outline returned equals strategy.sections for each type.
    """

    def _run_init(self, doc_type: str) -> dict:
        doc_id = str(uuid.uuid4())
        state = build_initial_state(doc_id, doc_type)
        with patch(MOCK_INGEST_PATCH):
            return initialize_document(state)

    def test_srs_outline_matches_strategy(self):
        result = self._run_init("srs")
        assert result["sections_outline"] == get_strategy("srs").sections

    def test_prd_outline_matches_strategy(self):
        result = self._run_init("prd")
        assert result["sections_outline"] == get_strategy("prd").sections

    def test_sdd_outline_matches_strategy(self):
        result = self._run_init("sdd")
        assert result["sections_outline"] == get_strategy("sdd").sections

    def test_init_resets_index_and_history(self):
        result = self._run_init("srs")
        assert result["current_section_index"] == 0
        assert result["refinement_history"] == {}
        assert result["final_document"] == []
        assert result["is_complete"] is False
        assert result["pending_action"] is None


# ── 4. Prompt snapshot tests — persona strings in LLM calls ──────────────────

class TestPersonaPromptSnapshot:
    """
    Intercepts the HumanMessage sent to llm.invoke and asserts that the
    correct strategy persona and document-type name appear verbatim.
    """

    def _capture_prompt(self, doc_type: str, test_graph, mock_qdrant) -> str:
        """Run first draft generation and return the prompt text sent to the LLM."""
        doc_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": doc_id}}
        state = build_initial_state(doc_id, doc_type)

        with patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.return_value = make_llm_response("Snapshot test content.")
            test_graph.invoke(state, config=config)

            assert mock_llm.invoke.called, "llm.invoke was never called"
            # The node calls llm.invoke([HumanMessage(content=prompt)])
            first_call_args = mock_llm.invoke.call_args_list[0]
            messages = first_call_args[0][0]   # positional arg 0: list of messages
            return messages[0].content          # HumanMessage.content

    def test_srs_prompt_contains_technical_writer_persona(self, test_graph, mock_qdrant):
        prompt = self._capture_prompt("srs", test_graph, mock_qdrant)
        assert "Technical Writer" in prompt or "Systems Analyst" in prompt, (
            f"SRS persona not found in prompt. Got:\n{prompt[:500]}"
        )

    def test_srs_prompt_contains_document_type_name(self, test_graph, mock_qdrant):
        prompt = self._capture_prompt("srs", test_graph, mock_qdrant)
        assert "Software Requirements Specification" in prompt, (
            f"SRS document name missing. Got:\n{prompt[:500]}"
        )

    def test_prd_prompt_contains_product_manager_persona(self, test_graph, mock_qdrant):
        prompt = self._capture_prompt("prd", test_graph, mock_qdrant)
        assert "Product Manager" in prompt, (
            f"PRD persona not found in prompt. Got:\n{prompt[:500]}"
        )

    def test_prd_prompt_contains_document_type_name(self, test_graph, mock_qdrant):
        prompt = self._capture_prompt("prd", test_graph, mock_qdrant)
        assert "Product Requirements Document" in prompt, (
            f"PRD document name missing. Got:\n{prompt[:500]}"
        )

    def test_sdd_prompt_contains_architect_persona(self, test_graph, mock_qdrant):
        prompt = self._capture_prompt("sdd", test_graph, mock_qdrant)
        assert "Architect" in prompt, (
            f"SDD persona not found in prompt. Got:\n{prompt[:500]}"
        )

    def test_sdd_prompt_contains_document_type_name(self, test_graph, mock_qdrant):
        prompt = self._capture_prompt("sdd", test_graph, mock_qdrant)
        assert "Software Design Document" in prompt, (
            f"SDD document name missing. Got:\n{prompt[:500]}"
        )

    def test_section_focus_injected_for_sdd_architecture(self, test_graph, mock_qdrant):
        """The Architecture Design section should include architecture-specific focus text."""
        strategy = get_strategy("sdd")
        arch_section = next(s for s in strategy.sections if "Architecture" in s)

        prompt = strategy.build_initial_prompt(
            section_title=arch_section,
            context=MOCK_CONTEXT,
        )
        assert "architectural style" in prompt.lower() or "component" in prompt.lower(), (
            f"Architecture section focus missing from SDD prompt:\n{prompt[:500]}"
        )

    def test_section_focus_injected_for_prd_kpi(self):
        """The Success Metrics section should include KPI-specific focus text."""
        strategy = get_strategy("prd")
        kpi_section = next(s for s in strategy.sections if "Metrics" in s or "KPI" in s)

        prompt = strategy.build_initial_prompt(
            section_title=kpi_section,
            context=MOCK_CONTEXT,
        )
        assert "KPI" in prompt or "metric" in prompt.lower(), (
            f"KPI focus missing from PRD prompt:\n{prompt[:500]}"
        )


# ── 5. Strategy spy — correct class resolved during graph execution ────────────

class TestStrategySpyDuringGeneration:
    """
    Uses unittest.mock wraps= to spy on get_strategy() without replacing it,
    then asserts the expected doc_type was requested during graph execution.
    """

    @pytest.mark.parametrize("doc_type", ["srs", "prd", "sdd"])
    def test_correct_strategy_class_resolved(
        self, doc_type, test_graph, mock_qdrant, mock_llm
    ):
        doc_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": doc_id}}
        state = build_initial_state(doc_id, doc_type)

        spy_path = (
            "vocalog_ai_api.application.pipelines.doc_generation_pipeline"
            ".nodes.get_strategy"
        )
        with patch(spy_path, wraps=get_strategy) as spy:
            test_graph.invoke(state, config=config)

            # get_strategy is called in both initialize_document AND generate_section
            calls = [c.args[0] for c in spy.call_args_list]
            assert doc_type in calls, (
                f"Expected get_strategy('{doc_type}') to be called. "
                f"Actual calls: {calls}"
            )

    @pytest.mark.parametrize("doc_type", ["srs", "prd", "sdd"])
    def test_wrong_strategy_not_used(self, doc_type, test_graph, mock_qdrant, mock_llm):
        """Verify the other two strategies are NOT instantiated for a given doc_type."""
        other_types = {"srs", "prd", "sdd"} - {doc_type}
        doc_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": doc_id}}
        state = build_initial_state(doc_id, doc_type)

        spy_path = (
            "vocalog_ai_api.application.pipelines.doc_generation_pipeline"
            ".nodes.get_strategy"
        )
        with patch(spy_path, wraps=get_strategy) as spy:
            test_graph.invoke(state, config=config)

            calls = {c.args[0] for c in spy.call_args_list}
            for wrong_type in other_types:
                assert wrong_type not in calls, (
                    f"Strategy '{wrong_type}' was unexpectedly resolved "
                    f"during a '{doc_type}' generation."
                )
