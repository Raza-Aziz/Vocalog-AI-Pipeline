"""
Test suite 4 — Final Document Compilation & Order (Happy Path)

Simulates a user approving every section of a document from start to finish.
Validates:
  • final_document contains all sections in the correct chronological order.
  • Each section's title matches the corresponding heading from strategy.sections.
  • Each section's content is the mock content that was generated for that slot.
  • is_complete is toggled to True only after the last section is approved.
  • is_complete is False at every intermediate step.
  • The completed state is durable — get_state() after completion returns is_complete=True.
  • All three document types (SRS, PRD, SDD) are validated independently.

Design: the mock LLM is configured with side_effect so each call returns a
unique, verifiable string ("Section N content"), allowing order verification
without relying on real LLM output.
"""

import uuid
from unittest.mock import patch

import pytest

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import (
    get_strategy,
)
from tests.test_doc_generation.conftest import (
    build_initial_state,
    make_llm_response,
    MOCK_LLM_PATCH,
    MOCK_INGEST_PATCH,
    MOCK_RETRIEVE_PATCH,
    MOCK_CONTEXT,
)


# ── Core happy-path driver ────────────────────────────────────────────────────

def run_full_approval(graph, doc_type: str) -> tuple[str, list[dict], dict]:
    """
    Drive a document from scratch to is_complete=True by approving every section.

    The LLM mock returns "Section {i} content." for each sequential call so
    content and order can be asserted deterministically.

    Returns:
        (doc_id, intermediate_states, final_state)
        where intermediate_states[i] is the result state after approving section i
        (before the last approve which triggers is_complete).
    """
    strategy = get_strategy(doc_type)
    n_sections = len(strategy.sections)
    doc_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": doc_id}}

    # Prepare one unique LLM response per section
    draft_contents = [f"Section {i} content for {doc_type.upper()}." for i in range(n_sections)]
    llm_responses = [make_llm_response(c) for c in draft_contents]

    intermediate_states = []

    with patch(MOCK_INGEST_PATCH), \
         patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
         patch(MOCK_LLM_PATCH) as mock_llm:

        mock_llm.invoke.side_effect = llm_responses

        # --- Initial generation (section 0) ---
        initial = build_initial_state(doc_id, doc_type)
        state = graph.invoke(initial, config=config)
        assert state["current_section_content"] == draft_contents[0]

        # --- Approve sections 0 … N-2 (each approve triggers next generation) ---
        for section_idx in range(n_sections - 1):
            graph.update_state(config, {"pending_action": "approve", "feedback_notes": None})
            state = graph.invoke(None, config=config)
            intermediate_states.append(state)

            assert state["is_complete"] is False, (
                f"is_complete must be False after approving section {section_idx} "
                f"(there are still more sections to review)."
            )
            assert state["current_section_index"] == section_idx + 1

        # --- Approve the final section ---
        graph.update_state(config, {"pending_action": "approve", "feedback_notes": None})
        final_state = graph.invoke(None, config=config)

    return doc_id, intermediate_states, final_state, draft_contents


# ── Parametrised happy-path tests ─────────────────────────────────────────────

@pytest.mark.parametrize("doc_type", ["srs", "prd", "sdd"])
class TestHappyPath:

    def test_is_complete_true_after_last_approval(self, doc_type, test_graph, mock_qdrant):
        _, _, final, _ = run_full_approval(test_graph, doc_type)
        assert final["is_complete"] is True, (
            f"is_complete must be True after approving all {doc_type.upper()} sections."
        )

    def test_final_document_contains_all_sections(self, doc_type, test_graph, mock_qdrant):
        strategy = get_strategy(doc_type)
        _, _, final, _ = run_full_approval(test_graph, doc_type)

        assert len(final["final_document"]) == len(strategy.sections), (
            f"final_document has {len(final['final_document'])} sections, "
            f"expected {len(strategy.sections)} for {doc_type.upper()}."
        )

    def test_section_titles_match_strategy_in_order(self, doc_type, test_graph, mock_qdrant):
        strategy = get_strategy(doc_type)
        _, _, final, _ = run_full_approval(test_graph, doc_type)

        for i, (expected_title, section) in enumerate(
            zip(strategy.sections, final["final_document"])
        ):
            assert section["title"] == expected_title, (
                f"Section {i} title mismatch for {doc_type.upper()}. "
                f"Expected '{expected_title}', got '{section['title']}'."
            )

    def test_section_contents_in_correct_order(self, doc_type, test_graph, mock_qdrant):
        """Each section must hold the content generated for its specific slot."""
        _, _, final, draft_contents = run_full_approval(test_graph, doc_type)

        for i, (expected_content, section) in enumerate(
            zip(draft_contents, final["final_document"])
        ):
            assert section["content"] == expected_content, (
                f"Content order mismatch at position {i} for {doc_type.upper()}. "
                f"Expected:\n  '{expected_content}'\nGot:\n  '{section['content']}'"
            )

    def test_is_complete_false_at_every_intermediate_step(
        self, doc_type, test_graph, mock_qdrant
    ):
        _, intermediates, _, _ = run_full_approval(test_graph, doc_type)
        for i, state in enumerate(intermediates):
            assert state["is_complete"] is False, (
                f"is_complete was True at intermediate step {i} for {doc_type.upper()} "
                f"— flag must only flip on the final approval."
            )

    def test_completed_state_durable_in_checkpoint(self, doc_type, test_graph, mock_qdrant):
        """is_complete=True must be readable from the SQLite checkpoint after completion."""
        doc_id, _, _, _ = run_full_approval(test_graph, doc_type)
        config = {"configurable": {"thread_id": doc_id}}

        persisted = test_graph.get_state(config).values
        assert persisted["is_complete"] is True, (
            "is_complete=True was not persisted to the SQLite checkpoint."
        )

    def test_completed_final_document_durable_in_checkpoint(
        self, doc_type, test_graph, mock_qdrant
    ):
        strategy = get_strategy(doc_type)
        doc_id, _, _, _ = run_full_approval(test_graph, doc_type)
        config = {"configurable": {"thread_id": doc_id}}

        persisted = test_graph.get_state(config).values
        assert len(persisted["final_document"]) == len(strategy.sections)

    def test_refinement_history_cleared_after_completion(
        self, doc_type, test_graph, mock_qdrant
    ):
        """
        process_approval drops each section's history as it moves through.
        After full completion, refinement_history must be empty.
        """
        _, _, final, _ = run_full_approval(test_graph, doc_type)
        assert final.get("refinement_history") == {}, (
            f"refinement_history should be empty after all sections approved, "
            f"got: {final.get('refinement_history')}"
        )

    def test_pending_action_none_after_completion(self, doc_type, test_graph, mock_qdrant):
        _, _, final, _ = run_full_approval(test_graph, doc_type)
        assert final.get("pending_action") is None

    def test_llm_called_once_per_section(self, doc_type, test_graph, mock_qdrant):
        """LLM must be called exactly N times — one per section, no extra calls."""
        strategy = get_strategy(doc_type)
        n_sections = len(strategy.sections)
        doc_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": doc_id}}
        draft_contents = [f"S{i}." for i in range(n_sections)]

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
             patch(MOCK_LLM_PATCH) as mock_llm:

            mock_llm.invoke.side_effect = [make_llm_response(c) for c in draft_contents]

            graph = test_graph
            graph.invoke(build_initial_state(doc_id, doc_type), config=config)

            for _ in range(n_sections - 1):
                graph.update_state(config, {"pending_action": "approve"})
                graph.invoke(None, config=config)

            graph.update_state(config, {"pending_action": "approve"})
            graph.invoke(None, config=config)

            assert mock_llm.invoke.call_count == n_sections, (
                f"Expected {n_sections} LLM calls for {doc_type.upper()}, "
                f"got {mock_llm.invoke.call_count}."
            )


# ── Mixed workflow: some sections refined before approval ─────────────────────

class TestHappyPathWithRefinements:
    """Verify that refinements mid-flow don't corrupt the final document order."""

    def test_refine_then_approve_all_preserves_order(self, test_graph, mock_qdrant):
        """
        SRS, 7 sections.  Section 1 is refined once before being approved.
        final_document must still contain all 7 sections in order.
        """
        from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import (
            get_strategy,
        )
        strategy = get_strategy("srs")
        n = len(strategy.sections)
        doc_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": doc_id}}

        # n sections + 1 extra call for the refinement of section 1
        total_calls = n + 1
        drafts = [f"S{i} content." for i in range(n)]
        refined_s1 = "S1 revised content."

        llm_responses = [
            make_llm_response(drafts[0]),   # section 0 initial
            make_llm_response(drafts[1]),   # section 1 initial
            make_llm_response(refined_s1),  # section 1 refined
        ] + [make_llm_response(drafts[i]) for i in range(2, n)]

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
             patch(MOCK_LLM_PATCH) as mock_llm:

            mock_llm.invoke.side_effect = llm_responses

            # Section 0: approve immediately
            test_graph.invoke(build_initial_state(doc_id, "srs"), config=config)
            test_graph.update_state(config, {"pending_action": "approve"})
            test_graph.invoke(None, config=config)

            # Section 1: refine once, then approve
            test_graph.update_state(config, {
                "pending_action": "refine",
                "feedback_notes": "More detail please.",
            })
            test_graph.invoke(None, config=config)
            test_graph.update_state(config, {"pending_action": "approve"})
            test_graph.invoke(None, config=config)

            # Sections 2 … N-1: approve directly
            for _ in range(2, n - 1):
                test_graph.update_state(config, {"pending_action": "approve"})
                test_graph.invoke(None, config=config)

            # Final section
            test_graph.update_state(config, {"pending_action": "approve"})
            final = test_graph.invoke(None, config=config)

        assert final["is_complete"] is True
        assert len(final["final_document"]) == n

        # Titles must match strategy headings in order
        for i, heading in enumerate(strategy.sections):
            assert final["final_document"][i]["title"] == heading

        # Section 1 content must be the REFINED version, not the original
        assert final["final_document"][1]["content"] == refined_s1

        # All other sections retain their original content
        assert final["final_document"][0]["content"] == drafts[0]
        for i in range(2, n):
            assert final["final_document"][i]["content"] == drafts[i]
