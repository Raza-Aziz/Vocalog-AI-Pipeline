"""
Test suite 3 — Human-in-the-Loop Logic Flow

Verifies the three user actions (approve / refine / regenerate) against the
LangGraph HITL interrupt model:

  Action       Expected behaviour
  ─────────    ──────────────────────────────────────────────────────────────
  approve      current section moves to final_document; index advances;
               pending_action + feedback_notes are cleared in checkpoint.
  refine       same section is regenerated with feedback_notes injected into
               the LLM prompt; refinement_history gains one entry;
               pending_action + feedback_notes are cleared after resume.
  regenerate   same section is regenerated fresh (no feedback_notes sent);
               refinement_history is NOT updated; fields are cleared.

All tests use the in-memory checkpointer fixture so they run without a
production database and are fully isolated from each other.
"""

import uuid
from unittest.mock import patch, call

import pytest

from tests.test_doc_generation.conftest import (
    build_initial_state,
    make_llm_response,
    MOCK_LLM_PATCH,
    MOCK_INGEST_PATCH,
    MOCK_RETRIEVE_PATCH,
    MOCK_CONTEXT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _start_session(graph, doc_type: str, first_draft: str):
    """
    Initialise a new document session and run until the first generate interrupt.
    Returns (doc_id, config, state_after_first_draft).
    """
    doc_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": doc_id}}
    initial = build_initial_state(doc_id, doc_type)

    with patch(MOCK_INGEST_PATCH), \
         patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
         patch(MOCK_LLM_PATCH) as mock_llm:
        mock_llm.invoke.return_value = make_llm_response(first_draft)
        state = graph.invoke(initial, config=config)

    return doc_id, config, state


def _resume(graph, config, action: str, feedback: str | None, *llm_drafts: str):
    """
    Inject pending_action (and optionally feedback_notes) into the checkpoint,
    then resume the graph.  Returns the result state.
    """
    update = {"pending_action": action, "feedback_notes": feedback}
    graph.update_state(config, update)

    with patch(MOCK_INGEST_PATCH), \
         patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
         patch(MOCK_LLM_PATCH) as mock_llm:
        mock_llm.invoke.side_effect = [make_llm_response(d) for d in llm_drafts]
        return graph.invoke(None, config=config)


# ── Approve action ────────────────────────────────────────────────────────────

class TestApproveAction:

    def test_approve_moves_section_to_final_document(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "Draft of section 0.")
        result = _resume(test_graph, config, "approve", None, "Draft of section 1.")

        assert len(result["final_document"]) == 1, (
            "After approving section 0 there must be exactly one finalised section."
        )
        assert result["final_document"][0]["content"] == "Draft of section 0."

    def test_approve_advances_section_index(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "prd", "Draft 0.")
        result = _resume(test_graph, config, "approve", None, "Draft 1.")
        assert result["current_section_index"] == 1

    def test_approve_clears_pending_action_in_checkpoint(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "sdd", "Draft 0.")
        result = _resume(test_graph, config, "approve", None, "Draft 1.")

        assert result.get("pending_action") is None, (
            "pending_action must be None after graph resumes — "
            f"got: {result.get('pending_action')}"
        )

    def test_approve_clears_feedback_notes_in_checkpoint(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "Draft 0.")
        # Inject a stray feedback_notes alongside approve (edge case)
        result = _resume(test_graph, config, "approve", "Stale note", "Draft 1.")
        assert result.get("feedback_notes") is None

    def test_approve_title_matches_strategy_heading(self, test_graph, mock_qdrant):
        from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import (
            get_strategy,
        )
        strategy = get_strategy("srs")
        _, config, _ = _start_session(test_graph, "srs", "Draft 0.")
        _resume(test_graph, config, "approve", None, "Draft 1.")

        checkpoint = test_graph.get_state(config)
        final_doc = checkpoint.values["final_document"]
        assert final_doc[0]["title"] == strategy.sections[0]

    def test_approve_generates_next_section_draft(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "Draft 0.")
        result = _resume(test_graph, config, "approve", None, "Draft 1 freshly generated.")
        assert result["current_section_content"] == "Draft 1 freshly generated."

    def test_double_approve_moves_two_sections(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "prd", "Draft 0.")
        _resume(test_graph, config, "approve", None, "Draft 1.")
        result = _resume(test_graph, config, "approve", None, "Draft 2.")

        assert len(result["final_document"]) == 2
        assert result["final_document"][0]["content"] == "Draft 0."
        assert result["final_document"][1]["content"] == "Draft 1."
        assert result["current_section_index"] == 2


# ── Refine action ─────────────────────────────────────────────────────────────

class TestRefineAction:

    def test_refine_stays_on_same_section(self, test_graph, mock_qdrant):
        _, config, before = _start_session(test_graph, "srs", "Initial draft.")
        result = _resume(test_graph, config, "refine", "Add more detail.", "Refined draft.")
        assert result["current_section_index"] == before["current_section_index"]

    def test_refine_updates_section_content(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "prd", "Initial draft.")
        result = _resume(test_graph, config, "refine", "Focus on user value.", "Refined PRD content.")
        assert result["current_section_content"] == "Refined PRD content."

    def test_refine_does_not_advance_final_document(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "Draft.")
        result = _resume(test_graph, config, "refine", "Some feedback.", "Refined.")
        assert len(result["final_document"]) == 0

    def test_refine_records_previous_draft_in_history(self, test_graph, mock_qdrant):
        _, config, initial = _start_session(test_graph, "sdd", "Original draft.")
        _resume(test_graph, config, "refine", "Be more technical.", "Revised draft.")

        checkpoint = test_graph.get_state(config)
        history = checkpoint.values.get("refinement_history", {})

        assert "0" in history, "refinement_history must have an entry for section 0."
        assert len(history["0"]) == 1
        assert history["0"][0]["draft"] == "Original draft."
        assert history["0"][0]["feedback"] == "Be more technical."

    def test_refine_twice_accumulates_history(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "Draft v1.")
        _resume(test_graph, config, "refine", "First feedback.", "Draft v2.")
        _resume(test_graph, config, "refine", "Second feedback.", "Draft v3.")

        history = test_graph.get_state(config).values.get("refinement_history", {})
        assert len(history["0"]) == 2
        assert history["0"][0]["feedback"] == "First feedback."
        assert history["0"][1]["feedback"] == "Second feedback."

    def test_refine_clears_pending_action_after_resume(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "Draft.")
        result = _resume(test_graph, config, "refine", "Notes.", "Refined.")
        assert result.get("pending_action") is None

    def test_refine_clears_feedback_notes_after_resume(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "prd", "Draft.")
        result = _resume(test_graph, config, "refine", "Notes to incorporate.", "Refined.")
        assert result.get("feedback_notes") is None

    def test_refine_feedback_injected_into_llm_prompt(self, test_graph, mock_qdrant):
        """The feedback_notes string must appear in the refinement prompt sent to the LLM."""
        _, config, _ = _start_session(test_graph, "sdd", "Original architecture draft.")
        feedback_text = "Please include justification for the microservices choice."

        update = {"pending_action": "refine", "feedback_notes": feedback_text}
        test_graph.update_state(config, update)

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
             patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.return_value = make_llm_response("Revised draft.")
            test_graph.invoke(None, config=config)

            prompt = mock_llm.invoke.call_args_list[0][0][0][0].content
            assert feedback_text in prompt, (
                f"Feedback notes not found in LLM prompt.\n"
                f"Feedback: '{feedback_text}'\nPrompt excerpt: {prompt[:600]}"
            )


# ── Regenerate action ─────────────────────────────────────────────────────────

class TestRegenerateAction:

    def test_regenerate_stays_on_same_section(self, test_graph, mock_qdrant):
        _, config, before = _start_session(test_graph, "srs", "Draft 1.")
        result = _resume(test_graph, config, "regenerate", None, "Fresh attempt.")
        assert result["current_section_index"] == before["current_section_index"]

    def test_regenerate_updates_section_content(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "prd", "Stale draft.")
        result = _resume(test_graph, config, "regenerate", None, "Brand new draft.")
        assert result["current_section_content"] == "Brand new draft."

    def test_regenerate_does_not_add_history_entry(self, test_graph, mock_qdrant):
        """
        Regenerate is a fresh attempt without specific feedback — it must NOT
        record an entry in refinement_history (history is only for 'refine').
        """
        _, config, _ = _start_session(test_graph, "srs", "Draft 1.")
        _resume(test_graph, config, "regenerate", None, "Draft 2.")

        history = test_graph.get_state(config).values.get("refinement_history", {})
        # Section 0 history should be absent or empty
        assert history.get("0", []) == [], (
            "refinement_history must NOT be updated on a regenerate action. "
            f"Got: {history}"
        )

    def test_regenerate_clears_pending_action(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "sdd", "First attempt.")
        result = _resume(test_graph, config, "regenerate", None, "Second attempt.")
        assert result.get("pending_action") is None

    def test_regenerate_does_not_use_feedback_notes_in_prompt(self, test_graph, mock_qdrant):
        """The LLM must receive an initial-draft prompt (no 'Reviewer Feedback' block)."""
        _, config, _ = _start_session(test_graph, "srs", "Draft.")
        update = {"pending_action": "regenerate", "feedback_notes": None}
        test_graph.update_state(config, update)

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
             patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.return_value = make_llm_response("Regenerated draft.")
            test_graph.invoke(None, config=config)

            prompt = mock_llm.invoke.call_args_list[0][0][0][0].content
            assert "Reviewer Feedback" not in prompt, (
                "Regenerate must use an initial-draft prompt, not a refinement prompt."
            )


# ── Checkpoint field integrity after each action ──────────────────────────────

class TestCheckpointFieldIntegrity:
    """
    Cross-cutting: verify that the checkpoint stored in SQLite (not just the
    return value of invoke()) has the correct field values after each action.
    These tests read back via get_state() to confirm the DB reflects reality.
    """

    def test_approve_checkpoint_index_incremented(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "srs", "D0.")
        _resume(test_graph, config, "approve", None, "D1.")
        saved = test_graph.get_state(config).values
        assert saved["current_section_index"] == 1

    def test_refine_checkpoint_history_updated(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "prd", "D0.")
        _resume(test_graph, config, "refine", "My notes.", "D0 refined.")
        saved = test_graph.get_state(config).values
        assert saved["refinement_history"].get("0")

    def test_refine_checkpoint_pending_action_none(self, test_graph, mock_qdrant):
        _, config, _ = _start_session(test_graph, "sdd", "D0.")
        _resume(test_graph, config, "refine", "Notes.", "D0 refined.")
        saved = test_graph.get_state(config).values
        assert saved["pending_action"] is None

    def test_approve_then_refine_checkpoint_correct(self, test_graph, mock_qdrant):
        """Mixed action sequence: approve section 0, then refine section 1."""
        _, config, _ = _start_session(test_graph, "srs", "S0 draft.")
        _resume(test_graph, config, "approve", None, "S1 draft.")
        _resume(test_graph, config, "refine", "Improve section 1.", "S1 revised.")

        saved = test_graph.get_state(config).values
        # Section 0 should be in final_document
        assert len(saved["final_document"]) == 1
        assert saved["final_document"][0]["content"] == "S0 draft."
        # Still on section 1
        assert saved["current_section_index"] == 1
        # History for section 1 recorded
        assert "1" in saved["refinement_history"]
        # HITL fields cleared
        assert saved["pending_action"] is None
        assert saved["feedback_notes"] is None
