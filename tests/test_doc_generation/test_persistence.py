"""
Test suite 2 — Thread-Based Persistence Integrity (Kill-and-Resume)

Simulates a server crash or restart mid-session and verifies that the SQLite
checkpointer correctly preserves the complete graph state so that a newly
initialised graph instance can rehydrate it using only the document_id (thread_id).

A "crash" is modelled by:
  1. Committing the first section draft to a FILE-based SQLite database.
  2. Closing and deleting the first graph + connection object (simulating process death).
  3. Opening a brand-new connection to the same database file.
  4. Creating a second graph instance with the new connection.
  5. Calling graph.get_state(config) with only the original document_id.

CRITICAL FAILURE conditions are explicitly marked in assertion messages so that
CI logs make the root cause immediately obvious.
"""

import sqlite3
import uuid
from unittest.mock import patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    create_doc_gen_graph,
)
from tests.test_doc_generation.conftest import (
    build_initial_state,
    make_llm_response,
    MOCK_LLM_PATCH,
    MOCK_INGEST_PATCH,
    MOCK_RETRIEVE_PATCH,
    MOCK_CONTEXT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_file_checkpointer(db_path: str) -> SqliteSaver:
    """Open a new file-based connection and return a ready-to-use SqliteSaver."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cp = SqliteSaver(conn)
    cp.setup()
    return cp, conn


def _run_first_draft(graph, doc_id: str, doc_type: str, draft_content: str) -> dict:
    """
    Invoke the graph from scratch, returning the result state after the first
    generate interrupt.  Uses patched LLM + Qdrant.
    """
    config = {"configurable": {"thread_id": doc_id}}
    state = build_initial_state(doc_id, doc_type)

    with patch(MOCK_INGEST_PATCH), \
         patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
         patch(MOCK_LLM_PATCH) as mock_llm:
        mock_llm.invoke.return_value = make_llm_response(draft_content)
        return graph.invoke(state, config=config)


# ── Test class ────────────────────────────────────────────────────────────────

class TestKillAndResume:
    """Each test uses a tmp_path-scoped file database — not the production DB."""

    @pytest.fixture()
    def db_path(self, tmp_path) -> str:
        return str(tmp_path / "checkpoint.db")

    # ── Core kill-and-resume ─────────────────────────────────────────────────

    @pytest.mark.parametrize("doc_type", ["srs", "prd", "sdd"])
    def test_state_rehydrates_after_simulated_crash(self, db_path, doc_type):
        """
        After the first graph process is 'killed', a new process using only
        the document_id must recover the identical state from the database.
        """
        doc_id = str(uuid.uuid4())
        draft_content = f"Original {doc_type.upper()} draft — written before the crash."

        # ── Process 1: generate first draft ──
        cp1, conn1 = _make_file_checkpointer(db_path)
        graph1 = create_doc_gen_graph(checkpointer=cp1)
        result1 = _run_first_draft(graph1, doc_id, doc_type, draft_content)

        original_content = result1["current_section_content"]
        assert original_content, "Pre-condition: first draft must be non-empty"

        # CRASH — close and discard everything from process 1
        conn1.close()
        del graph1, cp1, conn1

        # ── Process 2: open new connection to same file ──
        cp2, conn2 = _make_file_checkpointer(db_path)
        graph2 = create_doc_gen_graph(checkpointer=cp2)
        config = {"configurable": {"thread_id": doc_id}}

        checkpoint = graph2.get_state(config)

        # ── Assertions ───────────────────────────────────────────────────────
        assert checkpoint is not None, (
            "CRITICAL FAILURE: get_state() returned None — "
            "state was NOT persisted to the SQLite checkpointer."
        )
        assert checkpoint.values is not None, (
            "CRITICAL FAILURE: checkpoint.values is None — "
            "checkpoint record exists but holds no state data."
        )

        state = checkpoint.values
        assert state["document_type"] == doc_type, (
            f"CRITICAL FAILURE: document_type mismatch after resume. "
            f"Expected '{doc_type}', got '{state.get('document_type')}'."
        )
        assert state["thread_id"] == doc_id, (
            f"CRITICAL FAILURE: thread_id mismatch. "
            f"Expected '{doc_id}', got '{state.get('thread_id')}'."
        )
        assert state["current_section_content"] == original_content, (
            "CRITICAL FAILURE: section content changed after resume. "
            f"Expected:\n{original_content}\nGot:\n{state.get('current_section_content')}"
        )
        assert state["current_section_index"] == 0, (
            "CRITICAL FAILURE: section index must be 0 after first draft."
        )
        assert state["is_complete"] is False, (
            "CRITICAL FAILURE: is_complete must be False after first draft."
        )

        conn2.close()

    def test_sections_outline_persisted(self, db_path):
        """The full sections_outline (from strategy) must survive the restart."""
        from vocalog_ai_api.application.pipelines.doc_generation_pipeline.strategies import (
            get_strategy,
        )

        doc_id = str(uuid.uuid4())
        doc_type = "srs"
        expected_outline = get_strategy(doc_type).sections

        cp1, conn1 = _make_file_checkpointer(db_path)
        graph1 = create_doc_gen_graph(checkpointer=cp1)
        _run_first_draft(graph1, doc_id, doc_type, "Draft content.")
        conn1.close()
        del graph1, cp1, conn1

        cp2, conn2 = _make_file_checkpointer(db_path)
        graph2 = create_doc_gen_graph(checkpointer=cp2)
        state = graph2.get_state({"configurable": {"thread_id": doc_id}}).values

        assert state["sections_outline"] == expected_outline, (
            "CRITICAL FAILURE: sections_outline changed after restart. "
            f"Expected {expected_outline}, got {state.get('sections_outline')}."
        )
        conn2.close()

    def test_project_id_persisted(self, db_path):
        """Metadata fields (project_id) must survive a crash."""
        doc_id = str(uuid.uuid4())
        project_id = "my-test-project"

        cp1, conn1 = _make_file_checkpointer(db_path)
        graph1 = create_doc_gen_graph(checkpointer=cp1)
        config = {"configurable": {"thread_id": doc_id}}
        state = build_initial_state(doc_id, "prd")
        state["project_id"] = project_id

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
             patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.return_value = make_llm_response("Draft.")
            graph1.invoke(state, config=config)

        conn1.close()
        del graph1, cp1, conn1

        cp2, conn2 = _make_file_checkpointer(db_path)
        graph2 = create_doc_gen_graph(checkpointer=cp2)
        recovered = graph2.get_state(config).values

        assert recovered["project_id"] == project_id
        conn2.close()

    def test_missing_document_id_returns_none(self, db_path):
        """
        Querying a document_id that never existed must return a checkpoint whose
        values are None (or an empty snapshot) — never raise an exception.
        """
        cp, conn = _make_file_checkpointer(db_path)
        graph = create_doc_gen_graph(checkpointer=cp)
        phantom_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": phantom_id}}

        checkpoint = graph.get_state(config)
        # LangGraph returns a StateSnapshot with empty values for unknown thread_ids
        assert checkpoint is not None  # The call itself must not raise
        assert not checkpoint.values, (
            f"Expected no state for unknown document_id '{phantom_id}', "
            f"got: {checkpoint.values}"
        )
        conn.close()

    # ── Refinement history persistence ───────────────────────────────────────

    def test_refinement_history_persisted_across_restart(self, db_path):
        """
        After a refine action, the refinement_history entry for section 0
        must be recoverable from the database.
        """
        doc_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": doc_id}}

        cp1, conn1 = _make_file_checkpointer(db_path)
        graph1 = create_doc_gen_graph(checkpointer=cp1)

        # Start generation
        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
             patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.side_effect = [
                make_llm_response("First draft."),
                make_llm_response("Refined draft."),
            ]

            # First draft
            graph1.invoke(build_initial_state(doc_id, "srs"), config=config)

            # Refine section 0
            graph1.update_state(config, {
                "pending_action": "refine",
                "feedback_notes": "Please add more detail.",
            })
            graph1.invoke(None, config=config)

        conn1.close()
        del graph1, cp1, conn1

        # Recover state in a new process
        cp2, conn2 = _make_file_checkpointer(db_path)
        graph2 = create_doc_gen_graph(checkpointer=cp2)
        state = graph2.get_state(config).values

        history = state.get("refinement_history", {})
        assert "0" in history, (
            "CRITICAL FAILURE: refinement_history for section 0 not found after restart."
        )
        assert len(history["0"]) == 1
        assert history["0"][0]["feedback"] == "Please add more detail."
        conn2.close()

    # ── Multiple documents survive independently ──────────────────────────────

    def test_two_documents_both_recoverable(self, db_path):
        """
        Two independent documents written to the same file database must both
        be independently recoverable after restart.
        """
        id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
        content_a = "Document A — executive summary draft."
        content_b = "Document B — architecture overview draft."

        cp1, conn1 = _make_file_checkpointer(db_path)
        graph1 = create_doc_gen_graph(checkpointer=cp1)

        _run_first_draft(graph1, id_a, "prd", content_a)
        _run_first_draft(graph1, id_b, "sdd", content_b)

        conn1.close()
        del graph1, cp1, conn1

        cp2, conn2 = _make_file_checkpointer(db_path)
        graph2 = create_doc_gen_graph(checkpointer=cp2)

        state_a = graph2.get_state({"configurable": {"thread_id": id_a}}).values
        state_b = graph2.get_state({"configurable": {"thread_id": id_b}}).values

        assert state_a["document_type"] == "prd"
        assert state_a["current_section_content"] == content_a

        assert state_b["document_type"] == "sdd"
        assert state_b["current_section_content"] == content_b

        conn2.close()
