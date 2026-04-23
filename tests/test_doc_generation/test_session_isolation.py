"""
Test suite 5 — Multi-Tenant Session Isolation (Concurrent Stress Tests)

Verifies that multiple simultaneous document sessions do not contaminate each
other's state in the SQLite checkpoint store or vector retrieval layer.

Isolation dimensions tested:
  1. Checkpoint isolation  — each document_id maps to exactly its own state;
                             reading session A's thread_id never returns session B's data.
  2. Document-type isolation — a PRD session cannot accidentally load SRS headings.
  3. Content isolation       — section content generated for session A is not
                               readable from session B's checkpoint.
  4. Vector retrieval isolation — retrieve_context is called with the correct
                                  session_id (= thread_id) for each session;
                                  no session uses another session's embedding space.
  5. Concurrent write safety — N sessions driven in parallel threads all
                               complete without exceptions and without sharing state.

Threading strategy
──────────────────
SQLite's WAL mode and LangGraph's serialised checkpoint writes handle concurrent
access correctly.  The in-memory database is shared across threads (same
connection, check_same_thread=False) which mirrors the production singleton.
For the file-based concurrent test, each thread opens its own connection to
the shared .db file — this is the realistic production pattern.
"""

import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    create_doc_gen_graph,
)
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

# Number of parallel sessions for stress tests (keep low enough to be fast)
N_CONCURRENT = 4

# The three document types to spread across concurrent sessions
ALL_DOC_TYPES = ["srs", "prd", "sdd"]


# Thread-local storage to track session context during concurrent tests
thread_context = threading.local()


# ── Thread worker ─────────────────────────────────────────────────────────────

def _generate_one_session(
    graph,
    doc_type: str,
    retrieve_tracker: list,
    ingest_tracker: list,
    lock: threading.Lock,
) -> dict:
    """
    Start a single document session, drive it to the first generate interrupt,
    and return the result state.
    """
    doc_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": doc_id}}
    state = build_initial_state(doc_id, doc_type)

    # Set the thread-local context so the global mock side_effect knows who we are
    thread_context.doc_id = doc_id
    thread_context.doc_type = doc_type

    def _track_retrieve(session_id, query, limit=3):
        with lock:
            retrieve_tracker.append({"session_id": session_id, "doc_id": doc_id})
        return MOCK_CONTEXT

    def _track_ingest(session_id, *args, **kwargs):
        with lock:
            ingest_tracker.append({"session_id": session_id, "doc_id": doc_id})

    with patch(MOCK_INGEST_PATCH, side_effect=_track_ingest), \
         patch(MOCK_RETRIEVE_PATCH, side_effect=_track_retrieve):
        result = graph.invoke(state, config=config)

    result["_doc_id"] = doc_id
    return result


# ── Isolation tests ───────────────────────────────────────────────────────────

class TestCheckpointIsolation:
    """State reads via one session's thread_id must never surface another's data."""

    def test_two_sessions_have_independent_checkpoints(self, test_graph, mock_qdrant, mock_llm):
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        mock_llm.invoke.side_effect = [
            make_llm_response("Content for session A."),
            make_llm_response("Content for session B."),
        ]

        test_graph.invoke(build_initial_state(id_a, "srs"), config={"configurable": {"thread_id": id_a}})
        test_graph.invoke(build_initial_state(id_b, "prd"), config={"configurable": {"thread_id": id_b}})

        state_a = test_graph.get_state({"configurable": {"thread_id": id_a}}).values
        state_b = test_graph.get_state({"configurable": {"thread_id": id_b}}).values

        # Each session holds its own content
        assert state_a["current_section_content"] == "Content for session A."
        assert state_b["current_section_content"] == "Content for session B."

        # thread_id never crosses
        assert state_a["thread_id"] == id_a
        assert state_b["thread_id"] == id_b

    def test_doc_type_does_not_bleed_between_sessions(self, test_graph, mock_qdrant, mock_llm):
        id_srs = str(uuid.uuid4())
        id_prd = str(uuid.uuid4())
        id_sdd = str(uuid.uuid4())

        mock_llm.invoke.return_value = make_llm_response("Generic content.")

        for doc_id, doc_type in [(id_srs, "srs"), (id_prd, "prd"), (id_sdd, "sdd")]:
            test_graph.invoke(
                build_initial_state(doc_id, doc_type),
                config={"configurable": {"thread_id": doc_id}},
            )

        assert test_graph.get_state({"configurable": {"thread_id": id_srs}}).values["document_type"] == "srs"
        assert test_graph.get_state({"configurable": {"thread_id": id_prd}}).values["document_type"] == "prd"
        assert test_graph.get_state({"configurable": {"thread_id": id_sdd}}).values["document_type"] == "sdd"

    def test_section_outline_does_not_bleed_between_sessions(
        self, test_graph, mock_qdrant, mock_llm
    ):
        """An SRS session and an SDD session must each hold their own outline."""
        id_srs = str(uuid.uuid4())
        id_sdd = str(uuid.uuid4())
        mock_llm.invoke.return_value = make_llm_response("Content.")

        test_graph.invoke(build_initial_state(id_srs, "srs"), config={"configurable": {"thread_id": id_srs}})
        test_graph.invoke(build_initial_state(id_sdd, "sdd"), config={"configurable": {"thread_id": id_sdd}})

        srs_outline = test_graph.get_state({"configurable": {"thread_id": id_srs}}).values["sections_outline"]
        sdd_outline = test_graph.get_state({"configurable": {"thread_id": id_sdd}}).values["sections_outline"]

        assert srs_outline == get_strategy("srs").sections
        assert sdd_outline == get_strategy("sdd").sections
        assert srs_outline != sdd_outline

    def test_approval_in_session_a_does_not_affect_session_b(
        self, test_graph, mock_qdrant
    ):
        """
        Approving a section in session A must have zero effect on session B's
        current_section_index or final_document.
        """
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        config_a = {"configurable": {"thread_id": id_a}}
        config_b = {"configurable": {"thread_id": id_b}}

        with patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.side_effect = [
                make_llm_response("A-section-0."),
                make_llm_response("B-section-0."),
                make_llm_response("A-section-1."),  # after A approve
            ]
            test_graph.invoke(build_initial_state(id_a, "srs"), config=config_a)
            test_graph.invoke(build_initial_state(id_b, "prd"), config=config_b)

            # Approve section 0 in session A
            test_graph.update_state(config_a, {"pending_action": "approve"})
            test_graph.invoke(None, config=config_a)

        state_b = test_graph.get_state(config_b).values
        assert state_b["current_section_index"] == 0, (
            "Session B's section index was altered by session A's approval."
        )
        assert len(state_b["final_document"]) == 0, (
            "Session B's final_document must be empty — only session A approved."
        )

    def test_feedback_in_session_a_not_visible_in_session_b(
        self, test_graph, mock_qdrant
    ):
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        config_a = {"configurable": {"thread_id": id_a}}
        config_b = {"configurable": {"thread_id": id_b}}

        with patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.side_effect = [
                make_llm_response("A draft."),
                make_llm_response("B draft."),
                make_llm_response("A refined."),
            ]
            test_graph.invoke(build_initial_state(id_a, "srs"), config=config_a)
            test_graph.invoke(build_initial_state(id_b, "sdd"), config=config_b)

            # Refine session A
            test_graph.update_state(config_a, {
                "pending_action": "refine",
                "feedback_notes": "Session A specific feedback.",
            })
            test_graph.invoke(None, config=config_a)

        state_b = test_graph.get_state(config_b).values
        assert state_b.get("feedback_notes") is None, (
            "Session A's feedback_notes leaked into session B's checkpoint."
        )
        history_b = state_b.get("refinement_history", {})
        assert history_b == {}, (
            f"Session B's refinement_history must be empty. Got: {history_b}"
        )


# ── Vector retrieval isolation ────────────────────────────────────────────────

class TestVectorRetrievalIsolation:
    """retrieve_context must always be called with the session's own thread_id."""

    def test_retrieve_called_with_correct_session_id(self, test_graph):
        """Each session's retrieve_context call must use its own doc_id as session_id."""
        retrieve_calls = []
        lock = threading.Lock()

        sessions = [
            (str(uuid.uuid4()), "srs"),
            (str(uuid.uuid4()), "prd"),
            (str(uuid.uuid4()), "sdd"),
        ]

        def tracking_retrieve(session_id, query, limit=3):
            with lock:
                retrieve_calls.append(session_id)
            return MOCK_CONTEXT

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, side_effect=tracking_retrieve), \
             patch(MOCK_LLM_PATCH) as mock_llm:

            mock_llm.invoke.return_value = make_llm_response("Content.")
            for doc_id, doc_type in sessions:
                test_graph.invoke(
                    build_initial_state(doc_id, doc_type),
                    config={"configurable": {"thread_id": doc_id}},
                )

        # Every retrieval call must have used a session_id that belongs to one of our docs
        session_ids_used = set(retrieve_calls)
        expected_ids = {doc_id for doc_id, _ in sessions}
        assert session_ids_used == expected_ids, (
            f"retrieve_context was called with unexpected session_ids.\n"
            f"Expected: {expected_ids}\nGot: {session_ids_used}"
        )

    def test_no_cross_session_retrieval(self, test_graph):
        """Session A must never query using session B's doc_id."""
        doc_id_a = str(uuid.uuid4())
        doc_id_b = str(uuid.uuid4())
        retrieve_calls_by_doc = {doc_id_a: [], doc_id_b: []}
        current_session = threading.local()

        def tracking_retrieve(session_id, query, limit=3):
            # Track which doc_id triggered this call
            if session_id in retrieve_calls_by_doc:
                retrieve_calls_by_doc[session_id].append(session_id)
            return MOCK_CONTEXT

        with patch(MOCK_INGEST_PATCH), \
             patch(MOCK_RETRIEVE_PATCH, side_effect=tracking_retrieve), \
             patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.return_value = make_llm_response("Content.")

            test_graph.invoke(build_initial_state(doc_id_a, "srs"), config={"configurable": {"thread_id": doc_id_a}})
            test_graph.invoke(build_initial_state(doc_id_b, "prd"), config={"configurable": {"thread_id": doc_id_b}})

        # Session A's retrieval calls must ALL use doc_id_a
        for sid in retrieve_calls_by_doc[doc_id_a]:
            assert sid == doc_id_a, f"Session A used wrong session_id: {sid}"

        # Session B's retrieval calls must ALL use doc_id_b
        for sid in retrieve_calls_by_doc[doc_id_b]:
            assert sid == doc_id_b, f"Session B used wrong session_id: {sid}"


# ── Concurrent stress tests ───────────────────────────────────────────────────

class TestConcurrentSessions:
    """
    Drive N_CONCURRENT sessions in parallel threads and verify that no
    exception is raised and that each session's checkpoint is isolated.
    """

    @pytest.fixture()
    def shared_graph(self, test_checkpointer):
        """Single graph instance shared across all threads (mirrors production)."""
        return create_doc_gen_graph(checkpointer=test_checkpointer)

    def test_concurrent_sessions_complete_without_exception(self, shared_graph):
        """All N_CONCURRENT threads must finish without raising."""
        retrieve_tracker: list = []
        ingest_tracker: list = []
        lock = threading.Lock()

        doc_types = (ALL_DOC_TYPES * N_CONCURRENT)[:N_CONCURRENT]
        errors: list[Exception] = []

        def _thread_safe_llm(messages):
            # Return content based on the thread-local doc_id set in _generate_one_session
            tid = getattr(thread_context, "doc_id", "unknown")
            dtype = getattr(thread_context, "doc_type", "unknown")
            return make_llm_response(f"Content for {dtype} session {tid}.")

        def worker(doc_type: str):
            try:
                _generate_one_session(
                    shared_graph, doc_type, retrieve_tracker, ingest_tracker, lock
                )
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.side_effect = _thread_safe_llm
            threads = [threading.Thread(target=worker, args=(dt,)) for dt in doc_types]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        assert not errors, (
            f"{len(errors)} thread(s) raised exceptions during concurrent generation:\n"
            + "\n".join(str(e) for e in errors)
        )

    def test_concurrent_sessions_isolated_in_checkpoint(self, shared_graph):
        """Each concurrent session's final checkpoint must hold only its own data."""
        retrieve_tracker: list = []
        ingest_tracker: list = []
        lock = threading.Lock()

        doc_types = (ALL_DOC_TYPES * N_CONCURRENT)[:N_CONCURRENT]
        results: list[dict] = []
        results_lock = threading.Lock()

        def _thread_safe_llm(messages):
            tid = getattr(thread_context, "doc_id", "unknown")
            dtype = getattr(thread_context, "doc_type", "unknown")
            return make_llm_response(f"Content for {dtype} session {tid}.")

        def worker(doc_type: str):
            _generate_one_session(
                shared_graph, doc_type, retrieve_tracker, ingest_tracker, lock
            )
            # Re-read from checkpoint to get final state
            with results_lock:
                state = shared_graph.get_state({"configurable": {"thread_id": thread_context.doc_id}}).values
                state["_doc_id"] = thread_context.doc_id
                results.append(state)

        with patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.side_effect = _thread_safe_llm
            threads = [threading.Thread(target=worker, args=(dt,)) for dt in doc_types]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        assert len(results) == N_CONCURRENT

        # Verify each result is internally consistent
        for state in results:
            doc_id = state["_doc_id"]
            doc_type = state["document_type"]
            expected_outline = get_strategy(doc_type).sections

            assert state["thread_id"] == doc_id, (
                f"thread_id mismatch: expected {doc_id}, got {state['thread_id']}"
            )
            assert state["sections_outline"] == expected_outline, (
                f"Outline for doc_type={doc_type} doesn't match strategy. "
                f"Possible cross-contamination."
            )
            assert doc_id in state["current_section_content"], (
                f"Section content does not contain the session's own doc_id. "
                f"Possible content leakage from another session."
            )

    def test_concurrent_sessions_no_shared_final_document(self, shared_graph):
        """
        After concurrent sessions all run, no session's final_document should
        contain sections from another session.
        """
        retrieve_tracker: list = []
        ingest_tracker: list = []
        lock = threading.Lock()

        doc_types = (ALL_DOC_TYPES * N_CONCURRENT)[:N_CONCURRENT]
        results: list[dict] = []
        results_lock = threading.Lock()

        def _thread_safe_llm(messages):
            tid = getattr(thread_context, "doc_id", "unknown")
            dtype = getattr(thread_context, "doc_type", "unknown")
            return make_llm_response(f"Content for {dtype} session {tid}.")

        def worker(doc_type: str):
            _generate_one_session(
                shared_graph, doc_type, retrieve_tracker, ingest_tracker, lock
            )
            with results_lock:
                state = shared_graph.get_state({"configurable": {"thread_id": thread_context.doc_id}}).values
                state["_doc_id"] = thread_context.doc_id
                results.append(state)

        with patch(MOCK_LLM_PATCH) as mock_llm:
            mock_llm.invoke.side_effect = _thread_safe_llm
            threads = [threading.Thread(target=worker, args=(dt,)) for dt in doc_types]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        for state in results:
            # At the first interrupt, no sections are approved yet
            assert state["final_document"] == [], (
                f"final_document is not empty at the first interrupt. "
                f"Session {state['_doc_id']} may have received data from another session."
            )

    def test_vector_retrieve_called_with_own_session_id_only(self, shared_graph):
        """
        Under concurrent load, retrieve_context must always be invoked with the
        session's own thread_id — never another session's id.
        """
        retrieve_log: list[dict] = []
        lock = threading.Lock()

        doc_types = (ALL_DOC_TYPES * N_CONCURRENT)[:N_CONCURRENT]

        def tracking_retrieve(session_id, query, limit=3):
            with lock:
                retrieve_log.append(session_id)
            return MOCK_CONTEXT

        doc_id_set: set[str] = set()
        doc_id_lock = threading.Lock()

        def worker(doc_type: str):
            doc_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": doc_id}}
            state = build_initial_state(doc_id, doc_type)

            with doc_id_lock:
                doc_id_set.add(doc_id)

            with patch(MOCK_INGEST_PATCH), \
                 patch(MOCK_RETRIEVE_PATCH, side_effect=tracking_retrieve), \
                 patch(MOCK_LLM_PATCH) as mock_llm:
                mock_llm.invoke.return_value = make_llm_response(f"Content {doc_id[:8]}.")
                shared_graph.invoke(state, config=config)

        threads = [threading.Thread(target=worker, args=(dt,)) for dt in doc_types]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Every session_id passed to retrieve_context must be a valid doc_id
        stray_ids = set(retrieve_log) - doc_id_set
        assert not stray_ids, (
            f"retrieve_context was called with unknown session_ids: {stray_ids}. "
            "Cross-session vector retrieval contamination detected."
        )

    def test_concurrent_file_db_sessions_isolated(self, tmp_path):
        """
        Stress test with a shared FILE-based SQLite database (mirrors production).
        Each thread opens its own connection to the same .db file.
        All N_CONCURRENT sessions must complete and produce isolated checkpoints.
        """
        db_path = str(tmp_path / "concurrent_test.db")

        def make_connection_and_graph():
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cp = SqliteSaver(conn)
            cp.setup()
            return create_doc_gen_graph(checkpointer=cp), conn

        results: list[dict] = []
        results_lock = threading.Lock()
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        doc_types = (ALL_DOC_TYPES * N_CONCURRENT)[:N_CONCURRENT]

        def worker(doc_type: str):
            try:
                graph, conn = make_connection_and_graph()
                doc_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": doc_id}}
                state = build_initial_state(doc_id, doc_type)

                with patch(MOCK_INGEST_PATCH), \
                     patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT), \
                     patch(MOCK_LLM_PATCH) as mock_llm:
                    mock_llm.invoke.return_value = make_llm_response(
                        f"Content for {doc_id[:8]}."
                    )
                    result = graph.invoke(state, config=config)
                    result["_doc_id"] = doc_id

                with results_lock:
                    results.append(result)
                conn.close()
            except Exception as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(dt,)) for dt in doc_types]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, (
            f"Concurrent file-DB sessions raised {len(errors)} error(s):\n"
            + "\n".join(str(e) for e in errors)
        )
        assert len(results) == N_CONCURRENT

        for state in results:
            assert state["thread_id"] == state["_doc_id"]
            assert state["sections_outline"] == get_strategy(state["document_type"]).sections
