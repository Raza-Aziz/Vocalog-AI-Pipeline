"""
Shared fixtures for the document-generation integration test suite.

Design principles
─────────────────
• Every test gets its own in-memory SQLite checkpointer — zero cross-test pollution.
• The LLM (Groq) is always mocked — no real API calls, deterministic prompts.
• Qdrant is always mocked — tests run without a live vector-store process.
• Only the SQLite checkpointer layer is exercised with real I/O so that
  persistence behaviour is tested faithfully.

Patch paths
───────────
All patches target the symbols as they are imported in nodes.py, not where they
are defined.  This is the correct unittest.mock pattern.
"""

import sqlite3
import uuid
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    create_doc_gen_graph,
)

# ── Constants ────────────────────────────────────────────────────────────────

MOCK_LLM_PATCH = (
    "vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes.llm"
)
MOCK_INGEST_PATCH = (
    "vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes.ingest_minutes"
)
MOCK_RETRIEVE_PATCH = (
    "vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes.retrieve_context"
)
MOCK_CONTEXT = "Relevant meeting context: the team discussed OAuth 2.0 and PostgreSQL."

# Realistic multi-speaker meeting transcript for RAG ingestion tests
SAMPLE_MINUTES = """\
Project Kickoff — Q4 2024
Attendees: Alice (PM), Bob (Lead Engineer), Carol (UX Designer)

Alice: Today we're aligning on the new authentication service.
Bob: I propose we go with OAuth 2.0 — Google and GitHub as initial providers.
      The backend will be FastAPI with JWT access tokens and refresh-token rotation.
Carol: I've drafted wireframes for the login and signup flows. Mobile-first.
Alice: Great. Let's target December 15 for the initial release.
Bob: We'll need PostgreSQL for user storage. I'll set up the schema by November 8.
Carol: UI mockups will be ready November 5.
Alice: I'll have the user personas done by November 1.
Bob: One constraint — we must implement rate limiting on all auth endpoints.
Alice: Agreed. Logging to our centralised observability stack as well.
"""


# ── Primitive helpers ────────────────────────────────────────────────────────

def build_initial_state(doc_id: str, doc_type: str, minutes: str = SAMPLE_MINUTES) -> dict:
    """Construct a fresh DocumentGenerationState dict for graph invocation."""
    return {
        "thread_id": doc_id,
        "project_id": "test-project",
        "document_type": doc_type,
        "meeting_minutes": minutes,
        "sections_outline": [],
        "current_section_index": 0,
        "current_section_content": "",
        "pending_action": None,
        "feedback_notes": None,
        "refinement_history": {},
        "final_document": [],
        "is_complete": False,
    }


def make_llm_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


# ── pytest fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def sample_minutes() -> str:
    return SAMPLE_MINUTES


@pytest.fixture()
def test_checkpointer() -> SqliteSaver:
    """
    Fresh in-memory SqliteSaver per test.
    Using :memory: guarantees zero bleed between tests while still exercising
    the real LangGraph checkpoint read/write code paths.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cp = SqliteSaver(conn)
    cp.setup()
    return cp


@pytest.fixture()
def test_graph(test_checkpointer: SqliteSaver):
    """Compiled doc-gen graph wired to the per-test in-memory checkpointer."""
    return create_doc_gen_graph(checkpointer=test_checkpointer)


@pytest.fixture()
def mock_llm():
    """
    Patches the LLM singleton in nodes.py.
    Yields the mock object so individual tests can customise side_effect.
    Default: single call returns a generic string.
    """
    with patch(MOCK_LLM_PATCH) as mock:
        mock.invoke.return_value = make_llm_response(
            "Default mock section content generated for testing."
        )
        yield mock


@pytest.fixture()
def mock_qdrant():
    """
    Patches both Qdrant-dependent functions used by nodes.py:
      • ingest_minutes  → no-op (avoids needing a live Qdrant process)
      • retrieve_context → returns a fixed, deterministic context string
    Yields (mock_ingest, mock_retrieve) so tests can assert on call args.
    """
    with patch(MOCK_INGEST_PATCH) as mock_ingest, \
         patch(MOCK_RETRIEVE_PATCH, return_value=MOCK_CONTEXT) as mock_retrieve:
        yield mock_ingest, mock_retrieve


@pytest.fixture()
def initial_state_factory() -> Callable:
    """
    Returns a factory callable: (doc_type, doc_id=None) → (state_dict, doc_id).
    Generates a fresh UUID if doc_id is not supplied.
    """
    def _factory(doc_type: str, doc_id: str | None = None):
        _id = doc_id or str(uuid.uuid4())
        return build_initial_state(_id, doc_type), _id
    return _factory
