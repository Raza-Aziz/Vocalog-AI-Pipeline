"""
SQLite connection manager for the Vocalog local test environment.

Responsibilities:
- Open a single, application-lifetime connection to vocalog_local.db
- Apply the full Prisma-mirror schema on first run (idempotent CREATE IF NOT EXISTS)
- Expose `get_conn()` for raw queries and `DB_PATH` for checkpointer initialisation
"""

import sqlite3
import os
from pathlib import Path

# Place the DB file at the project root (next to pyproject.toml / docker-compose.yml)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DB_PATH = str(_PROJECT_ROOT / "vocalog_local.db")

_SCHEMA_FILE = Path(__file__).with_name("sqlite_schema.sql")

# check_same_thread=False is required for FastAPI's async request handling
_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    """Return the singleton SQLite connection, initialising it on first call."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _apply_schema(_conn)
    return _conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    ddl = _SCHEMA_FILE.read_text(encoding="utf-8")
    # executescript requires its own transaction semantics — commit first
    conn.executescript(ddl)
    conn.commit()


def close_conn() -> None:
    """Explicitly close the connection (call on app shutdown if needed)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
