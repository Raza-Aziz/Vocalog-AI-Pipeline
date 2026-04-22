-- ============================================================
-- Vocalog SQLite Schema — exact mirror of backend_prisma_schema.prisma
-- All UUIDs are TEXT; enums are TEXT with CHECK constraints;
-- DateTime fields are TEXT (ISO-8601); Json fields are TEXT (JSON string).
-- ============================================================

PRAGMA foreign_keys = ON;

-- -------------------------------------------------------
-- permissions  (no FK dependencies — create first)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS permissions (
    id          TEXT PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- users
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id               TEXT PRIMARY KEY,
    email            TEXT UNIQUE NOT NULL,
    name             TEXT UNIQUE,
    supabase_id      TEXT UNIQUE,
    user_type        TEXT NOT NULL DEFAULT 'INDIVIDUAL'
                         CHECK(user_type IN ('INDIVIDUAL', 'ORGANIZATION')),
    profile_complete INTEGER NOT NULL DEFAULT 0,
    avatar_url       TEXT,
    created_at       TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at       TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- organizations
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS organizations (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    slug       TEXT UNIQUE NOT NULL,
    meta       TEXT,
    created_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- memberships
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS memberships (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    role            TEXT NOT NULL DEFAULT 'MEMBER'
                        CHECK(role IN ('OWNER', 'ADMIN', 'MEMBER', 'GUEST')),
    accepted        INTEGER NOT NULL DEFAULT 0,
    invited_by_id   TEXT,
    created_at      TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(user_id, organization_id)
);

-- -------------------------------------------------------
-- invites
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS invites (
    id              TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    token           TEXT UNIQUE NOT NULL,
    email           TEXT,
    created_by_id   TEXT,
    expires_at      TEXT,
    accepted        INTEGER NOT NULL DEFAULT 0,
    accepted_by_id  TEXT,
    created_at      TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- project_templates  (no FK dependencies)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS project_templates (
    id          TEXT PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    meta        TEXT,
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- template_roles
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS template_roles (
    id          TEXT PRIMARY KEY,
    template_id TEXT NOT NULL REFERENCES project_templates(id),
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(template_id, name)
);

-- -------------------------------------------------------
-- template_role_permissions
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS template_role_permissions (
    id               TEXT PRIMARY KEY,
    template_role_id TEXT NOT NULL REFERENCES template_roles(id),
    permission_id    TEXT NOT NULL REFERENCES permissions(id),
    UNIQUE(template_role_id, permission_id)
);

-- -------------------------------------------------------
-- org_role_permissions
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS org_role_permissions (
    id              TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    role            TEXT NOT NULL CHECK(role IN ('OWNER', 'ADMIN', 'MEMBER', 'GUEST')),
    permission_id   TEXT NOT NULL REFERENCES permissions(id),
    UNIQUE(organization_id, role, permission_id)
);

-- -------------------------------------------------------
-- projects
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(id),
    name            TEXT NOT NULL,
    description     TEXT,
    template_key    TEXT,
    meta            TEXT,
    created_at      TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- project_roles
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS project_roles (
    id         TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(project_id, name)
);

-- -------------------------------------------------------
-- project_role_permissions
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS project_role_permissions (
    id              TEXT PRIMARY KEY,
    project_role_id TEXT NOT NULL REFERENCES project_roles(id),
    permission_id   TEXT NOT NULL REFERENCES permissions(id),
    UNIQUE(project_role_id, permission_id)
);

-- -------------------------------------------------------
-- project_members
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS project_members (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES users(id),
    role            TEXT NOT NULL DEFAULT 'MEMBER'
                        CHECK(role IN ('OWNER', 'ADMIN', 'MEMBER')),
    project_role_id TEXT REFERENCES project_roles(id),
    joined_at       TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(project_id, user_id)
);

-- -------------------------------------------------------
-- meetings
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS meetings (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    ai_mom        TEXT,
    scheduled_at  TEXT,
    duration_min  INTEGER,
    type          TEXT NOT NULL DEFAULT 'ONLINE'
                      CHECK(type IN ('ONLINE', 'PHYSICAL', 'HYBRID')),
    platform      TEXT NOT NULL DEFAULT 'MOBILE'
                      CHECK(platform IN ('MOBILE', 'VIRTUAL')),
    status        TEXT NOT NULL DEFAULT 'SCHEDULED'
                      CHECK(status IN ('SCHEDULED', 'RECORDED', 'PROCESSING',
                                       'DRAFT_READY', 'FINALIZED', 'FAILED')),
    audio_url     TEXT,
    transcript    TEXT,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_by_id TEXT NOT NULL REFERENCES users(id),
    created_at    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    deleted       INTEGER NOT NULL DEFAULT 0
);

-- -------------------------------------------------------
-- virtual_meeting_sessions
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS virtual_meeting_sessions (
    id                   TEXT PRIMARY KEY,
    provider             TEXT NOT NULL DEFAULT 'GOOGLE_MEET'
                             CHECK(provider IN ('GOOGLE_MEET')),
    meeting_url          TEXT NOT NULL,
    title                TEXT,
    topic                TEXT,
    log_type             TEXT NOT NULL DEFAULT 'Meeting Minutes',
    language             TEXT NOT NULL DEFAULT 'auto',
    status               TEXT NOT NULL DEFAULT 'JOINING'
                             CHECK(status IN ('JOINING', 'RECORDING', 'STOPPING',
                                              'PROCESSING', 'COMPLETED', 'FAILED')),
    participant_names    TEXT,
    failure_reason       TEXT,
    joined_at            TEXT,
    recording_started_at TEXT,
    end_requested_at     TEXT,
    ended_at             TEXT,
    project_id           TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    started_by_id        TEXT NOT NULL REFERENCES users(id),
    meeting_id           TEXT UNIQUE REFERENCES meetings(id),
    created_at           TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at           TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_vms_project_status
    ON virtual_meeting_sessions(project_id, status);

-- -------------------------------------------------------
-- meeting_participants
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS meeting_participants (
    id           TEXT PRIMARY KEY,
    meeting_id   TEXT NOT NULL REFERENCES meetings(id),
    user_id      TEXT REFERENCES users(id),
    email        TEXT,
    display_name TEXT,
    is_host      INTEGER NOT NULL DEFAULT 0,
    joined_at    TEXT,
    left_at      TEXT,
    created_at   TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_meeting_participants_meeting_id
    ON meeting_participants(meeting_id);

-- -------------------------------------------------------
-- meeting_minutes
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS meeting_minutes (
    id               TEXT PRIMARY KEY,
    meeting_id       TEXT UNIQUE NOT NULL REFERENCES meetings(id),
    agenda           TEXT,
    attendees        TEXT,
    discussion_points TEXT,
    summary          TEXT,
    created_at       TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at       TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- action_items
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS action_items (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT,
    meeting_id  TEXT NOT NULL REFERENCES meetings(id),
    assignee_id TEXT REFERENCES users(id),
    status      TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK(status IN ('PENDING', 'IN_PROGRESS', 'DONE')),
    due_date    TEXT,
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- documents
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    content       TEXT,
    type          TEXT NOT NULL DEFAULT 'CUSTOM'
                      CHECK(type IN ('SRS', 'DESIGN_DOC', 'PRD', 'CUSTOM')),
    status        TEXT NOT NULL DEFAULT 'DRAFT'
                      CHECK(status IN ('DRAFT', 'IN_PROGRESS', 'REVIEW', 'FINALIZED')),
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_by_id TEXT NOT NULL REFERENCES users(id),
    meta          TEXT,
    created_at    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- google_calendar_integrations
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS google_calendar_integrations (
    id                      TEXT PRIMARY KEY,
    user_id                 TEXT UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    google_email            TEXT NOT NULL,
    encrypted_refresh_token TEXT NOT NULL,
    encrypted_access_token  TEXT,
    token_expiry            TEXT,
    calendar_sync_enabled   INTEGER NOT NULL DEFAULT 1,
    default_project_id      TEXT,
    created_at              TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at              TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- -------------------------------------------------------
-- calendar_event_bot_sessions
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS calendar_event_bot_sessions (
    id                            TEXT PRIMARY KEY,
    google_calendar_integration_id TEXT NOT NULL
                                      REFERENCES google_calendar_integrations(id) ON DELETE CASCADE,
    google_event_id               TEXT NOT NULL,
    virtual_meeting_session_id    TEXT,
    meeting_start_time            TEXT NOT NULL,
    created_at                    TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(google_calendar_integration_id, google_event_id)
);
