# Vocalog AI Pipeline API Documentation

This document provides a detailed explanation of the available API endpoints in the Vocalog AI Pipeline, designed to facilitate integration with NestJS backends and various frontends.

---

## 🚀 Overview

The Vocalog AI Pipeline is a collection of AI-driven services for meeting transcription analysis and automated document generation. It uses **FastAPI** for the API layer and **LangGraph** for orchestrating complex AI workflows.

---

## 🛠 Endpoints

### 1. Generate Minutes of Meeting (MoM)
Generates a structured, markdown-formatted Minutes of Meeting from a transcript object.

- **URL:** `/generate-mom`
- **Method:** `POST`
- **Description:** Takes a structured transcript object (matching Vocalog output) and returns a formatted Markdown string.
- **Request Body:** `TranscriptInput`
  ```json
  {
    "transcript": {
        "text": "Full transcript text...",
        "language_code": "en",
        "words": [...]
    },
    "user_id": "optional-user-id",
    "meeting_id": "optional-meeting-id"
  }
  ```
- **Success Response:**
  - **Code:** 200 OK
  - **Content:** `string` (Markdown formatted MoM)

---

### 2. Extract Action Items (Frontend Ready)
Extracts actionable tasks, owners, and deadlines for manual review.

- **URL:** `/action-items/extract-for-frontend`
- **Method:** `POST`
- **Description:** Optimized for frontend review. Returns a list of structured action items without any side-effects.
- **Request Body:** `ActionItemsForFrontendRequest`
  ```json
  {
    "transcript": { "text": "...", "words": [...] },
    "user_id": "user-1",
    "meeting_id": "meet-123"
  }
  ```
- **Success Response:** `ActionItemsForFrontendResponse`
  ```json
  {
    "session_id": "uuid-string",
    "actions": [
      {
        "assignee": "Hamza",
        "task_description": "Build the Slack client",
        "due_date": "tomorrow",
        "target_platform": "slack"
      }
    ],
    "total_count": 1
  }
  ```

---

### 3. Start Document Generation
Initializes a document generation session and generates the first section (Section 1).

- **URL:** `/generate-document`
- **Method:** `POST`
- **Description:** This is the entry point for the document generation flow. It creates a stateful session, ingests meeting minutes into a vector database (Qdrant), and returns the draft of the first section.
- **Request Body:** `DemoDocumentGenerationRequest`
  ```json
  {
    "meeting_minutes": "Structured MoM or raw text...",
    "project_id": "optional-project-uuid"
  }
  ```
- **Success Response:** `DemoSectionDraftResponse`
  ```json
  {
    "document_id": "uuid-string",
    "section_title": "Executive Summary",
    "content": "Draft content...",
    "is_complete": false,
    "refinement_count": 0,
    "message": "Review the section draft."
  }
  ```

---

### 4. Provide Feedback / Approve Section
Handles the Human-in-the-Loop (HITL) loop for document refinement and progression.

- **URL:** `/provide-feedback`
- **Method:** `POST`
- **Description:** Used to approve the current section, regenerate it with feedback, or refine it. If approved, the pipeline moves to the next section automatically.
- **Request Body:** `SectionFeedbackRequest`
  ```json
  {
    "document_id": "uuid-from-start-endpoint",
    "action": "approve" | "regenerate" | "refine",
    "feedback_notes": "Optional feedback for improvement"
  }
  ```
- **Success Response:** `DemoSectionDraftResponse`

---

### 5. Get Document Status
Retrieves the current progress of a specific document generation session.

- **URL:** `/document-status/{document_id}`
- **Method:** `GET`
- **Description:** Fetches stats like how many sections are completed and what the current section is.
- **Path Parameter:** `document_id` (string)
- **Success Response:** `DemoDocumentStatusResponse`

---

### 6. Health Check
Basic health check to ensure the API is running.

- **URL:** `/health`
- **Method:** `GET`
- **Response:** `{"status": "ok"}`

---

## 🔄 State Machine & Workflow

1. **Initialization:** Calling `/generate-document` sets up the session and context.
2. **Iteration Layer:** AI generates sections, user reviews/refines.
3. **Action Extraction:** Use `/action-items/extract-for-frontend` to get tasks from the meeting after or during the document generation process.

---

## ⚠️ Constraints & Considerations

1. **Session Persistence:** Currently stored in a local SQLite database (vocalog_local.db) for multi-session tracking.
2. **Idempotency:** The `/generate-document` endpoint will skip re-vectorizing if the session already exists in Qdrant to save costs.
3. **Input Format:** Always wrap transcripts in the `{ "transcript": { "text": "..." } }` object for maximum compatibility.
