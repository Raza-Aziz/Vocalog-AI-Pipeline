# Vocalog AI Pipeline API Documentation

This document provides a detailed explanation of the available API endpoints in the Vocalog AI Pipeline, designed to facilitate integration with NestJS backends and various frontends.

---

## 🚀 Overview

The Vocalog AI Pipeline is a collection of AI-driven services for meeting transcription analysis and automated document generation. It uses **FastAPI** for the API layer and **LangGraph** for orchestrating complex AI workflows.

---

## 🛠 Endpoints

### 1. Generate Minutes of Meeting (MoM)
Generates a structured, markdown-formatted Minutes of Meeting from a raw transcript.

- **URL:** `/generate-mom`
- **Method:** `POST`
- **Description:** Takes a raw transcript (dictionary format) and returns a formatted Markdown string.
- **Request Body:** `TranscriptInput`
  ```json
  {
    "raw_transcript": {
        "segments": [...],
        "text": "Full transcript text..."
    }
  }
  ```
- **Success Response:**
  - **Code:** 200 OK
  - **Content:** `string` (Markdown formatted MoM)
- **Example Usage (cURL):**
  ```bash
  curl -X POST "http://localhost:8000/generate-mom" \
       -H "Content-Type: application/json" \
       -d '{"raw_transcript": {"text": "Meeting about project X..."}}'
  ```

---

### 2. Start Document Generation
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

### 3. Provide Feedback / Approve Section
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
  - Returns the draft of the **next** section if `action` was `"approve"`.
  - Returns a **revised** version of the current section if `action` was `"regenerate"` or `"refine"`.
  - When `is_complete` is `true`, the generation process is finished.

---

### 4. Get Document Status
Retrieves the current progress of a specific document generation session.

- **URL:** `/document-status/{document_id}`
- **Method:** `GET`
- **Description:** Fetches stats like how many sections are completed and what the current section is.
- **Path Parameter:** `document_id` (string)
- **Success Response:** `DemoDocumentStatusResponse`
  ```json
  {
    "document_id": "uuid-string",
    "status": "in_progress" | "completed" | "error",
    "current_section_title": "Technical Architecture",
    "completed_sections": 2,
    "total_sections": 5
  }
  ```

---

### 5. Health Check
Basic health check to ensure the API is running.

- **URL:** `/health`
- **Method:** `GET`
- **Response:** `{"status": "ok"}`

---

## 🔄 State Machine & Workflow

The document generation process (Endpoints 2 & 3) follows a stateful multi-step workflow:

1. **Initialization:** Calling `/generate-document` sets up the session and context.
2. **Iteration Layer:**
   - The AI generates a section draft.
   - The user (Frontend/NestJS) reviews the draft.
   - User provides feedback via `/provide-feedback`.
     - **Approve:** Move to the next section.
     - **Refine/Regenerate:** AI improves the current section based on `feedback_notes`.
3. **Completion:** Once all sections defined in the internal outline are approved, `is_complete` becomes `true`.

---

## ⚠️ Constraints & Considerations

1. **Session Persistence:** Currently, sessions are stored in an in-memory `session_manager`. If the Python API restarts, active sessions will be lost. Ensure NestJS handles session ID storage or implement a persistent database (Redis/PostgreSQL) if needed for long-running generations.
2. **Context Limits:** Large transcripts may challenge LLM context windows. The pipeline uses RAG (via Qdrant) to manage this, but concise MoMs are recommended as input for `/generate-document`.
3. **Concurrency:** The current implementation is optimized for the demo. For high-concurrency production environments, consider moving LangGraph executions to a background worker (Celery/Temporal).
4. **Input Format:** The `/generate-mom` endpoint expects a dictionary in `raw_transcript`. If sending plain text, wrap it in an object: `{"raw_transcript": {"text": "..."}}`.
