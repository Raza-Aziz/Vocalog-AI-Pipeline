# Vocalog-AI-Pipeline
Complete AI Pipeline for Vocalog web app.

## 📖 Documentation
- [Detailed API Explanation](API_EXPLANATION.md): Guide for integrating the AI pipeline with NestJS and Frontend.

## 🛠 Technical Setup

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Qdrant (running locally or in cloud)

### Installation
1. Clone the repository and navigate to the root:
   ```bash
   git clone https://github.com/Raza-Aziz/Vocalog-AI-Pipeline.git
   cd Vocalog-AI-Pipeline
   ```
2. Create environment file:
   ```bash
   cp .env.example .env  # Or create a .env file
   ```
3. Install dependencies:
   ```bash
   uv sync
   # OR
   pip install -r requirements.txt
   ```

### Environment Variables
Ensure your `.env` file contains the following:
```env
GROQ_API_KEY=your_groq_key
QDRANT_URL=http://localhost:6333
# Add other necessary keys
```

## 🚀 Integration & Running
The API is built with FastAPI. By default, it runs on **port 8000**.

### Start the Server
Run the following command from the root of the project:

**Bash / Linux / macOS:**
```bash
export PYTHONPATH="src"
uv run uvicorn vocalog_ai_api.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"
uv run uvicorn vocalog_ai_api.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API URL:** `http://localhost:8000`
- **Swagger Docs:** `http://localhost:8000/docs`

> [!TIP]
> When integrating with NestJS, you can set the `AI_PIPELINE_URL` environment variable to `http://localhost:8000`.

## 🔄 Code Workflow & Data Flow

### 1. Minutes of Meeting (MoM) Pipeline
The MoM generation is a linear **LangGraph** workflow:
1. **Entry:** Raw transcript segments are sent to the `/generate-mom` endpoint.
2. **Analysis Node (`generate_mom`):** Uses an LLM with structured output to extract attendees, agenda, summaries, and action items into a schema.
3. **Drafting Node (`generate_markdown_mom`):** Converts the structured data into a polished Markdown document using targeted formatting instructions.
4. **Exit:** The final Markdown is returned to the user.

### 2. Document Generation (HITL) Pipeline
This is a stateful, iterative workflow managed by a **Session Manager**:
1. **Initialization:** 
   - User provides MoM text via `/generate-document`.
   - **RAG Ingestion:** The text is split and embedded into **Qdrant** for context-aware retrieval.
   - **Generation:** Section 1 is generated and returned for review.
2. **The Feedback Loop (Human-in-the-Loop):**
   - **Approve:** Moves the current section to the "Final Document" store and triggers generation of the next section.
   - **Refine/Regenerate:** The user provides `feedback_notes`. The AI retrieves relevant context from Qdrant and regenerates the *same* section to improve it.
3. **Completion:** When all sections are approved, the session marks `is_complete: true`, and the final combined document is ready.

---

## 🏗 Project Structure
- `src/vocalog_ai_api/api`: FastAPI endpoints and schemas.
- `src/vocalog_ai_api/application`: LangGraph pipelines (MoM, Document Generation).
- `src/vocalog_ai_api/infrastructure`: LLM and Vector DB (Qdrant) integrations.
- `src/vocalog_ai_api/domain`: Prompts and core logic.


