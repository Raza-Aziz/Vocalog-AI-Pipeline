import time
from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState

# --- NEW IMPORTS ---
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import (
    ingest_minutes,
    retrieve_context
)

from src.vocalog_ai_api.infrastructure.llm_providers.groq import llm

# Toggle this for real LLM vs Fast Demo
MOCK_MODE = False  # Set to False to test real Qdrant + OpenAI interaction

# llm = ChatOpenAI(model="gpt-4o", temperature=0.7) if not MOCK_MODE else None

def initialize_document(state: DocumentGenerationState) -> DocumentGenerationState:
    """
    1. Ingests raw minutes into Qdrant Vector Store.
    2. Creates the SRS outline.
    """
    print("--- Initializing Document & Vectorizing Data ---")
    
    session_id = state["session_id"]
    raw_minutes = state["meeting_minutes"]

    # --- STEP 1: Vector Ingestion ---
    # We only want to do this once. In a real app, you might check if it already exists.
    try:
        ingest_minutes(session_id, raw_minutes)
        print(f"Successfully vectorized minutes for Session: {session_id}")
    except Exception as e:
        print(f"Error during vectorization: {e}")
        # In production, you might raise an error or handle gracefully
        
    # --- STEP 2: Create Outline ---
    outline = [
        "1. Introduction",
        "2. Meeting Attendees & Roles",
        "3. Functional Requirements",
        "4. Action Items & Timeline"
    ]
    
    return {
        **state,
        "sections_outline": outline,
        "current_section_index": 0,
        "is_complete": False
    }

def generate_section(state: DocumentGenerationState) -> DocumentGenerationState:
    """
    Generates content using RAG:
    1. Retrieves relevant chunks from Qdrant based on section title.
    2. Feeds those chunks to LLM.
    """
    current_idx = state["current_section_index"]
    sections = state["sections_outline"]
    
    if current_idx >= len(sections):
        return {**state, "is_complete": True}

    section_title = sections[current_idx]
    session_id = state["session_id"]
    feedback = state.get("feedback_notes")
    
    print(f"--- Generating Section: {section_title} ---")

    # --- STEP 3: RAG Retrieval ---
    # Instead of dumping all 'meeting_minutes', we fetch what is relevant for THIS section.
    # We use the section title as the search query.
    relevant_context = retrieve_context(session_id, query=section_title, limit=4)
    
    print(f"Context Retrieved: {len(relevant_context)} chars")

    # --- STEP 4: Construct Prompt with Context ---
    prompt = f"""
    You are an expert Technical Writer creating a Software Requirements Specification (SRS).
    
    Task: Write the content for the section: '{section_title}'.
    
    Reference Context (Use ONLY this information to write the section):
    {relevant_context}
    
    If the context does not contain enough information for this specific section, 
    write a placeholder stating what is missing, but do not hallucinate facts.
    """
    
    if feedback:
        prompt += f"\nIMPORTANT: The user rejected the previous draft. Please incorporate this feedback: {feedback}"

    content = ""
    if MOCK_MODE:
        time.sleep(1)
        content = f"### {section_title}\n\n[Generated via RAG]\nContext used: {relevant_context[:50]}...\nContent..."
    else:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

    return {
        **state,
        "current_section_content": content,
        "feedback_notes": None, 
        "feedback_action": None
    }

def process_approval(state: DocumentGenerationState) -> DocumentGenerationState:
    """Moves approved content to final document and advances index."""
    # (This function remains unchanged from your previous code)
    print("--- Processing Approval ---")
    
    new_section = {
        "title": state["sections_outline"][state["current_section_index"]],
        "content": state["current_section_content"]
    }
    
    updated_doc = state["final_document"] + [new_section]
    next_index = state["current_section_index"] + 1
    
    is_done = next_index >= len(state["sections_outline"])
    
    return {
        **state,
        "final_document": updated_doc,
        "current_section_index": next_index,
        "is_complete": is_done
    }