import time
from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState

# --- NEW IMPORTS ---
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.vector_store import (
    ingest_minutes,
    retrieve_context
)

from vocalog_ai_api.infrastructure.llm_providers.groq import llm

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
    3. If feedback is present, treats this as a refinement step.
    """
    current_idx = state["current_section_index"]
    sections = state["sections_outline"]
    
    if current_idx >= len(sections):
        return {**state, "is_complete": True}

    section_title = sections[current_idx]
    session_id = state["session_id"]
    feedback = state.get("feedback_notes")
    history = state.get("refinement_history", {})
    
    # Track history if we are refining
    if feedback and state.get("current_section_content"):
        str_idx = str(current_idx)
        if str_idx not in history:
            history[str_idx] = []
        history[str_idx].append({
            "draft": state["current_section_content"],
            "feedback": feedback
        })
    
    print(f"--- Generating Section: {section_title} (Refinement: {bool(feedback)}) ---")

    # --- STEP 3: RAG Retrieval ---
    relevant_context = retrieve_context(session_id, query=section_title, limit=4)
    
    print(f"Context Retrieved: {len(relevant_context)} chars")

    # --- STEP 4: Construct Prompt ---
    if feedback:
        # --- REFINEMENT PROMPT ---
        prompt = f"""
        You are refining a section for a Software Requirements Specification (SRS).
        
        Section Title: '{section_title}'
        
        Previous Draft:
        {state.get("current_section_content")}
        
        User Feedback:
        {feedback}
        
        Source Context (Reference for facts):
        {relevant_context}
        
        Task: 
        Generate a NEW draft of this section that specifically addresses the user feedback. 
        Ensure you keep the relevant facts from the source context but adjust the tone, 
        structure, or detail as requested by the user.
        """
    else:
        # --- INITIAL DRAFT PROMPT ---
        prompt = f"""
        You are an expert Technical Writer creating a Software Requirements Specification (SRS).
        
        Task: Write the content for the section: '{section_title}'.
        
        Reference Context (Use ONLY this information to write the section):
        {relevant_context}
        
        If the context does not contain enough information for this specific section, 
        write a placeholder stating what is missing, but do not hallucinate facts.
        """

    content = ""
    if MOCK_MODE:
        time.sleep(1)
        content = f"### {section_title}\n\n[Refined via Feedback]\nOriginal: {state.get('current_section_content')[:30]}...\nFeedback: {feedback}\nContent..." if feedback else f"### {section_title}\n\n[Generated via RAG]\nContext used: {relevant_context[:50]}...\nContent..."
    else:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

    return {
        **state,
        "current_section_content": content,
        "feedback_notes": None, 
        "feedback_action": None,
        "refinement_history": history
    }

def process_approval(state: DocumentGenerationState) -> DocumentGenerationState:
    """Moves approved content to final document and advances index."""
    print("--- Processing Approval ---")
    
    new_section = {
        "title": state["sections_outline"][state["current_section_index"]],
        "content": state["current_section_content"]
    }
    
    updated_doc = state["final_document"] + [new_section]
    next_index = state["current_section_index"] + 1
    
    # Clean up transient state for the section we just finished
    history = state.get("refinement_history", {})
    str_idx = str(state["current_section_index"])
    if str_idx in history:
        del history[str_idx]

    is_done = next_index >= len(state["sections_outline"])
    
    return {
        **state,
        "final_document": updated_doc,
        "current_section_index": next_index,
        "feedback_notes": None,
        "feedback_action": None,
        "refinement_history": history,
        "is_complete": is_done
    }