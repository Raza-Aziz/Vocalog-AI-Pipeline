import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState
from src.vocalog_ai_api.infrastructure.llm_providers.groq import llm

# Toggle this for real LLM vs Fast Demo
MOCK_MODE = False 

# llm = ChatOpenAI(model="gpt-4o", temperature=0.7) if not MOCK_MODE else None

def initialize_document(state: DocumentGenerationState) -> DocumentGenerationState:
    """Parses minutes and creates an SRS outline."""
    print("--- Initializing Document ---")
    
    # In a real scenario, LLM decides sections based on minutes.
    # For demo, we hardcode standard SRS sections.
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
    """Generates content for the CURRENT section based on context."""
    current_idx = state["current_section_index"]
    sections = state["sections_outline"]
    
    if current_idx >= len(sections):
        return {**state, "is_complete": True}

    section_title = sections[current_idx]
    minutes = state["meeting_minutes"]
    feedback = state.get("feedback_notes")
    
    print(f"--- Generating Section: {section_title} ---")

    # Construct Prompt
    prompt = f"""
    You are writing an SRS document based on these meeting minutes:
    {minutes}
    
    Current Task: Write content for section '{section_title}'.
    """
    
    if feedback:
        prompt += f"\nIMPORTANT: The user rejected the previous draft. Please incorporate this feedback: {feedback}"

    content = ""
    if MOCK_MODE:
        time.sleep(1) # Simulate thinking
        content = f"### {section_title}\n\n[Demo Content based on minutes]\nThis is a generated draft for {section_title}.\nBased on discussion: {minutes[:50]}..."
        if feedback:
            content += f"\n\n(Addressed feedback: {feedback})"
    else:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

    return {
        **state,
        "current_section_content": content,
        # Reset feedback after usage so we don't apply it to the next section
        "feedback_notes": None, 
        "feedback_action": None
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
    
    is_done = next_index >= len(state["sections_outline"])
    
    return {
        **state,
        "final_document": updated_doc,
        "current_section_index": next_index,
        "is_complete": is_done
    }