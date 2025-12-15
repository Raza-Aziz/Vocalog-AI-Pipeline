from langgraph.graph import StateGraph, END
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.state import DocumentGenerationState
from src.vocalog_ai_api.application.pipelines.doc_generation_pipeline.nodes import (
    initialize_document,
    generate_section,
    process_approval
)

def create_doc_gen_graph():
    workflow = StateGraph(DocumentGenerationState)

    # Add Nodes
    workflow.add_node("init", initialize_document)
    workflow.add_node("generate", generate_section)
    workflow.add_node("save_approved", process_approval)

    # --- Routing Logic ---
    
    # 1. Start -> Init
    workflow.set_entry_point("init")

    # 2. Init -> Generate
    workflow.add_edge("init", "generate")

    # 3. Logic Router (Manual Step Wrapper)
    # We don't actually loop here for the demo. 
    # The API calls specific functions, but to make the graph valid:
    workflow.add_edge("save_approved", END)
    workflow.add_edge("generate", END)

    return workflow.compile()

# For the demo, we often want to run specific "chunks" of logic.
# However, standard LangGraph runs start to finish.
# To achieve "Step-by-Step", we will rely on the State stored in SessionManager
# and deciding which Node to simulate or if we just re-run the 'generate' node.

# Helper to run just the generation step given a state
async def run_generation_step(current_state: DocumentGenerationState):
    # This acts as a mini-graph execution for just the generation node
    # Since we are manually managing state, we can just call the node function directly
    # or wrap it in a graph. For simplicity/reliability in demo:
    return generate_section(current_state)

async def run_approval_step(current_state: DocumentGenerationState):
    return process_approval(current_state)

async def run_init_step(current_state: DocumentGenerationState):
    state_after_init = initialize_document(current_state)
    return generate_section(state_after_init)