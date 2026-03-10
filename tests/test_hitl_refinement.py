"""
Verification script for Module 6: HITL Refinement via Feedback.
Simulates a multi-step interaction:
1. Init & Initial Generation.
2. Feedback-driven Refinement.
3. Approval and State Progression.
"""
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.session_manager import session_manager
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    run_init_step,
    run_generation_step,
    run_approval_step
)

async def test_refinement_flow():
    print("="*60)
    print("MODULE 6: HITL Refinement Validation")
    print("="*60)

    # 1. Initialize Session
    minutes = "Raza: We need to build a hybrid retriever. Hamza: Agreed. Shahzeb: Let's use BM25."
    doc_id = session_manager.create_session(minutes, "test_proj")
    print(f"\n[Step 1] Session Created: {doc_id}")

    # 2. Run Init (Ingestion + First Outline + First Generation)
    state = session_manager.get_session(doc_id)
    state = await run_init_step(state) # This calls init + generate
    session_manager.update_session(doc_id, state)
    
    print(f"\n[Step 2] Initial Generation for '{state['sections_outline'][0]}':")
    print(f"Content: {state['current_section_content'][:100]}...")

    # 3. Simulate User Feedback
    feedback = "This is too long. Please make it a concise bulleted list."
    state["feedback_notes"] = feedback
    # Note: normally the API would set this
    session_manager.update_session(doc_id, state)
    print(f"\n[Step 3] User Feedback Added: '{feedback}'")

    # 4. Run Refinement (Regeneration)
    print("\n[Step 4] Running Refinement...")
    state = await run_generation_step(state)
    session_manager.update_session(doc_id, state)
    
    print("\nRefined Content:")
    print(state['current_section_content'])
    
    # Check History
    history = state.get("refinement_history", {})
    if "0" in history and len(history["0"]) == 1:
        print("\n✅ Success: Refinement history captured previous draft.")
        print(f"Captured Feedback: {history['0'][0]['feedback']}")
    else:
        print("\n❌ Error: Refinement history missing or incorrect.")

    # 5. Approve Section
    print("\n[Step 5] Approving Section...")
    state = await run_approval_step(state)
    session_manager.update_session(doc_id, state)
    
    if len(state["final_document"]) == 1:
        print(f"✅ Success: Section moved to final document.")
    if "0" not in state.get("refinement_history", {}):
        print(f"✅ Success: History cleared for modern section index.")
    
    print(f"\nNext pointer: {state['current_section_index']}")

    # 6. Generate Next Section (Section 1: Meeting Attendees)
    print(f"\n[Step 6] Generating Next Section: '{state['sections_outline'][state['current_section_index']]}'...")
    state = await run_generation_step(state)
    session_manager.update_session(doc_id, state)
    print(f"Content: {state['current_section_content'][:100]}...")

    # 7. Approve Next Section immediately (No refinement this time)
    print("\n[Step 7] Approving Section 1...")
    state = await run_approval_step(state)
    session_manager.update_session(doc_id, state)

    # 8. Final Check
    print("\n" + "="*60)
    print("FINAL DOCUMENT SUMMARY")
    print("="*60)
    for i, sec in enumerate(state["final_document"]):
        print(f"\nSection {i+1}: {sec['title']}")
        print(f"--- Content Preview ---\n{sec['content'][:150]}...\n-----------------------")

    if len(state["final_document"]) >= 2:
        print(f"\n✅ Success: Document generation continued correctly through multiple sections.")
    
    print("\n" + "="*60)
    print("Validation Complete.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_refinement_flow())
