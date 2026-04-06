"""
Test: Action Extractor — Module 7.1 Verification
Uses real transcript data to run a complete pipeline flow:
  1. Ingest transcript → 2. Generate sections → 3. Approve all → 4. Extract actions
"""

import sys
import os
import json
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dotenv import load_dotenv
load_dotenv()

from vocalog_ai_api.application.pipelines.doc_generation_pipeline.session_manager import session_manager
from vocalog_ai_api.application.pipelines.doc_generation_pipeline.graph import (
    run_init_step,
    run_generation_step,
    run_approval_step,
    run_extraction_step
)


async def test_full_pipeline_with_extraction():
    print("=" * 60)
    print("Module 7 Test: Full Pipeline -> Action Extraction")
    print("=" * 60)

    # 1. Load real transcript
    transcript_path = os.path.join(os.path.dirname(__file__), "transcript1.json")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    
    meeting_text = transcript_data["text"]
    print(f"\n[Step 1] Loaded transcript1.json ({len(meeting_text)} chars)")
    print(f"Preview: {meeting_text[:100]}...")

    # 2. Create Session & Initialize
    doc_id = session_manager.create_session(meeting_text, "module7-test")
    state = session_manager.get_session(doc_id)
    print(f"\n[Step 2] Session created: {doc_id}")

    # 3. Run Init (vectorize + generate first section)
    state = await run_init_step(state)
    session_manager.update_session(doc_id, state)
    print(f"\n[Step 3] Initialized. Outline: {state['sections_outline']}")
    print(f"First section: '{state['sections_outline'][0]}'")
    print(f"Content preview: {state['current_section_content'][:80]}...")

    # 4. Approve all sections to complete the document
    total_sections = len(state["sections_outline"])
    for i in range(total_sections):
        section_title = state["sections_outline"][state["current_section_index"]]
        print(f"\n[Step 4.{i+1}] Approving: '{section_title}'")
        
        state = await run_approval_step(state)
        session_manager.update_session(doc_id, state)
        
        if state["is_complete"]:
            print("  → Document COMPLETE!")
            break
        
        # Generate the next section
        state = await run_generation_step(state)
        session_manager.update_session(doc_id, state)
        print(f"  → Next section generated: '{state['sections_outline'][state['current_section_index']]}'")

    # 5. Verify document is complete
    assert state["is_complete"], "Document should be complete after approving all sections"
    assert len(state["final_document"]) == total_sections, \
        f"Expected {total_sections} sections in final doc, got {len(state['final_document'])}"
    print(f"\n[Step 5] Final document has {len(state['final_document'])} sections ✅")

    # 6. Run Action Extraction
    print(f"\n[Step 6] Running Action Extraction...")
    state = await run_extraction_step(state)
    session_manager.update_session(doc_id, state)
    
    extracted = state.get("extracted_actions")
    assert extracted is not None, "extracted_actions should not be None"

    # 7. Display results
    print(f"\n{'='*60}")
    print("EXTRACTION RESULTS")
    print(f"{'='*60}")

    print(f"\n📝 Summary:\n  {extracted.get('summary', 'N/A')}")

    action_items = extracted.get("action_items", [])
    print(f"\n📋 Action Items ({len(action_items)}):")
    for i, item in enumerate(action_items, 1):
        print(f"  {i}. [{item.get('priority', '?').upper()}] {item.get('description', '?')}")
        print(f"     Owner: {item.get('owner', '?')} | Deadline: {item.get('deadline', '?')}")

    decisions = extracted.get("key_decisions", [])
    print(f"\n✅ Key Decisions ({len(decisions)}):")
    for d in decisions:
        print(f"  - {d}")

    # 8. Assertions
    assert len(action_items) >= 1, \
        f"Expected at least 1 action item, got {len(action_items)}"
    
    print(f"\n{'='*60}")
    print("✅ Module 7 Action Extraction Test PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline_with_extraction())
