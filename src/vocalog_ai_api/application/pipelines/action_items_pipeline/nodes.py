from vocalog_ai_api.application.pipelines.action_items_pipeline.state import ActionItemsState
from vocalog_ai_api.application.pipelines.action_items_pipeline.schema import ActionExtractionResult
from vocalog_ai_api.infrastructure.llm_providers.groq import llm
from langchain_core.messages import SystemMessage, HumanMessage

def extract_actions(state: ActionItemsState) -> dict:
    """
    Analyzes the meeting transcript and extracts actionable tasks.
    """
    transcript = state.get("transcript", "")
    
    system_prompt = """You are an AI assistant specialized in extracting every actionable item from meeting transcripts.

Carefully read the ENTIRE transcript and identify ALL tasks, follow-ups, commitments, and responsibilities.

For EACH action item, extract:
1. 'assignee': The full name or role of the responsible person.
2. 'task_description': A precise, self-contained description of the task.
3. 'due_date': The deadline if explicitly mentioned (e.g. 'Friday', '2026-05-10', 'end of sprint'). Use null if not stated.
4. 'priority': Urgency level — 'high' if the task is blocking, time-critical, or flagged as urgent; 'low' if explicitly deferred or a nice-to-have; 'medium' for everything else.
5. 'target_platform': 'slack', 'github', 'gmail', or 'unknown'.

You MUST return a valid JSON object with an 'actions' key containing a list of these items.
"""
    
    # Configure LLM to use JSON mode for better reliability on Groq
    structured_llm = llm.with_structured_output(ActionExtractionResult, method="json_mode")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{transcript}\n\nExtract the action items.")
    ]
    
    # Execute extraction
    result: ActionExtractionResult = structured_llm.invoke(messages)
    
    # Return state update
    return {"extracted_actions": result.actions}
