from vocalog_ai_api.application.pipelines.action_items_pipeline.state import ActionItemsState
from vocalog_ai_api.application.pipelines.action_items_pipeline.schema import ActionExtractionResult
from vocalog_ai_api.infrastructure.llm_providers.groq import llm
from langchain_core.messages import SystemMessage, HumanMessage

def extract_actions(state: ActionItemsState) -> dict:
    """
    Analyzes the meeting transcript and extracts actionable tasks.
    """
    transcript = state.get("transcript", "")
    
    system_prompt = """You are an AI assistant designed to extract actionable items from meeting transcripts.
Carefully review the following transcript and identify any tasks, action items, or follow-ups.

For each action item, extract:
1. 'assignee': The name or role of the person responsible for the task. If not explicitly assigned, use 'Unassigned'.
2. 'task_description': A clear, concise description of what needs to be done.
3. 'due_date': The deadline for the task, if mentioned in the transcript. Leave null if none is mentioned.
4. 'target_platform': Set to 'slack' by default, unless the user specifically mentions creating a GitHub issue ('github') or sending an email ('gmail').

If there are no action items, simply return an empty list.
"""
    
    # Configure LLM to force output to match the schema
    structured_llm = llm.with_structured_output(ActionExtractionResult)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{transcript}\n\nExtract the action items.")
    ]
    
    # Execute extraction
    result: ActionExtractionResult = structured_llm.invoke(messages)
    
    # Return state update
    return {"extracted_actions": result.actions}
