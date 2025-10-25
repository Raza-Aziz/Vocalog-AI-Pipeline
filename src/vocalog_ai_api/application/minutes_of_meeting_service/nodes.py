from state import MoMGraphState
from schema import MinutesOfMeeting
from vocalog_ai_api.infrastructure.llm_providers.groq import llm
from domain.prompts.minutes_of_meeting import structured_mom_prompt, mom_markdown_instructions

from langchain_core.messages import SystemMessage, HumanMessage


def generate_mom(state: MoMGraphState):

  structured_llm = llm.with_structured_output(MinutesOfMeeting)

  system_message = structured_mom_prompt.format(
      meeting_transcript=state["raw_transcript"]
  )

  # Generate MoM
  response_mom = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate Minutes of Meeting.")])

  return {'mom': response_mom}


def generate_markdown_mom(state: MoMGraphState):

  system_message = mom_markdown_instructions.format(
      structured_mom = state['mom']
  )

  response_markdown = llm.invoke([SystemMessage(content=system_message),
                                  HumanMessage(content='Generate structured markdown minutes of meeting')])

  return {'mom_markdown': response_markdown}