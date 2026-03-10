from vocalog_ai_api.application.pipelines.mom_pipeline.state import MoMGraphState
from vocalog_ai_api.application.pipelines.mom_pipeline.schema import MinutesOfMeeting
from vocalog_ai_api.infrastructure.llm_providers.groq import llm

from vocalog_ai_api.domain.prompts.minutes_of_meeting import structured_mom_prompt, mom_markdown_instructions

from langchain_core.messages import SystemMessage, HumanMessage


def generate_mom(state: MoMGraphState):

  structured_llm = llm.with_structured_output(MinutesOfMeeting)

  system_message = structured_mom_prompt.format(
      meeting_transcript=state["raw_transcript"]
  )

  # Generate MoM
  response_mom = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate Minutes of Meeting.")])
  print(response_mom)

  return {'mom': response_mom}


def generate_markdown_mom(state: MoMGraphState):

  system_message = mom_markdown_instructions.format(
      structured_mom = state['mom']
  )

  response_markdown = llm.invoke([SystemMessage(content=system_message),
                                  HumanMessage(content='Generate structured markdown minutes of meeting')])

  print("\n", response_markdown)
  print(type(response_markdown.content))
  return {'mom_markdown': response_markdown.content}