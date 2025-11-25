from fastapi import FastAPI
from vocalog_ai_api.infrastructure.api.schemas import MoMResponse, TranscriptInput

from application.minutes_of_meeting_service.graph import mom_graph

app = FastAPI()


@app.post("/generate-mom", response_model=MoMResponse)
def generate_minutes_of_meeting(data: TranscriptInput):
    """
    Generate standardized Minutes of Meeting from a transcript.
    """
    # 1. Run LangGraph pipeline
    result_state = mom_graph.invoke({"raw_transcript": data.raw_transcript})

    # 2. Extract Markdown output from state
    markdown = result_state.get("mom_markdown", "")

    # 3. Return Markdown response
    return {"markdown": markdown}
