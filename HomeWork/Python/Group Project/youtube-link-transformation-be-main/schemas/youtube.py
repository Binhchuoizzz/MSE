from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    youtube_url: str
    summary_length: int