from fastapi import APIRouter, HTTPException
from services.youtube_service import (
    extract_video_id,
    get_transcript,
    TranscriptException,
)
from services.openai_service import (
    summarize_text_with_openai,
    create_outline_with_openai,
    generate_mind_map_with_openai,
    OpenAIServiceException
)
from schemas.youtube import SummarizationRequest

router = APIRouter()

@router.post("/api/summarize")
async def summarize_video(request: SummarizationRequest):
    """
    Summarize a YouTube transcript and generate a mind map using OpenAI.
    """
    try:
        # 1) Extract video ID
        video_id = extract_video_id(request.youtube_url)

        # 2) Get the transcript
        transcript = get_transcript(video_id)

        # 3) Summarize the transcript
        summary = summarize_text_with_openai(transcript, request.summary_length)

        # 4) Generate the mind map
        outline = create_outline_with_openai(transcript)
        mind_map = generate_mind_map_with_openai(outline)

        return {
            "summary": summary,
            "mind_map": mind_map
        }

    except (ValueError, TranscriptException) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OpenAIServiceException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unknown error: {str(e)}")
