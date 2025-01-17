import re
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from services.exceptions import TranscriptsDisabledException, NoTranscriptFoundException, TranscriptException

def extract_video_id(youtube_url: str) -> str:
    """
    Extracts the video ID from a given YouTube link.
    It handles common formats like:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    """
    patterns = [
        r"(?:v=)([A-Za-z0-9_\-]+)",   # looks for "v=VIDEO_ID"
        r"(?:be/)([A-Za-z0-9_\-]+)",  # looks for "youtu.be/VIDEO_ID"
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    raise ValueError("Could not extract a valid video ID from the provided URL.")

def get_transcript(video_id: str, target_language="en") -> str:
    """
    Fetches the transcript for the given YouTube video ID.
    If multiple languages are available, tries to get `target_language`.
    Returns a single string containing the entire transcript.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to find the transcript in the target language
        transcript = None
        for t in transcript_list:
            if t.language_code == target_language:
                transcript = t.fetch()
                break

        # If we didn't find the target language, pick the first available
        if transcript is None:
            transcript = transcript_list.find_transcript(
                [t.language_code for t in transcript_list]
            ).fetch()

        # Join the transcript pieces into a single string
        transcript_text = " ".join([item["text"] for item in transcript])
        return transcript_text

    except TranscriptsDisabled:
        raise TranscriptsDisabledException("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise NoTranscriptFoundException("No transcript was found for this video.")
    except Exception as e:
        raise TranscriptException(f"Could not retrieve transcript: {str(e)}")