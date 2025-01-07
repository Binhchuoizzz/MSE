class OpenAIServiceException(Exception):
    """Base exception for OpenAI service errors."""
    pass

class TranscriptException(Exception):
    """Base class for transcript-related exceptions."""
    pass

class TranscriptsDisabledException(TranscriptException):
    """Raised if transcripts are disabled for a given YouTube video."""
    pass

class NoTranscriptFoundException(TranscriptException):
    """Raised if no transcript is found."""
    pass
