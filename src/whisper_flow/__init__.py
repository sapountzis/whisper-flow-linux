"""WhisperFlow - AI-powered voice-to-text with context-aware processing."""

__version__ = "0.1.0"
__author__ = "sapountzis"
__email__ = "sapountzis.andreas@gmail.com"

from .app import WhisperFlow
from .audio import AudioRecorder
from .completion import CompletionService
from .config import Config
from .daemon import WhisperFlowDaemon
from .logging import log, set_logging_enabled
from .prompts import PromptManager
from .system import SystemManager
from .transcription import TranscriptionService

__all__ = [
    "AudioRecorder",
    "CompletionService",
    "Config",
    "PromptManager",
    "SystemManager",
    "TranscriptionService",
    "WhisperFlow",
    "WhisperFlowDaemon",
    "log",
    "set_logging_enabled",
]
