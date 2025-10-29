"""Simplified prompt management for whisper-flow."""

from .completion import MessageType
from .system import SystemManager


class PromptManager:
    """Simplified prompt manager with single template approach."""

    def __init__(self, config, system_manager: SystemManager):
        """Initialize prompt manager.

        Args:
            config: Configuration object
            system_manager: System manager for window detection and highlighted text

        """
        self.config = config
        self.system_manager = system_manager

    def get_system_message(self) -> str:
        """Get the system message for the LLM.

        Returns:
            System message asking the LLM to be a helpful assistant

        """
        return (
            "You are a helpful assistant. "
            "Process the user's input and provide a useful response. "
            "You must always reply with only what the user asked for. "
            "No intro. No outro. No explanation. No apologies. "
            "No markdown. No formatting. No emojis. "
        )

    def get_user_message(self, transcript: str) -> str:
        """Get the user message with context variables.

        Args:
            transcript: Transcribed text from the user

        Returns:
            Formatted user message with highlighted text context

        """
        if not transcript.strip():
            return ""

        # Get highlighted text if any
        highlighted_text = self.system_manager.get_highlighted_text()

        message_template = """
# Use selected text if any
<selected_text>
{highlighted_text}
</selected_text>

# User input
<user_input>
{transcript}
</user_input>
"""

        return message_template.format(
            highlighted_text=f"Highlighted text: {highlighted_text}"
            if highlighted_text
            else "",
            transcript=transcript,
        )

    def get_messages(self, transcript: str) -> list[MessageType]:
        """Get the complete message list for the LLM.

        Args:
            transcript: Transcribed text from the user

        Returns:
            List of message dictionaries with system and user messages

        """
        if not transcript.strip():
            return []

        return [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": self.get_user_message(transcript)},
        ]

    def should_use_completion(self) -> bool:
        """Determine if completion should be used.

        Returns:
            True if completion should be used (always true for the new system)

        """
        return True
