"""Text completion functionality for whisper-flow."""

import time
from typing import TypedDict

from openai import OpenAI

from .config import Config


# Message type definitions
class MessageType(TypedDict):
    role: str
    content: str


class CompletionService:
    """Text completion service using OpenAI API."""

    def __init__(self, config: Config):
        """Initialize completion service.

        Args:
            config: Configuration object

        """
        self.config = config
        self.client = (
            OpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        )

    def complete_text(
        self,
        messages: list[MessageType],
        max_retries: int = 3,
    ) -> str | None:
        """Complete text using OpenAI chat completion API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_retries: Maximum number of retry attempts

        Returns:
            Completed text or None if failed

        """
        if not messages or not any(msg.get("content", "").strip() for msg in messages):
            return None

        for attempt in range(max_retries):
            try:
                return self._complete_with_openai(messages)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Completion failed after {max_retries} attempts: {e}",
                    )

                # Wait before retry (exponential backoff)
                wait_time = 2**attempt
                print(
                    f"Completion attempt {attempt + 1} failed, retrying in {wait_time}s...",
                )
                time.sleep(wait_time)

        return None

    def _complete_with_openai(self, messages: list[MessageType]) -> str:
        """Complete text using OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Completed text

        Raises:
            RuntimeError: If API key is not configured or client is not initialized

        """
        if not self.client:
            raise RuntimeError("OpenAI API key not configured")

        response = self.client.chat.completions.create(
            model=self.config.completion_model,
            temperature=self.config.temperature,
            messages=messages,
        )

        return response.choices[0].message.content.strip()

    def stream_completion(
        self,
        messages: list[MessageType],
        callback=None,
    ) -> str | None:
        """Stream text completion using OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            callback: Optional callback function for streaming updates

        Returns:
            Complete text or None if failed

        """
        if not self.client:
            raise RuntimeError("OpenAI API key not configured")

        try:
            stream = self.client.chat.completions.create(
                model=self.config.completion_model,
                temperature=self.config.temperature,
                messages=messages,
                stream=True,
            )

            complete_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    complete_text += content
                    if callback:
                        callback(content)

            return complete_text.strip()

        except Exception as e:
            print(f"Streaming completion failed: {e}")
            return None

    def validate_messages(self, messages: list[MessageType]) -> bool:
        """Validate messages for completion.

        Args:
            messages: List of message dictionaries

        Returns:
            True if valid, False otherwise

        """
        if not messages:
            return False

        for message in messages:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False
            if not message["role"] or not message["content"].strip():
                return False

        # Check total message length (approximate token limit)
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = 4000 * 4  # Conservative estimate for GPT-4
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        if total_length > max_chars:
            print(
                f"Messages too long: {total_length} characters (estimated max: {max_chars})",
            )
            return False

        return True

    def get_completion_info(self) -> dict:
        """Get information about completion capabilities.

        Returns:
            Dictionary containing completion service information

        """
        return {
            "openai_available": self.config.openai_api_key is not None,
            "current_model": self.config.completion_model,
            "temperature": self.config.temperature,
            "streaming_supported": True,
            "chat_mode": True,
        }

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def estimate_cost(
        self,
        messages: list[MessageType],
        completion: str = "",
    ) -> dict:
        """Estimate API cost for completion.

        Args:
            messages: List of message dictionaries
            completion: Completion text (if available)

        Returns:
            Dictionary with cost estimation

        """
        # Note: These are example prices and should be updated based on current OpenAI pricing
        model_pricing = {
            "gpt-4.1-nano": {"input": 0.0000005, "output": 0.0000015},  # per token
            "gpt-4o-mini": {"input": 0.000015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        }

        model = self.config.completion_model
        if model not in model_pricing:
            return {"error": f"Pricing not available for model: {model}"}

        pricing = model_pricing[model]
        input_tokens = sum(
            self.estimate_tokens(msg.get("content", "")) for msg in messages
        )
        output_tokens = self.estimate_tokens(completion)

        input_cost = input_tokens * pricing["input"] / 1000  # Pricing is per 1K tokens
        output_cost = output_tokens * pricing["output"] / 1000
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        }
