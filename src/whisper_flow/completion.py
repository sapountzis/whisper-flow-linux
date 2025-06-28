"""Text completion functionality for whisper-flow."""

import time

import requests

from .config import Config


class CompletionService:
    """Text completion service using OpenAI API."""

    def __init__(self, config: Config):
        """Initialize completion service.

        Args:
            config: Configuration object

        """
        self.config = config

    def complete_text(self, prompt: str, max_retries: int = 3) -> str | None:
        """Complete text using OpenAI chat completion API.

        Args:
            prompt: Input prompt for completion
            max_retries: Maximum number of retry attempts

        Returns:
            Completed text or None if failed

        """
        if not prompt.strip():
            return None

        for attempt in range(max_retries):
            try:
                return self._complete_with_openai(prompt)
            except requests.exceptions.RequestException as e:
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

    def _complete_with_openai(self, prompt: str) -> str:
        """Complete text using OpenAI API.

        Args:
            prompt: Input prompt for completion

        Returns:
            Completed text

        Raises:
            RuntimeError: If API key is not configured
            requests.exceptions.RequestException: If API request fails

        """
        if not self.config.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")

        payload = {
            "model": self.config.completion_model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        response = requests.post(
            self.config.openai_url,
            headers=self.config.openai_headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        response_data = response.json()
        if "choices" not in response_data or not response_data["choices"]:
            raise RuntimeError("Invalid response format from OpenAI API")

        return response_data["choices"][0]["message"]["content"].strip()

    def stream_completion(self, prompt: str, callback=None) -> str | None:
        """Stream text completion using OpenAI API.

        Args:
            prompt: Input prompt for completion
            callback: Optional callback function for streaming updates

        Returns:
            Complete text or None if failed

        """
        if not self.config.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")

        payload = {
            "model": self.config.completion_model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        try:
            response = requests.post(
                self.config.openai_url,
                headers=self.config.openai_headers,
                json=payload,
                timeout=60,
                stream=True,
            )
            response.raise_for_status()

            complete_text = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8")
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            import json

                            data = json.loads(data_str)
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    complete_text += content
                                    if callback:
                                        callback(content)
                        except json.JSONDecodeError:
                            continue

            return complete_text.strip()

        except requests.exceptions.RequestException as e:
            print(f"Streaming completion failed: {e}")
            return None

    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt for completion.

        Args:
            prompt: Prompt text to validate

        Returns:
            True if valid, False otherwise

        """
        if not prompt or not prompt.strip():
            return False

        # Check prompt length (approximate token limit)
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = 4000 * 4  # Conservative estimate for GPT-4
        if len(prompt) > max_chars:
            print(
                f"Prompt too long: {len(prompt)} characters (estimated max: {max_chars})",
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
            "completion_url": self.config.openai_url,
            "streaming_supported": True,
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

    def estimate_cost(self, prompt: str, completion: str = "") -> dict:
        """Estimate API cost for completion.

        Args:
            prompt: Input prompt
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
        input_tokens = self.estimate_tokens(prompt)
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
