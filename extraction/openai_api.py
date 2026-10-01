"""
OpenAI API wrapper for GPT-4.1 and GPT-5-mini.
Replaces Qwen models with OpenAI API calls.
"""

import json
from openai import OpenAI


class OpenAIApi:
    """
    Unified OpenAI API client supporting GPT-4.1, GPT-5-mini, and other OpenAI models.
    Supports JSON mode for guaranteed valid JSON outputs.
    """

    def __init__(
        self,
        model_id: str = "gpt-4.1",
        temperature: float = 0.2,
        top_p: float = 0.1,
        max_tokens: int = 10000,
        api_key: str = None,
    ):
        """
        Initialize OpenAI API client.

        Args:
            model_id: Model name (e.g., 'gpt-4.1', 'gpt-5-mini', 'gpt-4-turbo')
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Initialize client (uses OPENAI_API_KEY env var if api_key is None)
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    def request(self, user_prompt: str, json_mode: bool = False, system_prompt: str = None) -> str:
        """
        Make a request to the OpenAI API.

        Args:
            user_prompt: The user message/prompt
            json_mode: If True, enables JSON response format for guaranteed valid JSON
            system_prompt: Optional system message to set context

        Returns:
            The model's response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        # Enable JSON mode for structured outputs
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def request_json(self, user_prompt: str, system_prompt: str = None) -> dict:
        """
        Make a request expecting JSON response. Automatically parses the result.

        Args:
            user_prompt: The user message/prompt (should ask for JSON output)
            system_prompt: Optional system message

        Returns:
            Parsed JSON as a dictionary

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.request(user_prompt, json_mode=True, system_prompt=system_prompt)
        return json.loads(response)

    def request_with_retry(
        self, user_prompt: str, json_mode: bool = False, max_retries: int = 3
    ) -> str:
        """
        Make a request with automatic retries on failure.

        Args:
            user_prompt: The user message/prompt
            json_mode: If True, enables JSON response format
            max_retries: Maximum number of retry attempts

        Returns:
            The model's response text, or empty string if all retries fail
        """
        for attempt in range(1, max_retries + 1):
            try:
                return self.request(user_prompt, json_mode=json_mode)
            except Exception as e:
                print(f"API request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return ""
        return ""


class GPT41Api(OpenAIApi):
    """Convenience class for GPT-4.1 with optimized defaults."""

    def __init__(self, temperature: float = 0.2, top_p: float = 0.1, max_tokens: int = 10000):
        super().__init__(
            model_id="gpt-4.1",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


class GPT5MiniApi(OpenAIApi):
    """Convenience class for GPT-5-mini with optimized defaults."""

    def __init__(self, temperature: float = 0.2, top_p: float = 0.1, max_tokens: int = 10000):
        super().__init__(
            model_id="gpt-5-mini",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


# For backwards compatibility - factory function
def get_api(model_id: str, **kwargs):
    """
    Factory function to get the appropriate API client.

    Args:
        model_id: Model identifier (e.g., 'gpt-4.1', 'gpt-5-mini', 'Qwen3-VL-32B')
        **kwargs: Additional arguments passed to the API constructor

    Returns:
        API client instance
    """
    if model_id.startswith("gpt-"):
        return OpenAIApi(model_id=model_id, **kwargs)
    else:
        # Fall back to existing Qwen APIs if needed
        raise ValueError(
            f"Model {model_id} not supported. Use 'gpt-4.1', 'gpt-5-mini', or similar OpenAI models."
        )
