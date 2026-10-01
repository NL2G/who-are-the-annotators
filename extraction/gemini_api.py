"""
Google Gemini API wrapper for Gemini 3 (and 2.5) models.
Mirrors the interface of openai_api.py for drop-in usage.
"""

import json
import os
from typing import Optional

from google import genai
from google.genai import types


class GeminiApi:
    """
    Unified Google Gemini API client.
    Supports JSON mode for guaranteed valid JSON outputs.
    """

    def __init__(
        self,
        model_id: str = "gemini-3.1-pro-preview",
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 10000,
        thinking_level: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY env var, "
                "or pass api_key= explicitly."
            )
        self.client = genai.Client(api_key=key)

    def request(
        self,
        user_prompt: str,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
        response_json_schema: Optional[dict] = None,
    ) -> str:
        config_kwargs = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
        )

        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if response_json_schema is not None:
            # Structured output mode: schema-constrained JSON
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_json_schema
        elif json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        if self.thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level
            )

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=config,
        )
        return response.text

    def request_json(self, user_prompt: str, system_prompt: Optional[str] = None, response_json_schema: Optional[dict] = None) -> dict:
        response = self.request(user_prompt, json_mode=True, system_prompt=system_prompt, response_json_schema=response_json_schema)
        return json.loads(response)

    def request_with_retry(
        self,
        user_prompt: str,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                return self.request(user_prompt, json_mode=json_mode)
            except Exception as e:
                print(f"Gemini API request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return ""
        return ""


class Gemini3ProApi(GeminiApi):
    """Convenience class for Gemini 3.1 Pro Preview with recommended defaults."""

    def __init__(self, temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 10000, thinking_level: Optional[str] = None):
        super().__init__(
            model_id="gemini-3.1-pro-preview",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking_level=thinking_level,
        )


class Gemini25ProApi(GeminiApi):
    """Convenience class for Gemini 2.5 Pro (kept for backwards compatibility)."""

    def __init__(self, temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 10000):
        super().__init__(
            model_id="gemini-2.5-pro",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
