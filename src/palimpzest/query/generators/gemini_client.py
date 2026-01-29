"""
Direct client for Gemini (Google AI Studio and Vertex AI) that bypasses litellm.

This module provides a GeminiClient class that:
1. Calls Gemini API directly via google-genai SDK
2. Converts litellm/palimpzest message format to Gemini format
3. Relies on implicit context caching (automatic prefix matching)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Response object that mimics litellm completion response structure."""
    content: str
    usage: dict
    raw_response: Any = None


class GeminiClient:
    """
    Direct client for Gemini (Google AI Studio and Vertex AI) that bypasses litellm.
    Uses implicit caching (automatic prefix matching) for prompt caching.

    Uses a singleton pattern per (model, use_vertex) so that client state is shared
    across all Generator instances using the same model and provider.

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        use_vertex: If True, use Vertex AI; otherwise use Google AI Studio
    """

    _instances: dict[tuple[str, bool], GeminiClient] = {}
    
    # Maps reasoning_effort to Gemini thinking_budget token counts
    # Reference: https://github.com/BerriAI/litellm/blob/620664921902d7a9bfb29897a7b27c1a7ef4ddfb/litellm/constants.py#L88
    REASONING_EFFORT_TO_THINKING_BUDGET = {
        "disable": 0,
        "minimal": 128,
        "low": 1024,
        "medium": 2048,
        "high": 4096,
    }

    @classmethod
    def get_instance(cls, model: str, use_vertex: bool = False) -> GeminiClient:
        """Get or create a singleton GeminiClient for the given model and provider."""
        key = (model, use_vertex)
        if key not in cls._instances:
            cls._instances[key] = cls(model, use_vertex)
        return cls._instances[key]

    def __init__(self, model: str, use_vertex: bool = False):
        self.model = model
        self.use_vertex = use_vertex
        # Vertex AI: uses GOOGLE_APPLICATION_CREDENTIALS for auth
        self.client = genai.Client(vertexai=True) if use_vertex else genai.Client()

    def _detect_image_media_type(self, base64_data: str) -> str:
        """Detect image format from base64 data by examining magic bytes."""
        try:
            header = base64.b64decode(base64_data[:32])
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if header[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            if header[:6] in (b"GIF87a", b"GIF89a"):
                return "image/gif"
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return "image/webp"
        except Exception:
            pass
        return "image/jpeg"

    def _transform_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """
        Transform litellm/palimpzest message format to Gemini API format.

        Args:
            messages: List of messages in litellm/palimpzest format

        Returns:
            Tuple of (system_instruction, gemini_contents)
        """
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            msg_type = msg.get("type")
            content = msg.get("content")

            if role == "system":
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    system_instruction = "".join(text_parts)
                else:
                    system_instruction = content

            elif role == "user":
                parts = []

                if msg_type == "text" or msg_type is None:
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                parts.append({"text": block.get("text", "")})
                    elif isinstance(content, str):
                        parts.append({"text": content})

                elif msg_type == "image":
                    for img in content:
                        if img.get("type") == "image_url":
                            url = img["image_url"]["url"]
                            if url.startswith("data:"):
                                # Robust parsing: handle "data:[<mediatype>];base64,<data>"
                                base64_marker = ";base64,"
                                marker_idx = url.find(base64_marker)
                                if marker_idx == -1:
                                    continue
                                data = url[marker_idx + len(base64_marker):]
                                media_type = self._detect_image_media_type(data)
                                parts.append({
                                    "inline_data": {
                                        "mime_type": media_type,
                                        "data": data,
                                    }
                                })

                elif msg_type == "input_audio":
                    for audio in content:
                        if audio.get("type") == "input_audio":
                            audio_data = audio["input_audio"]
                            parts.append({
                                "inline_data": {
                                    "mime_type": f"audio/{audio_data.get('format', 'wav')}",
                                    "data": audio_data["data"],
                                }
                            })

                if parts:
                    # Merge consecutive user messages
                    if gemini_contents and gemini_contents[-1]["role"] == "user":
                        gemini_contents[-1]["parts"].extend(parts)
                    else:
                        gemini_contents.append({"role": "user", "parts": parts})

            elif role == "assistant":
                # Convert assistant to model role
                parts = []
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})

                if parts:
                    # Merge consecutive model messages (Gemini requires strict role alternation)
                    if gemini_contents and gemini_contents[-1]["role"] == "model":
                        gemini_contents[-1]["parts"].extend(parts)
                    else:
                        gemini_contents.append({"role": "model", "parts": parts})

        return system_instruction, gemini_contents

    def _extract_usage_stats(self, usage_metadata: Any) -> dict:
        """
        Extract and process usage statistics from Gemini response into the
        standard format expected by Generator.

        Args:
            usage_metadata: The usage_metadata from Gemini response

        Returns:
            Dictionary with information needed by GenerationStats.
        """
        generation_stats = {
            "input_text_tokens": 0,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "cache_read_tokens": 0,
            "text_cache_read_tokens": 0,
            "image_cache_read_tokens": 0,
            "audio_cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_text_tokens": 0
        }

        if usage_metadata is None:
            return generation_stats

        try:
            raw = usage_metadata.model_dump()
        except (AttributeError, Exception):
            # Fallback for SDK versions without model_dump()
            raw = vars(usage_metadata) if hasattr(usage_metadata, "__dict__") else {}
            logger.warning("Could not call model_dump() on usage_metadata, using fallback")

        # Parse cache read tokens by modality
        for detail in (raw.get("cache_tokens_details") or []):
            modality = (detail.get("modality") or "").upper()
            token_count = detail.get("token_count") or 0
            if modality == "TEXT":
                generation_stats["text_cache_read_tokens"] = token_count
            elif modality == "IMAGE":
                generation_stats["image_cache_read_tokens"] = token_count
            elif modality == "AUDIO":
                generation_stats["audio_cache_read_tokens"] = token_count

        generation_stats["cache_read_tokens"] = raw.get("cached_content_token_count") or 0

        # Parse input tokens by modality (excludes cached tokens)
        for detail in (raw.get("prompt_tokens_details") or []):
            modality = (detail.get("modality") or "").upper()
            token_count = detail.get("token_count") or 0
            if modality == "TEXT":
                generation_stats["input_text_tokens"] = max(0, token_count - generation_stats["text_cache_read_tokens"])
            elif modality == "IMAGE":
                generation_stats["input_image_tokens"] = max(0, token_count - generation_stats["image_cache_read_tokens"])
            elif modality == "AUDIO":
                generation_stats["input_audio_tokens"] = max(0, token_count - generation_stats["audio_cache_read_tokens"])

        generation_stats["output_text_tokens"] = (raw.get("candidates_token_count") or 0) + (raw.get("thoughts_token_count") or 0)

        return generation_stats

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> GeminiResponse:
        """
        Generate content using Gemini API directly.

        Args:
            messages: List of messages in litellm/palimpzest format
            temperature: Sampling temperature (default: 0.0)
            reasoning_effort: Optional thinking budget level ("low", "medium", "high")

        Returns:
            GeminiResponse with content, usage stats, and raw response
        """
        system_instruction, gemini_contents = self._transform_messages(messages)

        # Build config
        config_kwargs = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Map reasoning_effort to thinking_config
        if reasoning_effort is not None:
            budget = self.REASONING_EFFORT_TO_THINKING_BUDGET.get(reasoning_effort)
            if budget is None:
                raise ValueError(f"Invalid reasoning effort: {reasoning_effort}")
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Extract response content
        content = ""
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join(
                    part.text for part in parts
                    if hasattr(part, "text") and part.text
                )

        # Extract and process usage stats
        usage = self._extract_usage_stats(response.usage_metadata)

        return GeminiResponse(
            content=content,
            usage=usage,
            raw_response=response,
        )
