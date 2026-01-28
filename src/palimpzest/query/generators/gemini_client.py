"""
Direct client for Google AI Studio (Gemini) that bypasses litellm.

This module provides a GeminiClient class that:
1. Calls Gemini API directly via google-generativeai SDK
2. Converts litellm/palimpzest message format to Gemini format
3. Supports both implicit and explicit caching options
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any
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
    Direct client for Google AI Studio (Gemini) that bypasses litellm.

    Supports both implicit caching (automatic prefix matching) and explicit caching
    (using CachedContent for longer-lived caches).

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        use_explicit_cache: If True, use explicit CachedContent for caching.
                           If False (default), rely on implicit context caching.
        cache_ttl_seconds: TTL for explicit cache (default: 300 = 5 minutes)

    Example usage:
        # Implicit caching (automatic prefix matching)
        client = GeminiClient(model="gemini-2.5-flash")
        response = client.generate(messages)

        # Explicit caching (longer-lived cache with CachedContent)
        client = GeminiClient(
            model="gemini-2.5-flash",
            use_explicit_cache=True,
            cache_ttl_seconds=600
        )
        response = client.generate(messages)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        use_explicit_cache: bool = False,
        cache_ttl_seconds: int = 300,
    ):
        self.model = model
        self.use_explicit_cache = use_explicit_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = None
        self._cached_content = None  # For explicit caching
        self._cached_system_instruction = None  # Track what's cached

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

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
                # Extract system instruction
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
                                _, data = url.split(";base64,")
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
                    gemini_contents.append({"role": "model", "parts": parts})

        return system_instruction, gemini_contents

    def _create_or_get_cache(self, system_instruction: str | None) -> Any:
        """
        Create or retrieve explicit cache for the system instruction.

        Args:
            system_instruction: The system instruction to cache

        Returns:
            The cached_content object if explicit caching is enabled,
            otherwise returns None.
        """
        if not self.use_explicit_cache or not system_instruction:
            return None

        from datetime import timedelta
        from google.genai import types

        # Check if we already have a valid cache for this instruction
        if (self._cached_content is not None and
            self._cached_system_instruction == system_instruction):
            return self._cached_content

        client = self._get_client()

        # Create new cached content
        try:
            self._cached_content = client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    ttl=timedelta(seconds=self.cache_ttl_seconds),
                )
            )
            self._cached_system_instruction = system_instruction
            logger.debug(f"Created explicit cache: {self._cached_content.name}")
            return self._cached_content
        except Exception as e:
            logger.warning(f"Failed to create explicit cache: {e}")
            return None

    def _extract_usage_stats(self, usage_metadata: Any) -> dict:
        """
        Extract usage statistics from Gemini response.

        Args:
            usage_metadata: The usage_metadata from Gemini response

        Returns:
            Dictionary with usage statistics
        """
        if usage_metadata is None:
            return {}

        try:
            return usage_metadata.model_dump()
        except AttributeError:
            try:
                return usage_metadata.to_dict()
            except AttributeError:
                # Manual extraction
                return {
                    "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
                    "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
                    "total_token_count": getattr(usage_metadata, "total_token_count", None),
                    "cached_content_token_count": getattr(usage_metadata, "cached_content_token_count", None),
                    "prompt_tokens_details": getattr(usage_metadata, "prompt_tokens_details", None),
                    "cache_tokens_details": getattr(usage_metadata, "cache_tokens_details", None),
                }

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        **kwargs
    ) -> GeminiResponse:
        """
        Generate content using Gemini API directly.

        Args:
            messages: List of messages in litellm/palimpzest format
            temperature: Sampling temperature (default: 0.0)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            GeminiResponse with content, usage stats, and raw response
        """
        client = self._get_client()
        system_instruction, gemini_contents = self._transform_messages(messages)

        # Build config
        config_kwargs = {"temperature": temperature}

        if self.use_explicit_cache:
            # Use explicit caching
            cached_content = self._create_or_get_cache(system_instruction)
            if cached_content:
                # When using cached content, don't include system_instruction in config
                response = client.models.generate_content(
                    model=cached_content.name,  # Use cache name as model
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            else:
                # Fallback to regular call if cache creation failed
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction
                response = client.models.generate_content(
                    model=self.model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
        else:
            # Use implicit caching (automatic prefix matching)
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            response = client.models.generate_content(
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

        # Extract usage stats
        usage = self._extract_usage_stats(response.usage_metadata)

        return GeminiResponse(
            content=content,
            usage=usage,
            raw_response=response,
        )

    def clear_cache(self):
        """Clear the explicit cache if it exists."""
        if self._cached_content is not None:
            try:
                client = self._get_client()
                client.caches.delete(name=self._cached_content.name)
                logger.debug(f"Deleted explicit cache: {self._cached_content.name}")
            except Exception as e:
                logger.warning(f"Failed to delete cache: {e}")
            finally:
                self._cached_content = None
                self._cached_system_instruction = None
