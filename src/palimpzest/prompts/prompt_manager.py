"""
Prompt caching utility for different LLM providers.

This module provides provider-specific prompt caching configurations:
- OpenAI: Automatic prefix caching with prompt_cache_key for sticky routing
- Gemini (Google AI Studio / Vertex AI): Implicit caching (automatic prefix matching)
- Anthropic: Explicit cache_control with ephemeral type on system and user message content
"""

import copy
import uuid
from typing import Any

from palimpzest.constants import Model


class PromptManager:
    """
    Manages prompt caching configurations and message transformations for LLM providers.

    This class handles:
    1. Session-level state (e.g., OpenAI cache keys).
    2. Provider-specific request arguments (headers, extra_body).
    3. Transformation of messages for providers requiring explicit markers (Anthropic).
    4. Normalization of usage statistics.
    """
    
    CACHE_BOUNDARY_MARKER = "<<cache-boundary>>"

    def __init__(self, model: Model):
        self.model = model
        # Instance-level state ensures thread safety if we use one manager per plan/execution
        self.openai_cache_key = f"pz-cache-{uuid.uuid4().hex[:12]}" if self.model.is_provider_openai() else None

    def get_cache_kwargs(self) -> dict[str, Any]:
        """
        Get provider-specific cache configuration kwargs for litellm.completion().

        Returns:
            A dictionary of kwargs to pass to litellm.completion() for enabling caching
        """
        if not self.model.supports_prompt_caching():
            return {}
        # OpenAI: https://platform.openai.com/docs/guides/prompt-caching
        # Use prompt_cache_key for sticky routing to the same cache shard
        if self.model.is_provider_openai():
            return {"extra_body": {"prompt_cache_key": self.openai_cache_key}}
        else:
            return {}
    
    def inject_cache_isolation_id(self, messages: list[dict], session_id: str) -> list[dict]:
        """
        Inject a cache isolation ID into messages for testing cache behavior per-modality.

        This must happen BEFORE update_messages_for_caching so the ID becomes part of cached content.
        """
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system" and isinstance(content, str):
                msg["content"] = f"[{session_id}] " + content
            elif role == "user" and self.model.is_provider_anthropic() and msg.get("type") == "text" and isinstance(content, str):
                msg["content"] = f"[{session_id}] " + content
        return messages

    def update_messages_for_caching(self, messages: list[dict]) -> list[dict]:
        """
        Transform messages to conform to provider-specific caching requirements.

        - Anthropic: Adds explicit cache_control markers.
        - Others: Removes the generic cache boundary markers.

        Returns:
            The transformed messages list.
        """
        if not self.model.supports_prompt_caching():
            return messages

        # TODO: Update with changes from #265
        # Anthropic: Explicit cache_control with ephemeral type
        # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        if self.model.is_provider_anthropic():
            return self._transform_messages_for_anthropic(messages)
        # implicit caching for Gemini/OpenAI models that currently support caching
        # OpenAI: https://platform.openai.com/docs/guides/prompt-caching
        # Gemini: https://ai.google.dev/gemini-api/docs/caching
        elif (self.model.is_provider_openai() or
              self.model.is_provider_google_ai_studio() or self.model.is_provider_vertex_ai()):
            return self._remove_cache_boundary_markers(messages)

        return messages


    def extract_usage_stats(self, usage: dict, is_audio_op: bool) -> dict[str, int]:
        """
        Normalize cache statistics from provider-specific response formats.
        """
        stats = {
            "input_text_tokens": 0,
            "input_image_tokens": 0, # forward looking
            "input_audio_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0
        }

        details = usage.get("prompt_tokens_details") or {}

        if self.model.is_provider_openai():
            # assume audio models don't support caching for now
            # only realtime audio models do, but they are not supported by PZ
            if self.model.supports_prompt_caching() and not is_audio_op:
                stats["cache_read_tokens"] = details.get("cached_tokens") or 0
                stats["input_text_tokens"] = (usage.get("prompt_tokens") or 0) - stats["cache_read_tokens"]
            elif is_audio_op:
                stats["input_text_tokens"] = details.get("text_tokens") or 0
                stats["input_audio_tokens"] = details.get("audio_tokens") or 0
            else:
                stats["input_text_tokens"] = usage.get("prompt_tokens") or 0

        # TODO: verify for vertex ai
        elif self.model.is_provider_vertex_ai() or self.model.is_provider_google_ai_studio():
            # Try Gemini native field first, then litellm normalized field as fallback
            stats["cache_read_tokens"] = usage.get("cache_read_input_tokens") or 0
            if stats["cache_read_tokens"] == 0:
                # litellm may normalize Gemini responses to use prompt_tokens_details
                stats["cache_read_tokens"] = details.get("cached_tokens") or 0
            stats["input_text_tokens"] = details.get("text_tokens") or 0
            stats["input_audio_tokens"] = details.get("audio_tokens") or 0
            stats["input_image_tokens"] = details.get("image_tokens") or 0

        elif self.model.is_provider_anthropic():
            stats["cache_creation_tokens"] = usage.get("cache_creation_input_tokens") or 0
            stats["cache_read_tokens"] = usage.get("cache_read_input_tokens") or 0
            stats["input_text_tokens"] = max(0, (usage.get("prompt_tokens") or 0) - stats["cache_read_tokens"] - stats["cache_creation_tokens"])

        # all other models (assume caching not supported)
        else:
            if is_audio_op:
                stats["input_text_tokens"] = details.get("text_tokens") or 0
                stats["input_audio_tokens"] = details.get("audio_tokens") or 0
            else:
                stats["input_text_tokens"] = usage.get("prompt_tokens") or 0


        return stats


    def _remove_cache_boundary_markers(self, messages: list[dict]) -> list[dict]:
        """
        Remove <<cache-boundary>> markers from user messages.

        For providers with automatic (implicit) caching (OpenAI, Gemini), we don't need
        explicit cache markers. This function cleans up the markers from prompts.

        Args:
            messages: The list of messages to transform.

        Returns:
            A new list of messages with cache boundary markers removed.
        """
        result = []
        for message in messages:
            new_message = message.copy()
            if new_message.get("role") == "user":
                content = new_message.get("content", "")
                if isinstance(content, str) and self.CACHE_BOUNDARY_MARKER in content:
                    new_message["content"] = content.replace(self.CACHE_BOUNDARY_MARKER, "")
            result.append(new_message)
        return result


    def _transform_messages_for_anthropic(self, messages: list[dict]) -> list[dict]:
        """
        Add cache_control markers to system messages and user prompt prefixes for Anthropic models.

        This transforms messages to:
        1. Add cache_control to system message content blocks
        2. Convert user messages with <<cache-boundary>> marker into a single message with multiple content blocks:
            a. Static prefix block (with cache_control) - cacheable across records
            b. Dynamic content block (without cache_control) - changes per record

        Args:
            messages: The list of messages to transform.

        Returns:
            A new list of messages with cache_control markers added.
        """
        result = []
        for message in messages:
            new_message = copy.deepcopy(message)
            role = new_message.get("role")
            content = new_message.get("content", "")

            # 1. Handle System Messages
            if role == "system":
                if isinstance(content, str) and content:
                    new_message["content"] = [{
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }]
                elif isinstance(content, list) and content:
                    # Apply to last block if it's text
                    last_block = new_message["content"][-1]
                    if isinstance(last_block, dict) and last_block.get("type") == "text":
                        last_block["cache_control"] = {"type": "ephemeral"}

            # 2. Handle User Messages (The Split Logic)
            elif role == "user" and isinstance(content, str) and self.CACHE_BOUNDARY_MARKER in content:
                static, dynamic = content.split(self.CACHE_BOUNDARY_MARKER, 1)

                new_blocks = []
                if static.strip():
                    new_blocks.append({
                        "type": "text",
                        "text": static,
                        "cache_control": {"type": "ephemeral"}
                    })

                if dynamic.strip():
                    new_blocks.append({"type": "text", "text": dynamic})

                if new_blocks:
                    new_message["content"] = new_blocks
                else:
                    new_message["content"] = ""

            result.append(new_message)
        return result
