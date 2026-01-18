"""
Prompt caching utility for different LLM providers.

This module provides provider-specific prompt caching configurations:
- OpenAI: Automatic prefix caching with prompt_cache_key for sticky routing
- Gemini (Google AI Studio / Vertex AI): Implicit caching (automatic prefix matching)
- Anthropic: Explicit cache_control with ephemeral type on system and user message content
- DeepSeek: Automatic context caching (enabled by default, 64 token minimum)
"""

import uuid
from typing import Any

from palimpzest.constants import Model

# Session-level cache key for OpenAI sticky routing
# This ensures requests within the same session are routed to the same cache shard
_OPENAI_CACHE_KEY: str | None = None


def _get_openai_cache_key() -> str:
    """Get or create a session-level cache key for OpenAI."""
    global _OPENAI_CACHE_KEY
    if _OPENAI_CACHE_KEY is None:
        _OPENAI_CACHE_KEY = f"pz-cache-{uuid.uuid4().hex[:12]}"
    return _OPENAI_CACHE_KEY


def get_cache_kwargs(model: Model, messages: list[dict]) -> dict[str, Any]:
    """
    Get provider-specific cache configuration kwargs for litellm.completion().

    This function may modify the messages list in-place to add cache control
    markers for providers that require explicit cache annotations (Anthropic).

    Args:
        model: The Model enum representing the LLM being used
        messages: The list of messages being sent to the model (may be modified in-place)

    Returns:
        A dictionary of kwargs to pass to litellm.completion() for enabling caching
    """
    if not model.supports_prompt_caching():
        return {}

    cache_kwargs = {}

    if model.is_openai_model():
        # OpenAI: Automatic prefix caching based on matching prefixes
        # Use prompt_cache_key for sticky routing to the same cache shard
        # https://platform.openai.com/docs/guides/prompt-caching
        cache_kwargs["extra_body"] = {"prompt_cache_key": _get_openai_cache_key()}
        # Clean up cache boundary markers from user messages (not used by OpenAI)
        _remove_cache_boundary_markers(messages)

    elif model.is_anthropic_model():
        # Anthropic: Explicit cache_control with ephemeral type
        # We need to:
        # 1. Add the anthropic-beta header to enable prompt caching
        # 2. Mark system message content with cache_control
        # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        cache_kwargs["extra_headers"] = {
            "anthropic-beta": "prompt-caching-2024-07-31"
        }
        # Mark system messages with cache_control (modifies messages in-place)
        _add_anthropic_cache_control(messages)

    elif model.is_vertex_model() or model.is_google_ai_studio_model():
        # Gemini: Implicit caching (automatic prefix matching)
        # No additional kwargs needed - caching is automatic
        # https://ai.google.dev/gemini-api/docs/caching
        # Clean up cache boundary markers from user messages (not used by Gemini)
        _remove_cache_boundary_markers(messages)

    elif model.is_deepseek_model():
        # DeepSeek: Automatic context caching (enabled by default)
        # No special parameters needed - caching happens automatically
        # Minimum cacheable unit is 64 tokens
        # https://api-docs.deepseek.com/guides/kv_cache
        # Clean up cache boundary markers from user messages (not used by DeepSeek)
        _remove_cache_boundary_markers(messages)

    return cache_kwargs


# Marker used to identify the boundary between static and dynamic content in user prompts
CACHE_BOUNDARY_MARKER = "<<cache-boundary>>"


def _add_anthropic_cache_control(messages: list[dict]) -> None:
    """
    Add cache_control markers to system messages and user prompt prefixes for Anthropic models.

    This modifies the messages list in-place to:
    1. Add cache_control to system message content blocks
    2. Convert user messages with <<cache-boundary>> marker into content blocks,
       with cache_control on the static prefix

    Args:
        messages: The list of messages to modify (modified in-place)
    """
    # Handle system messages - add cache_control
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content", "")
            if isinstance(content, str) and content:
                # Convert string content to content block with cache_control
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            elif isinstance(content, list) and content:
                # Add cache_control to the last content block
                last_block = content[-1]
                if isinstance(last_block, dict) and last_block.get("type") == "text":
                    last_block["cache_control"] = {"type": "ephemeral"}

    # Handle user messages - convert to content blocks with cache_control on static prefix
    _convert_user_messages_to_content_blocks(messages)


def _convert_user_messages_to_content_blocks(messages: list[dict]) -> None:
    """
    Convert user messages with cache boundary markers into content blocks for Anthropic.

    For Anthropic, we want to cache the static prefix of user prompts. This function
    converts user messages containing the <<cache-boundary>> marker into a single message
    with multiple content blocks:
    1. Static prefix block (with cache_control) - cacheable across records
    2. Dynamic content block (without cache_control) - changes per record

    This matches Anthropic's expected format where a single message can have multiple
    content blocks with different cache settings.

    Args:
        messages: The list of messages to modify (modified in-place)
    """
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str) and CACHE_BOUNDARY_MARKER in content:
                # Split at the cache boundary marker
                static_prefix, dynamic_content = content.split(CACHE_BOUNDARY_MARKER, 1)

                # Create content blocks within a single message
                content_blocks = []

                if static_prefix.strip():
                    # Static prefix block with cache_control
                    content_blocks.append({
                        "type": "text",
                        "text": static_prefix,
                        "cache_control": {"type": "ephemeral"}
                    })

                if dynamic_content.strip():
                    # Dynamic content block without cache_control
                    content_blocks.append({
                        "type": "text",
                        "text": dynamic_content
                    })

                # Update message content to use content blocks
                if content_blocks:
                    message["content"] = content_blocks
                else:
                    # Fallback: remove the marker if both parts are empty
                    message["content"] = ""


def _remove_cache_boundary_markers(messages: list[dict]) -> None:
    """
    Remove <<cache-boundary>> markers from user messages.

    For providers with automatic caching (OpenAI, Gemini, DeepSeek), we don't need
    explicit cache markers. This function cleans up the markers from prompts.

    Args:
        messages: The list of messages to modify (modified in-place)
    """
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str) and CACHE_BOUNDARY_MARKER in content:
                message["content"] = content.replace(CACHE_BOUNDARY_MARKER, "")


def extract_cache_stats_from_usage(usage: dict, model: Model) -> dict[str, int]:
    """
    Extract cache-related statistics from the LiteLLM usage response.

    Different providers return cache stats in different formats:
    - OpenAI: cached_tokens in prompt_tokens_details
    - Anthropic: cache_creation_input_tokens, cache_read_input_tokens
    - Gemini: cached_content_token_count
    - DeepSeek: prompt_cache_hit_tokens, prompt_cache_miss_tokens

    Args:
        usage: The usage dictionary from litellm completion response
        model: The Model enum representing the LLM being used

    Returns:
        A dictionary with cache_creation_tokens and cache_read_tokens
    """
    cache_stats = {
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "audio_cache_creation_tokens": 0,
        "audio_cache_read_tokens": 0,
    }

    if not model.supports_prompt_caching():
        return cache_stats

    # Try to extract cache stats from various provider formats
    prompt_tokens_details = usage.get("prompt_tokens_details", {}) or {}

    if model.is_openai_model():
        # OpenAI returns cached_tokens in prompt_tokens_details
        cache_stats["cache_read_tokens"] = prompt_tokens_details.get("cached_tokens", 0)
        # OpenAI doesn't charge separately for cache creation
        cache_stats["cache_creation_tokens"] = 0
        # Audio cache tokens (if present)
        cache_stats["audio_cache_read_tokens"] = prompt_tokens_details.get("audio_cached_tokens", 0)

    elif model.is_anthropic_model():
        # Anthropic returns cache_creation_input_tokens and cache_read_input_tokens
        cache_stats["cache_creation_tokens"] = usage.get("cache_creation_input_tokens", 0)
        cache_stats["cache_read_tokens"] = usage.get("cache_read_input_tokens", 0)

    elif model.is_vertex_model() or model.is_google_ai_studio_model():
        # Gemini returns cached_content_token_count
        cache_stats["cache_read_tokens"] = usage.get("cached_content_token_count", 0)
        # Gemini's implicit caching doesn't have separate creation tokens
        cache_stats["cache_creation_tokens"] = 0

    elif model.is_deepseek_model():
        # DeepSeek returns prompt_cache_hit_tokens and prompt_cache_miss_tokens
        cache_stats["cache_read_tokens"] = usage.get("prompt_cache_hit_tokens", 0)
        # DeepSeek's automatic caching doesn't have separate creation tokens
        cache_stats["cache_creation_tokens"] = 0

    return cache_stats
