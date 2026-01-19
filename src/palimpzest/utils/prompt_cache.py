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

class PromptCacheManager:
    """
    Manages prompt caching configurations and message transformations for LLM providers.
    
    This class handles:
    1. Session-level state (e.g., OpenAI cache keys).
    2. Provider-specific request arguments (headers, extra_body).
    3. In-place modification of messages for providers requiring explicit markers (Anthropic).
    4. Normalization of usage statistics.
    """
    
    CACHE_BOUNDARY_MARKER = "<<cache-boundary>>"

    def __init__(self, model: Model):
        self.model = model
        # Instance-level state ensures thread safety if we use one manager per plan/execution
        self.openai_cache_key = f"pz-cache-{uuid.uuid4().hex[:12]}" if self.model.is_openai_model() else None

    def get_cache_kwargs(self, messages: list[dict]) -> dict[str, Any]:
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
        if not self.model.supports_prompt_caching():
            return {}
        # TODO: Update with changes from #265
        if self.model.is_anthropic_model():
            # Anthropic: Explicit cache_control with ephemeral type
            # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
            # Mark system messages with cache_control (modifies messages in-place)
            self._transform_messages_for_anthropic(messages)
            return {}
        # implicit caching for Deepseek/Gemini/Openai Models that current support caching
        elif self.model.is_openai_model():
            # OpenAI: Automatic prefix caching based on matching prefixes
            # Use prompt_cache_key for sticky routing to the same cache shard
            # https://platform.openai.com/docs/guides/prompt-caching
            self._remove_cache_boundary_markers(messages)
            return {"extra_body": {"prompt_cache_key": self.openai_cache_key}}
        elif self.model.is_google_ai_studio_model() or self.model.is_vertex_model():
            # Gemini: Implicit caching (automatic prefix matching)
            # No additional kwargs needed - caching is automatic
            # https://ai.google.dev/gemini-api/docs/caching
            self._remove_cache_boundary_markers(messages)
            return {}
        elif self.model.is_deepseek_model():
            # DeepSeek: Automatic context caching (enabled by default)
            # No special parameters needed - caching happens automatically
            # Minimum cacheable unit is 64 tokens
            # https://api-docs.deepseek.com/guides/kv_cache
            self._remove_cache_boundary_markers(messages)
            return {}
        return {}


    def extract_cache_stats(self, usage: Dict, model: Model) -> Dict[str, int]:
        """
        Normalize cache statistics from provider-specific response formats.
        """
        stats = {
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "audio_cache_creation_tokens": 0,
            "audio_cache_read_tokens": 0,
        }

        if not model.supports_prompt_caching() or not usage:
            return stats

        if model.is_openai_model():
            details = usage.get("prompt_tokens_details", {}) or {}
            stats["cache_read_tokens"] = details.get("cached_tokens", 0)
            stats["audio_cache_read_tokens"] = details.get("audio_cached_tokens", 0)

        elif model.is_anthropic_model():
            stats["cache_creation_tokens"] = usage.get("cache_creation_input_tokens", 0)
            stats["cache_read_tokens"] = usage.get("cache_read_input_tokens", 0)

        elif model.is_vertex_model() or model.is_google_ai_studio_model():
            stats["cache_read_tokens"] = usage.get("cached_content_token_count", 0)

        elif model.is_deepseek_model():
            stats["cache_read_tokens"] = usage.get("prompt_cache_hit_tokens", 0)
            stats["cache_creation_tokens"] = 0

        return stats


    def _remove_cache_boundary_markers(self, messages: List[Dict]) -> None:
        """
        Remove <<cache-boundary>> markers from user messages.

        For providers with automatic (implicit) caching (OpenAI, Gemini, DeepSeek), we don't need
        explicit cache markers. This function cleans up the markers from prompts.

        Args:
            messages: The list of messages to modify (modified in-place)
        """
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str) and self.CACHE_BOUNDARY_MARKER in content:
                    message["content"] = content.replace(self.CACHE_BOUNDARY_MARKER, "")


    def _transform_messages_for_anthropic(self, messages: List[Dict]) -> None:
        """
        Add cache_control markers to system messages and user prompt prefixes for Anthropic models.

        This modifies the messages list in-place to:
        1. Add cache_control to system message content blocks
        2. Convert user messages with <<cache-boundary>> marker into a single message with multiple content blocks:
            a. Static prefix block (with cache_control) - cacheable across records
            b. Dynamic content block (without cache_control) - changes per record

        Args:
            messages: The list of messages to modify (modified in-place)     
        """
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            # 1. Handle System Messages
            if role == "system":
                if isinstance(content, str) and content:
                    message["content"] = [{
                        "type": "text", 
                        "text": content, 
                        "cache_control": {"type": "ephemeral"}
                    }]
                elif isinstance(content, list) and content:
                    # Apply to last block if it's text
                    last_block = content[-1]
                    if isinstance(last_block, dict) and last_block.get("type") == "text":
                        last_block["cache_control"] = {"type": "ephemeral"}

            # 2. Handle User Messages (The Split Logic)
            elif role == "user":
                if isinstance(content, str) and self.CACHE_BOUNDARY_MARKER in content:
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
                        message["content"] = new_blocks
                    else:
                        message["content"] = "" # Handle empty case gracefully
