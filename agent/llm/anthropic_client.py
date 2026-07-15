"""Anthropic backend — native Messages API with prompt caching."""

from __future__ import annotations

import os
from typing import Optional

from agent.llm.base import (
    BaseLLMClient, LLMResponse, TextBlock, ToolUseBlock, Usage, block_to_dict,
)


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, enable_caching: bool = True):
        super().__init__(model)
        from anthropic import Anthropic  # lazy import — only needed for this backend
        kwargs: dict = {"api_key": api_key or os.environ.get("ANTHROPIC_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)
        self.enable_caching = enable_caching

    @staticmethod
    def _serialize_messages(messages: list) -> list:
        """Our normalized blocks → plain dicts the SDK accepts."""
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = [block_to_dict(b) for b in content]
            out.append({"role": m["role"], "content": content})
        return out

    def _call_api(self, system, messages, tools, max_tokens, temperature, tool_choice) -> LLMResponse:
        # Prompt caching: mark the system prompt (and the tool list, which the
        # API caches together with it) so long agent loops reuse the prefix.
        if self.enable_caching:
            system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            system_param = system

        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_param,
            messages=self._serialize_messages(messages),
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        resp = self.client.messages.create(**kwargs)

        content: list = []
        for b in resp.content:
            if b.type == "text":
                content.append(TextBlock(text=b.text))
            elif b.type == "tool_use":
                content.append(ToolUseBlock(id=b.id, name=b.name, input=b.input))

        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            calls=1,
        )
        return LLMResponse(content=content, stop_reason=resp.stop_reason, model=self.model, usage=usage)
