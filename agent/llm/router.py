"""
BYO-LLM router — one string picks provider + model.

Model spec format:  [provider:]model[@base_url]

  "claude-sonnet-4-6"                          → Anthropic (default provider)
  "anthropic:claude-opus-4-8"                  → Anthropic
  "openai:gpt-4o"                              → OpenAI
  "ollama:llama3.1"                            → local Ollama (localhost:11434)
  "vllm:meta-llama/Llama-3.1-70B@http://gpu:8000/v1"
  "gemini:gemini-2.5-pro"                      → Gemini OpenAI-compat endpoint
  "groq:llama-3.3-70b-versatile"               → Groq
  "openai-compat:mymodel@https://my-endpoint/v1"

Environment overrides:
  SWARN_MODEL           default model spec when none is given
  SWARN_LLM_BASE_URL    base_url override for the chosen provider
  <PROVIDER>_API_KEY  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY /
                      GROQ_API_KEY, etc.
"""

from __future__ import annotations

import os
from typing import Optional

from agent.llm.base import BaseLLMClient

DEFAULT_MODEL = "claude-sonnet-4-6"

# provider → (default_base_url, api_key_env)
_OPENAI_COMPAT_PRESETS: dict[str, tuple[Optional[str], str]] = {
    "openai":        (None, "OPENAI_API_KEY"),
    "ollama":        ("http://localhost:11434/v1", "OLLAMA_API_KEY"),
    "vllm":          ("http://localhost:8000/v1", "VLLM_API_KEY"),
    "gemini":        ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq":          ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together":      ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openai-compat": (None, "OPENAI_API_KEY"),
}


def parse_model_spec(spec: Optional[str]) -> tuple[str, str, Optional[str]]:
    """'provider:model@url' → (provider, model, base_url)."""
    spec = spec or os.environ.get("SWARN_MODEL") or DEFAULT_MODEL
    base_url = None
    if "@" in spec:
        spec, base_url = spec.rsplit("@", 1)
    if ":" in spec:
        provider, model = spec.split(":", 1)
        provider = provider.lower()
    else:
        provider, model = "anthropic", spec
    base_url = base_url or os.environ.get("SWARN_LLM_BASE_URL")
    return provider, model, base_url


_client_cache: dict[str, BaseLLMClient] = {}


def create_client(spec: Optional[str] = None, cache: bool = True) -> BaseLLMClient:
    """Create (or reuse) the right client for a model spec string."""
    provider, model, base_url = parse_model_spec(spec)
    key = f"{provider}:{model}@{base_url}"
    if cache and key in _client_cache:
        return _client_cache[key]

    client: BaseLLMClient
    if provider == "anthropic":
        from agent.llm.anthropic_client import AnthropicClient
        client = AnthropicClient(model=model, base_url=base_url)
    elif provider == "mock":
        from agent.llm.mock_client import MockLLMClient
        client = MockLLMClient(model=model)
    elif provider in _OPENAI_COMPAT_PRESETS:
        from agent.llm.openai_client import OpenAICompatClient
        preset_url, key_env = _OPENAI_COMPAT_PRESETS[provider]
        client = OpenAICompatClient(
            model=model,
            base_url=base_url or preset_url,
            api_key=os.environ.get(key_env),
        )
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Use one of: anthropic, "
            f"{', '.join(_OPENAI_COMPAT_PRESETS)}, mock — or 'openai-compat:<model>@<base_url>'."
        )

    if cache:
        _client_cache[key] = client
    return client
