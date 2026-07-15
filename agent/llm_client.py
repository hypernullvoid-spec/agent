"""
Back-compat shim — the original Phase 1 `LLMClient` now routes through the
provider-agnostic layer in agent/llm/. Existing imports keep working, but
`model` may now be any BYO-LLM spec ("openai:gpt-4o", "ollama:llama3.1", …).
"""

from agent.llm import create_client


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self._client = create_client(model)
        self.model = model

    def call(self, system: str, messages: list, tools: list, max_tokens: int = 8192):
        return self._client.call(system, messages, tools=tools, max_tokens=max_tokens)

    @property
    def total_usage(self):
        return self._client.total_usage
