"""LLM layer: model-spec parsing, normalization, message conversion, mock client."""

from agent.llm.base import TextBlock, ToolUseBlock, block_to_dict
from agent.llm.mock_client import MockLLMClient, text_response, tool_response
from agent.llm.openai_client import OpenAICompatClient
from agent.llm.router import parse_model_spec


def test_parse_model_spec_defaults():
    assert parse_model_spec("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6", None)


def test_parse_model_spec_provider_and_url():
    assert parse_model_spec("ollama:llama3.1") == ("ollama", "llama3.1", None)
    p, m, u = parse_model_spec("vllm:meta-llama/Llama-3.1-70B@http://gpu:8000/v1")
    assert (p, m, u) == ("vllm", "meta-llama/Llama-3.1-70B", "http://gpu:8000/v1")


def test_block_serialization_roundtrip():
    t = TextBlock(text="hi")
    tu = ToolUseBlock(id="x1", name="run", input={"a": 1})
    assert block_to_dict(t) == {"type": "text", "text": "hi"}
    assert block_to_dict(tu)["name"] == "run"
    assert block_to_dict({"type": "tool_result", "tool_use_id": "x1", "content": "ok"})["type"] == "tool_result"


def test_mock_client_script_and_recording():
    mock = MockLLMClient(script=["hello", tool_response("t", {"k": "v"}, text="plan")])
    r1 = mock.call("sys", [{"role": "user", "content": "hi"}])
    assert r1.text == "hello"
    r2 = mock.call("sys", [{"role": "user", "content": "again"}])
    assert r2.tool_uses()[0].name == "t"
    assert len(mock.calls) == 2
    assert mock.total_usage.calls == 2


def test_openai_message_conversion():
    messages = [
        {"role": "user", "content": "solve it"},
        {"role": "assistant", "content": [
            TextBlock(text="thinking"),
            ToolUseBlock(id="c1", name="run_python", input={"code": "print(1)"}),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "1"},
        ]},
    ]
    out = OpenAICompatClient._convert_messages("SYS", messages)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "solve it"}
    asst = out[2]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["function"]["name"] == "run_python"
    tool_msg = out[3]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"


def test_openai_tool_conversion():
    tools = [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    conv = OpenAICompatClient._convert_tools(tools)
    assert conv[0]["type"] == "function" and conv[0]["function"]["name"] == "f"
    assert OpenAICompatClient._convert_tool_choice({"type": "tool", "name": "f"}) == {
        "type": "function", "function": {"name": "f"}}
    assert OpenAICompatClient._convert_tool_choice({"type": "any"}) == "required"


def test_retry_gives_up_on_non_retryable():
    class Boom(MockLLMClient):
        def _call_api(self, *a, **k):
            raise ValueError("invalid api key")
    c = Boom()
    try:
        c.call("s", [{"role": "user", "content": "x"}])
        assert False, "should have raised"
    except Exception as e:
        assert "invalid api key" in str(e)
