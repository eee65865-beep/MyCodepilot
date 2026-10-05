"""用脚本化模型验证真正的 LangChain Agent 循环，工具仍真实执行。"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent import code_agent
from llm.deepseek import DeepSeekConfigurationError


class ScriptedModel(BaseChatModel):
    """测试模型按顺序返回响应，不声称验证了 DeepSeek 自主决策。"""

    responses: list[AIMessage]
    seen_messages: list = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        self.seen_messages.append(list(messages))
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def tool_call(name, args, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": call_id, "type": "tool_call"
    }])


def test_file_reader_loop(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("read_code_file", {"path": "examples/bug.py"}),
        AIMessage(content="循环多访问一个索引，存在 IndexError。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("读取 examples/bug.py 并分析代码问题。")
    assert result.success
    assert model.bound_tools == ["read_code_file", "run_code"]
    assert len(model.seen_messages) == 2
    observation = model.seen_messages[1][-1]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == "call-1"
    assert "range(len(numbers) + 1)" in json.loads(observation.content)["content"]
    assert result.tool_calls[0].name == "read_code_file"
    assert result.tool_calls[0].success
    assert "IndexError" in result.answer


def test_code_runner_loop(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "python", "code": "print(1 + 2)"}),
        AIMessage(content="实际输出为 3。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("运行以下Python代码并告诉我输出：print(1 + 2)")
    assert result.success
    observation = model.seen_messages[1][-1]
    assert isinstance(observation, ToolMessage)
    assert json.loads(observation.content)["stdout"] == "3\n"
    assert result.tool_calls[0].name == "run_code"
    assert result.tool_calls[0].success


def test_multiple_tools(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("read_code_file", {"path": "examples/bug.py"}, "read"),
        tool_call("run_code", {"language": "python", "code": "print(1 + 2)"}, "run"),
        AIMessage(content="读取和执行完成。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("读取文件，再执行一个演示。")
    assert result.success
    assert [record.name for record in result.tool_calls] == ["read_code_file", "run_code"]
    assert len(model.seen_messages) == 3
    assert sum(isinstance(m, ToolMessage) for m in model.seen_messages[2]) == 2


def test_direct_answer_and_hidden_reasoning(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(
        content=[{"type": "reasoning", "reasoning": "hidden"}, {"type": "text", "text": "代码输出 3。"}],
        additional_kwargs={"reasoning_content": "hidden"},
    )])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("解释 print(1 + 2)")
    assert result.success
    assert result.tool_calls == []
    assert result.answer == "代码输出 3。"


def test_tool_failure_returns_to_model(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("read_code_file", {"path": "missing.py"}),
        AIMessage(content="文件不存在，无法分析。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("读取 missing.py")
    assert result.success  # Agent 完成回答，但工具失败。
    assert not result.tool_calls[0].success
    assert not json.loads(model.seen_messages[1][-1].content)["success"]


def test_missing_key(monkeypatch):
    def missing():
        raise DeepSeekConfigurationError("未配置DEEPSEEK_API_KEY，请在.env文件中设置。")

    monkeypatch.setattr(code_agent, "create_deepseek_model", missing)
    result = code_agent.CodePilotAgent().invoke("hello")
    assert not result.success
    assert "未配置DEEPSEEK_API_KEY" in result.error


def test_api_error_preserves_completed_tools(monkeypatch):
    model = ScriptedModel(responses=[tool_call("run_code", {"language": "python", "code": "print(3)"})])
    # 第二次模型调用耗尽脚本并报错，模拟工具完成后模型请求失败。
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("run")
    assert not result.success
    assert result.tool_calls[0].success
    assert result.error


def test_recursion_limit(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "python", "code": "print(3)"}, str(index))
        for index in range(10)
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent(recursion_limit=4).invoke("run")
    assert not result.success
    assert result.error == "Agent执行步骤超过限制，请简化请求后重试。"


def test_empty_messages():
    assert not code_agent.CodePilotAgent().invoke([]).success


def test_observable_events(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "python", "code": "print(3)"}),
        AIMessage(content="输出 3。", additional_kwargs={"reasoning_content": "hidden"}),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    events = []
    result = code_agent.CodePilotAgent().invoke("run", on_event=events.append)
    assert result.success
    assert [event["type"] for event in events] == ["tool_start", "tool_end"]
    assert events[0]["name"] == "run_code"
    assert events[1]["success"]
    assert "hidden" not in json.dumps(events)
