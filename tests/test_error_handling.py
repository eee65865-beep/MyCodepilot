"""错误分类、真实 SDK 有限重试、工具异常和安全日志回归。"""

import json
import logging

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from openai import AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
from streamlit.testing.v1 import AppTest

from agent import code_agent
from config import PROJECT_ROOT
from services.diagnostics import logger, log_failure
from tests.test_agent import ScriptedModel, tool_call
from tools import file_reader


@pytest.mark.parametrize("kind,status,expected", [
    (AuthenticationError, 401, "身份验证失败"),
    (RateLimitError, 429, "限流"),
    (APIStatusError, 402, "余额不足"),
    (APIStatusError, 503, "HTTP 503"),
    (APIConnectionError, None, "无法连接"),
    (APITimeoutError, None, "超时"),
    (RuntimeError, None, "调用失败"),
])
def test_agent_api_error_mapping(monkeypatch, kind, status, expected):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    if status:
        error = kind("secret-error-body", response=httpx.Response(status, request=request), body=None)
    elif kind in (APIConnectionError, APITimeoutError):
        error = kind(request=request)
    else:
        error = kind("secret-error-body")

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(ScriptedModel, "_generate", failed)
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: ScriptedModel(responses=[]))
    result = code_agent.CodePilotAgent().invoke("hello")
    assert not result.success
    assert expected in result.error
    assert "secret-error-body" not in result.error


def test_unexpected_tool_error_returns_observation(monkeypatch):
    @tool("run_code")
    def broken_tool(language: str, code: str) -> str:
        """Test tool that unexpectedly fails."""
        raise RuntimeError("secret-tool-body")

    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "python", "code": "print(1)"}),
        AIMessage(content="工具发生异常，未执行成功。"),
    ])
    monkeypatch.setattr(code_agent, "run_code", broken_tool)
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("run")
    assert result.success
    assert not result.tool_calls[0].success
    observation = model.seen_messages[1][-1]
    assert isinstance(observation, ToolMessage)
    assert observation.status == "error"
    assert "secret-tool-body" not in observation.content


def test_file_reader_unexpected_error(tmp_path, monkeypatch):
    from pathlib import Path

    def failed(*args, **kwargs):
        raise TypeError("secret-file-body")

    monkeypatch.setattr(file_reader, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(Path, "resolve", failed)
    result = json.loads(file_reader.read_code_file.invoke({"path": "file.py"}))
    assert not result["success"]
    assert "secret-file-body" not in result["error"]


def test_logs_do_not_include_exception_body_or_traceback(caplog):
    logger.addHandler(caplog.handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        log_failure("test", RuntimeError("DEEPSEEK_API_KEY=secret-value"))
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)
    assert any("RuntimeError" in record.getMessage() for record in caplog.records)
    assert all("secret-value" not in record.getMessage() for record in caplog.records)
    assert all(record.exc_info is None for record in caplog.records)


def test_ui_unexpected_exception_has_friendly_error(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("secret-ui-body")

    monkeypatch.setattr(code_agent.CodePilotAgent, "invoke", failed)
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("hello").run()
    assert not app.exception
    assert "请求处理失败" in app.error[0].value
    assert "secret-ui-body" not in app.error[0].value


@pytest.mark.parametrize("status,count", [(429, 3), (503, 3), (401, 1)])
def test_installed_sdk_retry_is_bounded(status, count, monkeypatch):
    # SDK 版本可能使用 httpx 或 httpx2，使用其实际导入的 HTTP 模块。
    from openai import _base_client
    transport_module = _base_client.httpx2 if hasattr(_base_client, "httpx2") else _base_client.httpx
    calls = []

    def respond(request):
        calls.append(request)
        return transport_module.Response(status, json={"error": {"message": "test", "type": "test"}})

    # 跳过 SDK 的等待，不改变重试次数和判定逻辑。
    monkeypatch.setattr(_base_client.time, "sleep", lambda seconds: None)
    with transport_module.Client(transport=transport_module.MockTransport(respond)) as client:
        model = ChatDeepSeek(model="deepseek-chat", api_key="test-placeholder", timeout=30, max_retries=2, http_client=client)
        with pytest.raises(APIStatusError):
            model.invoke("test")
    assert len(calls) == count
