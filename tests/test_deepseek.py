"""不需要真实 API Key 或网络的配置与错误处理测试。"""

from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

import config
from llm import deepseek


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """确保测试不会读取用户真实 .env 或修改已有配置。"""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    # 先记录原值再删除，确保 dotenv 新增的变量也被 monkeypatch 恢复。
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"):
        monkeypatch.setenv(name, "test-placeholder")
        monkeypatch.delenv(name)
    return tmp_path


def test_missing_key(isolated_env):
    with pytest.raises(deepseek.DeepSeekConfigurationError, match="未配置DEEPSEEK_API_KEY"):
        deepseek.create_deepseek_model()
    result = deepseek.invoke_deepseek("hello")
    assert not result.success
    assert result.error == "未配置DEEPSEEK_API_KEY，请在.env文件中设置。"


def test_dotenv_and_defaults(isolated_env):
    # 此字符串只是测试占位值，不是真实密钥；模型构建不会联网。
    (isolated_env / ".env").write_text(
        "DEEPSEEK_API_KEY=test-placeholder\nDEEPSEEK_MODEL=\n", encoding="utf-8"
    )
    model = deepseek.create_deepseek_model()
    assert model.model == "deepseek-chat"
    assert model.temperature == 0
    assert model.request_timeout == 30
    assert model.max_retries == 2


def test_environment_has_priority(isolated_env, monkeypatch):
    (isolated_env / ".env").write_text(
        "DEEPSEEK_API_KEY=test-placeholder\nDEEPSEEK_MODEL=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-environment")
    assert deepseek.create_deepseek_model().model == "from-environment"


def test_success(monkeypatch):
    model = Mock()
    model.invoke.return_value = AIMessage(content="递归是函数调用自身来解决问题。")
    monkeypatch.setattr(deepseek, "create_deepseek_model", lambda: model)
    prompt = "请用一句话解释Python中的递归"
    result = deepseek.invoke_deepseek(prompt)
    assert result.success
    assert result.message.content == "递归是函数调用自身来解决问题。"
    assert result.error is None
    model.invoke.assert_called_once_with(prompt)


def test_empty_prompt(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(deepseek, "create_deepseek_model", factory)
    assert not deepseek.invoke_deepseek("  ").success
    factory.assert_not_called()


@pytest.mark.parametrize(
    "error_type,status,expected",
    [
        (AuthenticationError, 401, "身份验证失败"),
        (RateLimitError, 429, "限流"),
        (APIStatusError, 402, "余额不足"),
        (APIStatusError, 503, "HTTP 503"),
        (APITimeoutError, None, "超时"),
        (APIConnectionError, None, "无法连接"),
        (RuntimeError, None, "调用失败"),
    ],
)
def test_api_errors(monkeypatch, error_type, status, expected):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    if status is not None:
        exc = error_type(
            "sensitive-placeholder", response=httpx.Response(status, request=request), body=None
        )
    elif error_type is APITimeoutError:
        exc = error_type(request=request)
    elif error_type is APIConnectionError:
        exc = error_type(request=request)
    else:
        exc = error_type("sensitive-placeholder")

    model = Mock()
    model.invoke.side_effect = exc
    monkeypatch.setattr(deepseek, "create_deepseek_model", lambda: model)
    result = deepseek.invoke_deepseek("hello")
    assert not result.success
    assert result.message is None
    assert expected in result.error
    assert "sensitive-placeholder" not in result.error
