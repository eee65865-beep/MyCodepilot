"""Streamlit 会话隔离、跨轮保存、清空和错误展示。"""

from streamlit.testing.v1 import AppTest

from agent import code_agent
from langchain_core.messages import AIMessage
from tests.test_agent import ScriptedModel
from config import PROJECT_ROOT
from tests.test_agent import tool_call
from llm.deepseek import DeepSeekConfigurationError
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError


def test_chat_memory_and_clear(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="add 返回两数之和。"), AIMessage(content="O(1)。")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    original_agent = app.session_state["agent"]
    original_upload_generation = app.session_state["upload_generation"]
    app.chat_input[0].set_value("分析 def add(a,b): return a+b").run()
    app.chat_input[0].set_value("它的时间复杂度呢？").run()
    assert not app.exception
    assert len(app.chat_message) == 4
    assert "def add" in model.seen_messages[1][1].content
    assert len(app.session_state["memory"].messages) == 4
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["messages"] == []
    assert app.session_state["memory"].messages == []
    assert app.session_state["agent"] is not original_agent
    assert app.session_state["upload_generation"] > original_upload_generation


def test_sessions_are_separate():
    first = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    second = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    assert first.session_state["memory"] is not second.session_state["memory"]
    assert first.session_state["agent"] is not second.session_state["agent"]


def test_tool_status_and_final_answer(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "python", "code": "print(3)"}),
        AIMessage(content="真实输出为 3。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("运行 print(3)").run(timeout=10)
    assert not app.exception
    text = "\n".join(element.value for element in app.markdown)
    assert "正在调用工具：run_code" in text
    assert "run_code 执行成功" in text
    assert "Agent Answer" in text
    assert "真实输出为 3" in text
    assert app.status[0].state == "complete"


def test_api_failure_shows_error(monkeypatch):
    def missing():
        raise DeepSeekConfigurationError("未配置DEEPSEEK_API_KEY，请在.env文件中设置。")

    monkeypatch.setattr(code_agent, "create_deepseek_model", missing)
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("hello").run()
    assert not app.exception
    assert "未配置DEEPSEEK_API_KEY" in app.error[0].value
    assert app.status[0].state == "error"


def test_tool_failure_is_visible(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("read_code_file", {"path": "missing.py"}),
        AIMessage(content="文件不存在。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("读取 missing.py").run()
    assert not app.exception
    assert "read_code_file 执行失败" in app.error[0].value


def test_sidebar_settings_apply_and_survive_clear(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="done"), AIMessage(content="next")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.number_input(key="agent_steps").set_value(6).run()
    app.number_input(key="history_turns").set_value(1).run()
    app.chat_input[0].set_value("first").run()
    app.chat_input[0].set_value("second").run()
    assert not app.exception
    assert app.session_state["agent"].recursion_limit == 6
    assert len(app.session_state["memory"].messages) == 2
    assert len(model.seen_messages[1]) == 2
    app.button[0].click().run()
    assert app.session_state["agent"].recursion_limit == 6
    assert app.session_state["memory"].max_turns == 1
    assert app.session_state["messages"] == []


def test_streamlit_server_starts_without_model_request():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py"),
        "--server.headless=true", f"--server.port={port}", "--browser.gatherUsageStats=false",
    ], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            assert process.poll() is None, "Streamlit exited before readiness"
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                    assert response.read() == b"ok"
                    return
            except URLError:
                time.sleep(0.1)
        raise AssertionError("Streamlit did not become ready")
    finally:
        process.terminate()
        process.wait(timeout=5)
