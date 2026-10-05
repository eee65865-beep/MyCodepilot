"""上传校验与真实 Agent/工具链路测试，不使用真实 API 密钥。"""

from io import BytesIO
import json
import shutil

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage, ToolMessage

from config import PROJECT_ROOT
from agent import code_agent
from tests.test_agent import ScriptedModel, tool_call
from services.uploads import decode_upload, build_upload_context, MAX_UPLOAD_BYTES, UPLOAD_EXTENSIONS


SOURCE = (PROJECT_ROOT / "examples" / "solution.cpp").read_text(encoding="utf-8")


@pytest.mark.parametrize("extension", sorted(UPLOAD_EXTENSIONS))
def test_allowed_types(extension):
    assert decode_upload(f"solution.{extension}", b"code").content == "code"


@pytest.mark.parametrize("name,data,error", [
    ("bad.exe", b"code", "不支持"),
    ("bad.py", b"\xff", "UTF-8"),
    ("empty.py", b"", "为空"),
    ("large.py", b"x" * (MAX_UPLOAD_BYTES + 1), "文件过大"),
], ids=["unsupported", "invalid-utf8", "empty", "too-large"])
def test_invalid_upload(name, data, error):
    with pytest.raises(ValueError, match=error):
        decode_upload(name, data)


def test_bom_and_size_boundary():
    assert decode_upload("code.py", b"\xef\xbb\xbfprint(1)").content == "print(1)"
    assert decode_upload("limit.py", b"x" * MAX_UPLOAD_BYTES).size == MAX_UPLOAD_BYTES


class FakeUpload(BytesIO):
    name = "solution.cpp"

    @property
    def size(self):
        return len(self.getvalue())


def test_upload_only_does_not_call_agent(monkeypatch):
    factory = []
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: factory.append(True))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: FakeUpload(SOURCE.encode()))
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    assert not app.exception
    assert factory == []
    assert app.session_state["messages"] == []
    assert any("已选择：solution.cpp" in text.value for text in app.text)


def test_explain_uploaded_cpp(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="读取整数并输出它的平方。")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: FakeUpload(SOURCE.encode()))
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("帮我解释这份代码。").run()
    assert not app.exception
    context = model.seen_messages[0][-1].content
    assert "文件名：solution.cpp" in context
    assert SOURCE in context
    assert "帮我解释这份代码" in context
    assert app.session_state["messages"][-1]["tools"] == []


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
def test_run_uploaded_cpp_with_stdin(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "cpp", "code": SOURCE, "stdin": "5\n"}),
        AIMessage(content="真实输出为 25。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: FakeUpload(SOURCE.encode()))
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("运行这份代码，输入是5。").run(timeout=15)
    assert not app.exception
    observation = model.seen_messages[1][-1]
    assert isinstance(observation, ToolMessage)
    assert json.loads(observation.content)["stdout"] == "25\n"
    assert app.session_state["messages"][-1]["tools"][0]["success"]


def test_invalid_upload_blocks_submission(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: FakeUpload(b"\xff"))
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.chat_input[0].set_value("解释代码").run()
    assert not app.exception
    assert app.session_state["messages"] == []
    assert any("UTF-8" in error.value for error in app.error)
