"""通过 LangChain invoke 验证工具行为，不需要模型或网络。"""

import json
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from tools import file_reader


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(file_reader, "WORKSPACE_ROOT", root)
    return root


def invoke(path):
    return json.loads(file_reader.read_code_file.invoke({"path": path}))


def test_normal_read(workspace):
    (workspace / "examples").mkdir()
    content = "print('你好')\n"
    (workspace / "examples" / "hello.py").write_bytes(content.encode("utf-8"))
    assert invoke("examples/hello.py") == {
        "success": True,
        "path": "examples/hello.py",
        "content": content,
        "error": None,
    }


def test_missing_file(workspace):
    result = invoke("missing.py")
    assert not result["success"]
    assert result["content"] is None
    assert "文件不存在" in result["error"]


@pytest.mark.parametrize("path", ["../outside.py", "../../outside.py", "..\\outside.py", "sub/../hello.py"])
def test_path_traversal(workspace, path):
    (workspace.parent / "outside.py").write_text("secret", encoding="utf-8")
    result = invoke(path)
    assert not result["success"]
    assert "路径穿越" in result["error"]
    assert result["content"] is None


def test_invalid_extension(workspace):
    (workspace / "secret.txt").write_text("secret", encoding="utf-8")
    assert "非法扩展名" in invoke("secret.txt")["error"]


def test_empty_file(workspace):
    (workspace / "empty.py").touch()
    result = invoke("empty.py")
    assert not result["success"]
    assert "文件为空" in result["error"]


@pytest.mark.parametrize("extension", sorted(file_reader.ALLOWED_EXTENSIONS))
def test_allowed_extensions(workspace, extension):
    path = workspace / f"source{extension}"
    path.write_text("example code", encoding="utf-8")
    assert invoke(path.name)["success"]


def test_absolute_path_inside_workspace(workspace):
    path = workspace / "hello.py"
    path.write_text("print(1)", encoding="utf-8")
    assert invoke(str(path))["success"]


def test_absolute_path_outside_workspace(workspace):
    # 与 workspace 共享字符串前缀，但不是它的子目录。
    outside = workspace.parent / "workspace-other"
    outside.mkdir()
    path = outside / "secret.py"
    path.write_text("secret", encoding="utf-8")
    result = invoke(str(path))
    assert not result["success"]
    assert "路径越界" in result["error"]


def test_large_file(workspace):
    (workspace / "large.py").write_bytes(b"x" * (file_reader.MAX_FILE_SIZE + 1))
    assert "文件过大" in invoke("large.py")["error"]


def test_exact_size_limit(workspace):
    (workspace / "limit.py").write_bytes(b"x" * file_reader.MAX_FILE_SIZE)
    assert invoke("limit.py")["success"]


def test_encoding_error(workspace):
    (workspace / "invalid.py").write_bytes(b"\xff\xfe")
    assert "编码错误" in invoke("invalid.py")["error"]


def test_utf8_bom(workspace):
    (workspace / "bom.py").write_text("print(1)", encoding="utf-8-sig")
    assert invoke("bom.py")["content"] == "print(1)"


def test_permission_error(workspace, monkeypatch):
    (workspace / "private.py").write_text("print(1)", encoding="utf-8")

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied)
    assert "没有权限" in invoke("private.py")["error"]


def test_symlink_escape(workspace):
    outside = workspace.parent / "secret.py"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("当前系统没有创建符号链接的权限。")
    assert "路径越界" in invoke("link.py")["error"]


@pytest.mark.parametrize("path", ["", "\x00.py"])
def test_invalid_path(workspace, path):
    assert not invoke(path)["success"]


def test_directory_is_not_file(workspace):
    (workspace / "directory.py").mkdir()
    assert "不是普通文件" in invoke("directory.py")["error"]


@pytest.mark.parametrize("args", [{}, {"path": None}, {"path": 123}])
def test_invalid_arguments(args):
    result = json.loads(file_reader.read_code_file.invoke(args))
    assert not result["success"]
    assert "参数错误" in result["error"]


def test_langchain_metadata():
    tool = file_reader.read_code_file
    assert isinstance(tool, BaseTool)
    assert tool.name == "read_code_file"
    assert tool.description == "读取CodePilot workspace中的代码文件。当用户要求分析某个文件但没有直接提供文件内容时，应调用此工具。"
    assert tool.args["path"]["type"] == "string"
    assert tool.args_schema.model_json_schema()["required"] == ["path"]
