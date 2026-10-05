"""检查共享路径入口真的被 File Reader 使用。"""

import json

import pytest

from services.workspace import resolve_workspace_path
from tools import file_reader


def test_resolve_and_reader_share_boundary(tmp_path, monkeypatch):
    (tmp_path / "hello.py").write_text("print(1)", encoding="utf-8")
    calls = []
    def tracked(path, root):
        calls.append(path)
        return resolve_workspace_path(path, root)
    monkeypatch.setattr(file_reader, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(file_reader, "resolve_workspace_path", tracked)
    assert json.loads(file_reader.read_code_file.invoke({"path": "hello.py"}))["success"]
    assert calls == ["hello.py"]
    assert resolve_workspace_path("hello.py", tmp_path) == tmp_path / "hello.py"


@pytest.mark.parametrize("path", ["../outside.py", "..\\outside.py", "nested/../hello.py", "", "x:stream.py"])
def test_service_rejects_invalid_path(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_workspace_path(path, tmp_path)
