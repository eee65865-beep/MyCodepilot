"""独立验证 Code Runner，不调用 Agent 或模型。"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from tools import code_runner


requires_cpp = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")


def invoke(language, code, stdin=""):
    return json.loads(
        code_runner.run_code.invoke({"language": language, "code": code, "stdin": stdin})
    )


def test_python_success():
    result = invoke("python", "print('hello')")
    assert result == {
        "success": True, "stdout": "hello\n", "stderr": "", "exit_code": 0, "timed_out": False
    }


def test_python_runtime_error():
    result = invoke("python", "print(1 / 0)")
    assert not result["success"]
    assert result["exit_code"] != 0
    assert "ZeroDivisionError" in result["stderr"]
    assert not result["timed_out"]


def test_python_timeout(monkeypatch):
    monkeypatch.setattr(code_runner, "TIMEOUT_SECONDS", 0.5)
    result = invoke("python", "print('started', flush=True)\nwhile True: pass")
    assert not result["success"]
    assert result["timed_out"]
    assert result["exit_code"] is None
    assert "started" in result["stdout"]


def test_python_stdin_and_unicode():
    result = invoke("python", "print(input())", "你好\n")
    assert result["success"]
    assert result["stdout"] == "你好\n"


def test_python_syntax_error():
    result = invoke("python", "def broken(:")
    assert not result["success"]
    assert "SyntaxError" in result["stderr"]


def test_python_uses_current_interpreter():
    result = invoke("python", "import sys\nprint(sys.executable)")
    assert result["success"]
    assert Path(result["stdout"].strip()) == Path(sys.executable)


@requires_cpp
def test_cpp_success():
    result = invoke(
        "cpp",
        '#include <iostream>\nint main() { int n; std::cin >> n; std::cout << n * 2 << "\\n"; }',
        "21\n",
    )
    assert result["success"], result
    assert result["stdout"] == "42\n"
    assert result["exit_code"] == 0


@requires_cpp
def test_cpp_compile_error():
    result = invoke("cpp", "int main() { broken syntax }")
    assert not result["success"]
    assert result["stderr"]
    assert result["exit_code"] != 0
    assert not result["timed_out"]


@requires_cpp
def test_cpp_runtime_error():
    result = invoke(
        "cpp",
        '#include <iostream>\n#include <vector>\n#include <exception>\n'
        'int main() { try { std::vector<int> v; v.at(0); } '
        'catch (const std::exception& e) { std::cerr << e.what(); return 1; } }',
    )
    assert not result["success"]
    assert result["exit_code"] == 1
    assert result["stderr"]
    assert not result["timed_out"]


@requires_cpp
def test_cpp_timeout():
    # 使用默认 5 秒：编译和执行分别有超时预算。
    result = invoke("cpp", "int main() { while (true) {} }")
    assert not result["success"]
    assert result["timed_out"]
    assert result["exit_code"] is None


def test_cpp_compiler_missing(monkeypatch):
    monkeypatch.setattr(code_runner.shutil, "which", lambda name: None)
    result = invoke("cpp", "int main() {}")
    assert not result["success"]
    assert result["stderr"] == "C++ compiler not available"
    assert result["exit_code"] is None


@pytest.mark.parametrize("exception", [PermissionError, OSError, subprocess.SubprocessError])
def test_subprocess_error(monkeypatch, exception):
    def failed(*args, **kwargs):
        raise exception("test error")

    monkeypatch.setattr(code_runner.subprocess, "Popen", failed)
    result = invoke("python", "print(1)")
    assert not result["success"]
    assert result["exit_code"] is None
    assert not result["timed_out"]
    assert exception.__name__ in result["stderr"]


@pytest.mark.parametrize("code", ["print(1)", "raise RuntimeError('test')", "while True: pass"])
def test_temporary_directory_cleanup(tmp_path, monkeypatch, code):
    original = code_runner.tempfile.TemporaryDirectory
    directories = []

    def tracked_directory(**kwargs):
        directory = original(dir=tmp_path, **kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(code_runner.tempfile, "TemporaryDirectory", tracked_directory)
    monkeypatch.setattr(code_runner, "TIMEOUT_SECONDS", 0.5)
    result = invoke(
        "python", "import os\nprint(os.getcwd(), flush=True)\nopen('artifact.txt', 'w').write('test')\n" + code
    )
    assert len(directories) == 1
    assert Path(result["stdout"].splitlines()[0]) == directories[0]
    assert not directories[0].exists()


def test_compile_timeout_does_not_execute(monkeypatch):
    monkeypatch.setattr(code_runner.shutil, "which", lambda name: "g++")
    calls = []

    def timed_out(command, *args, **kwargs):
        calls.append(command)
        return code_runner._result(stdout="partial", stderr="compiler", timed_out=True)

    monkeypatch.setattr(code_runner, "_run_process", timed_out)
    result = invoke("cpp", "int main() {}")
    assert result["timed_out"]
    assert result["stdout"] == "partial"
    assert "compiler" in result["stderr"]
    assert len(calls) == 1
    assert "-std=c++17" in calls[0]


@pytest.mark.parametrize("language,code", [("java", "hello"), ("python", " ")])
def test_invalid_input(language, code):
    result = invoke(language, code)
    assert not result["success"]
    assert result["stderr"]


def test_invalid_tool_arguments():
    result = json.loads(code_runner.run_code.invoke({"language": "python"}))
    assert not result["success"]
    assert "Invalid arguments" in result["stderr"]


def test_langchain_metadata():
    tool = code_runner.run_code
    assert isinstance(tool, BaseTool)
    assert tool.name == "run_code"
    assert tool.args_schema.model_json_schema()["required"] == ["language", "code"]
    assert tool.args["stdin"]["default"] == ""


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_limit_stops_noisy_program(monkeypatch, stream):
    monkeypatch.setattr(code_runner, "MAX_OUTPUT_BYTES", 1024)
    result = invoke("python", f"import sys\nwhile True: sys.{stream}.write('x' * 4096); sys.{stream}.flush()")
    assert not result["success"]
    assert not result["timed_out"]
    assert "Output limit exceeded" in result["stderr"]
    assert len(result["stdout"]) <= 1024
    assert len(result["stderr"]) < 1200
