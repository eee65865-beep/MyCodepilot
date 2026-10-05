"""课程 Demo 代码运行工具，不是生产级安全沙箱。"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from threading import Thread, Event
from pathlib import Path

from langchain_core.tools import tool
from services.diagnostics import logger, log_failure
from models.schemas import CodeRunInput, CodeRunResult


TIMEOUT_SECONDS = 5
MAX_OUTPUT_BYTES = 32768  # stdout/stderr 各最多 32 KiB；超限终止直接子进程。


def _result(
    success: bool = False,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = None,
    timed_out: bool = False,
) -> dict:
    """未启动或超时的进程没有正常退出码，使用 None。"""
    if success:
        logger.info("Code Runner process completed exit_code=%s", exit_code)
    else:
        logger.warning("Code Runner process failed exit_code=%s timed_out=%s", exit_code, timed_out)
    return CodeRunResult(success=success, stdout=stdout, stderr=stderr,
                         exit_code=exit_code, timed_out=timed_out).model_dump()


def _run_process(command: list[str], directory: Path, stdin: str = "") -> dict:
    """并发读取两路输出，避免管道阻塞；超时或超量都停止进程。"""
    output = {"stdout": bytearray(), "stderr": bytearray()}
    exceeded = Event()
    # stdin 使用临时文件，不会因进程不读取 stdin 而阻塞主线程。
    with tempfile.TemporaryFile() as input_file:
        input_file.write(stdin.encode("utf-8"))
        input_file.seek(0)
        with subprocess.Popen(command, cwd=directory, stdin=input_file,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False) as process:
            def collect(pipe, name):
                try:
                    while chunk := pipe.read1(4096):
                        available = MAX_OUTPUT_BYTES - len(output[name])
                        output[name].extend(chunk[:available])
                        if len(chunk) > available:
                            exceeded.set()
                            try:
                                process.kill()
                            except OSError:
                                pass  # 进程可能已经退出。
                            break
                finally:
                    pipe.close()

            readers = [Thread(target=collect, args=(process.stdout, "stdout")),
                       Thread(target=collect, args=(process.stderr, "stderr"))]
            for reader in readers:
                reader.start()
            timed_out = False
            try:
                process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            finally:
                for reader in readers:
                    reader.join()

            stdout = bytes(output["stdout"]).decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            stderr = bytes(output["stderr"]).decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            if timed_out:
                stderr += f"\nProcess timed out after {TIMEOUT_SECONDS} seconds."
            if exceeded.is_set():
                stderr += f"\nOutput limit exceeded ({MAX_OUTPUT_BYTES} bytes per stream); execution stopped."
            return _result(success=process.returncode == 0 and not timed_out and not exceeded.is_set(),
                           stdout=stdout, stderr=stderr,
                           exit_code=None if timed_out else process.returncode, timed_out=timed_out)


@tool(args_schema=CodeRunInput)
def run_code(language: str, code: str, stdin: str = "") -> str:
    """用于运行Python或C++代码并获得真实执行结果。当用户要求运行代码、验证输出、测试程序或定位Runtime Error时使用。"""
    language = language.strip().lower()
    if language not in {"python", "cpp"}:
        return json.dumps(_result(stderr="Unsupported language: use python or cpp."))
    if not code.strip():
        return json.dumps(_result(stderr="Code must not be empty."))

    try:
        compiler = shutil.which("g++") if language == "cpp" else None
        if language == "cpp" and compiler is None:
            return json.dumps(_result(stderr="C++ compiler not available"))

        # return 或异常退出 with 块时，都由 TemporaryDirectory 清理文件。
        with tempfile.TemporaryDirectory(prefix="codepilot-") as temporary_directory:
            directory = Path(temporary_directory)
            if language == "python":
                source = directory / "main.py"
                source.write_text(code, encoding="utf-8")
                result = _run_process(
                    [sys.executable, "-X", "utf8", str(source)], directory, stdin
                )
            else:
                source = directory / "main.cpp"
                executable = directory / ("main.exe" if os.name == "nt" else "main")
                source.write_text(code, encoding="utf-8")
                compilation = _run_process(
                    [compiler, "-std=c++17", str(source), "-o", str(executable)],
                    directory,
                )
                if not compilation["success"]:
                    # 编译失败或超时，不继续运行可执行文件。
                    result = compilation
                else:
                    result = _run_process([str(executable)], directory, stdin)

        return json.dumps(result, ensure_ascii=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log_failure("code_runner", exc)
        return json.dumps(
            _result(stderr=f"Code execution failed: {type(exc).__name__}."),
            ensure_ascii=False,
        )
    except Exception as exc:
        log_failure("code_runner", exc)
        # 写文件、临时目录清理等意外错误也不能中断主程序。
        return json.dumps(_result(stderr="Unexpected code execution error."))


run_code.handle_validation_error = lambda error: json.dumps(
    _result(stderr="Invalid arguments: language, code and stdin must be strings.")
)
