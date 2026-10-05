"""LangChain 文件读取工具：只读取 workspace 中的代码文件。"""

import json
import stat

from langchain_core.tools import tool

from config import WORKSPACE_ROOT
from services.diagnostics import logger, log_failure
from models.schemas import FileReadInput, FileReadResult
from services.workspace import resolve_workspace_path


MAX_FILE_SIZE = 1024 * 1024  # 1 MiB
ALLOWED_EXTENSIONS = {".py", ".cpp", ".c", ".h", ".hpp", ".java", ".js", ".ts"}


def _result(path: str, content: str | None = None, error: str | None = None) -> str:
    """统一成功和失败的 JSON 格式，保留中文。"""
    if error is not None:
        logger.warning("File Reader request failed")
    else:
        logger.info("File Reader completed")
    return FileReadResult(success=error is None, path=path, content=content, error=error).model_dump_json()


@tool(args_schema=FileReadInput)
def read_code_file(path: str) -> str:
    """读取CodePilot workspace中的代码文件。当用户要求分析某个文件但没有直接提供文件内容时，应调用此工具。"""
    try:
        root = WORKSPACE_ROOT.resolve()
        resolved_path = resolve_workspace_path(path, root)
        if resolved_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return _result(path, error="非法扩展名：只支持 .py、.cpp、.c、.h、.hpp、.java、.js、.ts。")

        file_stat = resolved_path.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            return _result(path, error="指定路径不是普通文件。")
        if file_stat.st_size > MAX_FILE_SIZE:
            return _result(path, error="文件过大：最大允许 1 MiB（1048576 字节）。")

        # 限量读取，避免文件在大小检查后增长而造成无上限读取。
        with resolved_path.open("rb") as file:
            data = file.read(MAX_FILE_SIZE + 1)
        if len(data) > MAX_FILE_SIZE:
            return _result(path, error="文件过大：最大允许 1 MiB（1048576 字节）。")

        content = data.decode("utf-8-sig")  # 兼容 UTF-8 BOM，不忽略编码错误。
        if not content:
            return _result(path, error="文件为空，无法分析。")

        # 成功结果使用 workspace 相对路径，便于模型理解和再次调用。
        return _result(resolved_path.relative_to(root).as_posix(), content=content)
    except FileNotFoundError:
        return _result(path, error="文件不存在，请检查 workspace 中的文件路径。")
    except PermissionError:
        return _result(path, error="没有权限读取该文件。")
    except UnicodeDecodeError:
        return _result(path, error="文件编码错误，请将代码文件保存为 UTF-8。")
    except ValueError as exc:
        # 仅展示我们自己定义的路径错误，不泄露系统异常正文。
        safe_errors = {"文件路径不能为空。", "禁止使用 .. 进行路径穿越。", "非法文件路径。",
                       "路径越界：只能读取 CodePilot workspace 中的文件。"}
        return _result(path, error=str(exc) if str(exc) in safe_errors else "无法读取文件，请检查路径或文件状态。")
    except (OSError, RuntimeError):
        return _result(path, error="无法读取文件，请检查路径或文件状态。")
    except Exception as exc:
        log_failure("file_reader", exc)
        return _result(path, error="文件读取发生异常，请稍后重试。")


# 参数校验发生在函数执行前，也返回 JSON，避免错误工具参数中断调用链。
read_code_file.handle_validation_error = lambda error: _result(
    "", error="参数错误：请提供字符串类型的 path。"
)
