"""workspace 路径边界；File Reader 复用此入口。"""

from pathlib import Path


def resolve_workspace_path(path: str, root: Path) -> Path:
    if not path.strip():
        raise ValueError("文件路径不能为空。")
    requested = Path(path.replace("\\", "/"))
    if ".." in requested.parts:
        raise ValueError("禁止使用 .. 进行路径穿越。")
    if any(":" in part for part in requested.parts if part != requested.anchor):
        raise ValueError("非法文件路径。")
    resolved = (root.resolve() / requested).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("路径越界：只能读取 CodePilot workspace 中的文件。")
    return resolved
