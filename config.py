"""加载模型配置与 workspace 根目录；不发送模型请求。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


@dataclass(frozen=True)
class Settings:
    """API Key 不参与配置对象的打印，避免意外泄露。"""

    app_name: str = "CodePilot"
    app_subtitle: str = "基于 LangChain + DeepSeek 的智能代码审查 Agent"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: str = field(default="", repr=False)


def load_settings() -> Settings:
    """读取项目根目录的 .env，已有系统环境变量优先。"""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-chat",
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
    )
