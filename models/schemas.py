"""页面、Agent 与工具共享的数据模型。"""

from typing import Any

from langchain_core.messages import BaseMessage, convert_to_messages, messages_from_dict
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator


class ProjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentSettings(ProjectModel):
    recursion_limit: int = Field(default=20, ge=4, le=50)
    max_turns: int = Field(default=8, ge=1, le=20)


class FileReadInput(ProjectModel):
    path: str


class CodeRunInput(ProjectModel):
    language: str
    code: str
    stdin: str = ""


class FileReadResult(ProjectModel):
    success: bool
    path: str
    content: str | None = None
    error: str | None = None


class CodeRunResult(ProjectModel):
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False


class ToolRecord(ProjectModel):
    name: str
    input: dict[str, Any]
    success: bool
    output_summary: str


class AgentResult(ProjectModel):
    success: bool
    answer: str = ""
    tool_calls: list[ToolRecord] = Field(default_factory=list)
    messages: list[SerializeAsAny[BaseMessage]] = Field(default_factory=list, repr=False)
    error: str | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def restore_message_types(cls, messages):
        restored = []
        for message in messages:
            if isinstance(message, dict) and message.get("type") in {"human", "ai", "tool", "system", "function", "chat"}:
                # 使用序列化还原 API，避免把 AIMessage 专有字段移入 additional_kwargs。
                restored.extend(messages_from_dict([{"type": message["type"], "data": message}]))
            else:
                restored.extend(convert_to_messages([message]))
        return restored
