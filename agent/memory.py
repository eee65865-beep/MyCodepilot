"""短期会话历史：保留完整轮次，避免拆开工具调用与工具结果。"""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


MAX_TURNS = 8
MAX_HISTORY_CHARS = 60000
MAX_INPUT_CHARS = 12000


def _trim(messages: list[BaseMessage], max_turns: int, max_chars: int) -> list[BaseMessage]:
    """从最旧的一整轮开始删除；字符预算是粗略控制，不是 Token 计数。"""
    if max_turns <= 0:
        return []
    turns = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([])
        if turns:
            turns[-1].append(message)

    turns = turns[-max_turns:]
    while turns and sum(len(message.model_dump_json()) for turn in turns for message in turn) > max_chars:
        turns.pop(0)
    return [message for turn in turns for message in turn]


@dataclass
class ConversationMemory:
    """由 Streamlit session_state 持有；不使用数据库或全局共享缓存。"""

    messages: list[BaseMessage] = field(default_factory=list)
    max_turns: int = MAX_TURNS
    max_chars: int = MAX_HISTORY_CHARS

    def prepare(self, prompt: str, max_input_chars: int = MAX_INPUT_CHARS) -> list[BaseMessage]:
        """加入新请求，并为当前输入预留上下文预算。"""
        if not prompt.strip():
            raise ValueError("请输入非空消息。")
        if len(prompt) > max_input_chars:
            raise ValueError(f"本次请求超过 {max_input_chars} 个字符，请缩小代码或输入长度。")
        current = HumanMessage(content=prompt)
        if len(current.model_dump_json()) > self.max_chars:
            raise ValueError("文件和请求超过当前上下文预算，请缩小代码或输入长度。")
        self.messages = _trim(
            self.messages, self.max_turns - 1,
            max(0, self.max_chars - len(current.model_dump_json())),
        ) + [current]
        return list(self.messages)

    def complete(self, result) -> bool:
        """保存完整结果；请求失败时不保留可能未配对的部分工具调用。"""
        if result.success:
            completed = result.messages
        else:
            completed = self.messages + [AIMessage(content=result.error or "请求失败。")]
        self.messages = _trim(completed, self.max_turns, self.max_chars)
        # False 表示最新一轮也超出预算，已清空，页面应告知用户。
        return bool(self.messages)

    def clear(self) -> None:
        self.messages.clear()
