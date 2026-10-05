"""通过 LangChain create_agent 执行任务，并提取可观察的工具记录。"""

import json
from typing import Callable

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, convert_to_messages
from langchain.agents.middleware import wrap_tool_call
from langgraph.errors import GraphRecursionError

from agent.prompts import SYSTEM_PROMPT
from agent.limits import ContextBudgetError, check_context_budget
from llm.deepseek import create_deepseek_model, describe_model_error
from services.diagnostics import log_failure, logger
from models.schemas import AgentResult, ToolRecord, AgentSettings
from tools.code_runner import run_code
from tools.file_reader import read_code_file


MAX_TOOL_RESULT_CHARS = 16000


@wrap_tool_call
def handle_tool_errors(request, handler):
    """捕获工具意外异常，让错误 Observation 继续返回 Agent。"""
    try:
        result = handler(request)
        if isinstance(result, ToolMessage) and len(result.model_dump_json()) > MAX_TOOL_RESULT_CHARS:
            # 不把部分源码当作完整文件；向模型返回明确失败，保留调用配对。
            return ToolMessage(
                content=json.dumps({"success": False, "error": "工具结果超过Agent上下文预算，请缩小文件或程序输出后重试。"}, ensure_ascii=False),
                tool_call_id=request.tool_call["id"], name=request.tool_call["name"], status="error",
            )
        return result
    except Exception as exc:
        log_failure("agent_tool", exc)
        return ToolMessage(
            content=json.dumps({
                "success": False, "content": None,
                "error": "工具执行异常，请检查输入或稍后重试。",
                "stderr": "工具执行异常，请检查输入或稍后重试。",
            }, ensure_ascii=False),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"], status="error",
        )


def _text_content(message: BaseMessage) -> str:
    """只取公开文本块，不读取 additional_kwargs 中的推理内容。"""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in message.content
        if isinstance(block, str) or block.get("type") == "text"
    )


def _tool_result(message: ToolMessage) -> tuple[bool, str]:
    """两个项目工具返回 JSON；摘要仅包含可观察的执行结果。"""
    content = _text_content(message)
    try:
        result = json.loads(content)
        if not isinstance(result, dict):
            return False, content[:300]
        success = result.get("success") is True and message.status != "error"
        if "content" in result:
            summary = {
                "path": result.get("path"),
                "content_preview": (result.get("content") or "")[:200],
                "error": result.get("error"),
            }
        else:
            summary = {
                "stdout": str(result.get("stdout", ""))[:200],
                "stderr": str(result.get("stderr", ""))[:200],
                "exit_code": result.get("exit_code"),
                "timed_out": result.get("timed_out"),
                "error": result.get("error"),
            }
        return success, json.dumps(summary, ensure_ascii=False)
    except (ValueError, TypeError):
        return False, content[:300]


class CodePilotAgent:
    """Web 层调用 invoke；消息历史由会话的 ConversationMemory 管理。"""

    def __init__(self, recursion_limit: int = 20):
        self.recursion_limit = AgentSettings(recursion_limit=recursion_limit).recursion_limit
        self._agent = None

    def invoke(
        self, messages: str | list,
        on_event: Callable[[dict], None] | None = None,
    ) -> AgentResult:
        """接收请求字符串或 LangChain 消息列表，失败也返回结果。"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if not isinstance(messages, list) or not messages:
            return AgentResult(success=False, error="请提供用户消息。")
        try:
            messages = convert_to_messages(messages)
        except Exception as exc:
            log_failure("agent_messages", exc)
            return AgentResult(success=False, error="对话消息格式无效，请清空对话后重试。")

        records = []
        latest_messages = []
        pending_calls = {}
        recorded_ids = set()
        started_ids = set()
        # 传入历史时，不重复展示历史中已经完成的工具调用。
        for message in messages:
            if isinstance(message, ToolMessage):
                recorded_ids.add(message.tool_call_id)
            elif isinstance(message, dict) and message.get("role") == "tool":
                recorded_ids.add(message.get("tool_call_id"))
            if isinstance(message, AIMessage):
                started_ids.update(call["id"] for call in message.tool_calls)
            elif isinstance(message, dict):
                started_ids.update(call.get("id") for call in message.get("tool_calls", []))

        try:
            if self._agent is None:
                # 真正的模型/工具循环由 LangChain 管理，这里没有关键词路由。
                self._agent = create_agent(
                    model=create_deepseek_model(),
                    tools=[read_code_file, run_code],
                    system_prompt=SYSTEM_PROMPT,
                    middleware=[check_context_budget, handle_tool_errors],
                )

            # values 流返回 Agent 消息状态；这里只观察，不调度工具。
            for state in self._agent.stream(
                {"messages": messages},
                config={"recursion_limit": self.recursion_limit},
                stream_mode="values",
            ):
                latest_messages = state["messages"]
                for message in latest_messages:
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            pending_calls[call["id"]] = call
                            if call["id"] not in started_ids:
                                started_ids.add(call["id"])
                                logger.info("Agent tool call started")
                                if on_event:
                                    on_event({"type": "tool_start", "name": call["name"], "input": call["args"]})
                    elif isinstance(message, ToolMessage) and message.tool_call_id not in recorded_ids:
                        call = pending_calls.get(message.tool_call_id, {})
                        success, summary = _tool_result(message)
                        records.append(ToolRecord(
                            name=message.name or call.get("name", "unknown"),
                            input=call.get("args", {}),
                            success=success,
                            output_summary=summary,
                        ))
                        recorded_ids.add(message.tool_call_id)
                        logger.info("Agent tool completed success=%s", success)
                        if on_event:
                            on_event({
                                "type": "tool_end", "name": records[-1].name,
                                "success": success, "output_summary": summary,
                            })

            final_message = latest_messages[-1] if latest_messages else None
            if not isinstance(final_message, AIMessage) or final_message.tool_calls or not _text_content(final_message).strip():
                return AgentResult(
                    success=False, tool_calls=records, messages=latest_messages,
                    error="Agent 未生成最终回答。",
                )
            return AgentResult(
                success=True, answer=_text_content(final_message),
                tool_calls=records, messages=latest_messages,
            )
        except ContextBudgetError as exc:
            logger.warning("Agent context budget reached")
            error = str(exc)
        except GraphRecursionError:
            logger.warning("Agent execution step limit reached")
            error = "Agent执行步骤超过限制，请简化请求后重试。"
        except Exception as exc:
            log_failure("agent", exc)
            error = describe_model_error(exc)

        return AgentResult(
            success=False, error=error, tool_calls=records, messages=latest_messages
        )


def main() -> int:
    """两个真实 DeepSeek 案例：python -m agent.code_agent。"""
    agent = CodePilotAgent()
    requests = [
        ("读取 examples/bug.py 并分析代码问题。", "read_code_file"),
        ("运行以下Python代码并告诉我输出：\nprint(1 + 2)", "run_code"),
    ]
    for request, expected_tool in requests:
        print(f"用户：{request}")
        result = agent.invoke(request)
        for record in result.tool_calls:
            print(json.dumps({
                "tool_name": record.name,
                "tool_input": record.input,
                "success": record.success,
                "output_summary": record.output_summary,
            }, ensure_ascii=False))
        print(result.answer if result.success else result.error)
        if not result.success:
            return 1
        if not any(record.name == expected_tool and record.success for record in result.tool_calls):
            print(f"案例验证失败：未观察到 {expected_tool} 成功执行。")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
