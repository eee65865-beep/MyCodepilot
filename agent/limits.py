"""在框架每次模型请求前检查预算；超限不发送，也不静默截断代码。"""

from langchain.agents.middleware import wrap_model_call

from agent.memory import MAX_HISTORY_CHARS


class ContextBudgetError(ValueError):
    pass


@wrap_model_call
def check_context_budget(request, handler):
    messages = list(request.messages)
    if request.system_message is not None:
        messages = [request.system_message] + messages
    if sum(len(message.model_dump_json()) for message in messages) > MAX_HISTORY_CHARS:
        raise ContextBudgetError("Agent上下文超过预算，请缩小代码、输出或清空对话后重试。")
    return handler(request)
