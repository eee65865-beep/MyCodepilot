"""DeepSeek 模型工厂、安全调用入口和独立连接测试。"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from config import load_settings
from services.diagnostics import log_failure, logger


class DeepSeekConfigurationError(ValueError):
    """模型配置缺失；调用方可以捕获并展示错误。"""


def describe_model_error(error: Exception) -> str:
    """模型直调和 Agent 共用错误映射，不展示服务端原始异常正文。"""
    if isinstance(error, DeepSeekConfigurationError):
        return "未配置DEEPSEEK_API_KEY，请在.env文件中设置。"
    if isinstance(error, AuthenticationError):
        return "DeepSeek 身份验证失败，请检查 API Key 是否有效。"
    if isinstance(error, APITimeoutError):
        return "DeepSeek 请求超时，请稍后重试或检查网络。"
    if isinstance(error, APIConnectionError):
        return "无法连接 DeepSeek，请检查网络、代理或防火墙设置。"
    if isinstance(error, RateLimitError):
        return "DeepSeek 请求受到限流，请稍后重试。"
    if isinstance(error, APIStatusError):
        if error.status_code == 402:
            return "DeepSeek 账户余额不足，请检查账户余额。"
        return f"DeepSeek API 调用失败（HTTP {error.status_code}），请检查模型名称、请求参数或服务状态。"
    return "DeepSeek 调用失败，请检查配置、依赖版本或稍后重试。"


@dataclass
class ModelCallResult:
    """调用结果：成功返回消息，失败返回适合页面展示的错误。"""

    success: bool
    message: AIMessage | None = None
    error: str | None = None


def create_deepseek_model() -> ChatDeepSeek:
    """读取 .env 配置并创建模型；创建对象本身不会发送请求。"""
    # load_settings 内部使用 python-dotenv，且不依赖当前工作目录。
    settings = load_settings()
    if not settings.deepseek_api_key:
        raise DeepSeekConfigurationError(
            "未配置DEEPSEEK_API_KEY，请在.env文件中设置。"
        )

    logger.info("Creating DeepSeek model; timeout=30 max_retries=2")
    return ChatDeepSeek(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        temperature=0,
        timeout=30,
        max_retries=2,
    )


def invoke_deepseek(prompt: str) -> ModelCallResult:
    """独立模型验证入口；Agent 使用工厂并共享同一错误分类。"""
    if not isinstance(prompt, str) or not prompt.strip():
        return ModelCallResult(success=False, error="请输入非空的模型请求。")

    try:
        model = create_deepseek_model()
        message = model.invoke(prompt)
        return ModelCallResult(success=True, message=message)
    except Exception as exc:
        log_failure("deepseek", exc)
        return ModelCallResult(success=False, error=describe_model_error(exc))


def main() -> int:
    """运行 python -m llm.deepseek 验证最小模型调用链路。"""
    result = invoke_deepseek("请用一句话解释Python中的递归")
    if not result.success:
        print(result.error)
        return 1

    assert result.message is not None
    print(result.message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
