"""项目日志只记录白名单元数据，不记录异常正文、请求、代码或密钥。"""

import logging


logger = logging.getLogger("codepilot")


def configure_logging() -> None:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def log_failure(component: str, error: Exception) -> None:
    # 不使用 logger.exception、str(error)、repr(error) 或 exc_info。
    logger.error("component=%s error_type=%s", component, type(error).__name__)
