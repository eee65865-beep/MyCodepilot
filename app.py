"""CodePilot 会话页面：Agent 调用与短期 Memory。"""

import json

import streamlit as st

from config import load_settings
from agent.code_agent import AgentResult, CodePilotAgent
from agent.memory import ConversationMemory, MAX_INPUT_CHARS, MAX_TURNS, MAX_HISTORY_CHARS
from services.uploads import (
    UPLOAD_EXTENSIONS, MAX_UPLOAD_BYTES, MAX_UPLOAD_CONTEXT_CHARS,
    decode_upload, build_upload_context,
)
from services.diagnostics import configure_logging, log_failure
from models.schemas import AgentSettings
def reset_session() -> None:
    """清空页面消息，同时替换当前会话的 Memory 和 Agent。"""
    st.session_state.messages = []
    options = AgentSettings(
        recursion_limit=st.session_state.get("agent_steps", 20),
        max_turns=st.session_state.get("history_turns", MAX_TURNS),
    )
    st.session_state.memory = ConversationMemory(max_turns=options.max_turns)
    st.session_state.agent = CodePilotAgent(recursion_limit=options.recursion_limit)
    # 更换控件 key，避免清空后重新把旧上传文件带入对话。
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            if message["role"] == "assistant":
                st.markdown("**Agent Answer**")
            st.markdown(message["content"])
        if message.get("tools"):
            with st.expander("工具调用记录"):
                for record in message["tools"]:
                    status = "成功" if record["success"] else "失败"
                    st.write(f"{record['name']}：{status}")
                    st.code(record["input"], language="json")
                    st.code(record["output_summary"], language="json")


def main() -> None:
    configure_logging()
    settings = load_settings()
    st.set_page_config(page_title=settings.app_name, page_icon="💻", layout="wide")
    if "messages" not in st.session_state:
        reset_session()
    if "upload_generation" not in st.session_state:
        st.session_state.upload_generation = 0

    upload = None
    upload_error = False

    with st.sidebar:
        st.title(settings.app_name)
        st.caption("技术栈：LangChain · DeepSeek · Streamlit")
        st.write(f"当前模型：{settings.deepseek_model}")
        st.caption("密钥已配置" if settings.deepseek_api_key else "尚未配置 API Key")
        st.subheader("功能")
        st.markdown("✓ 代码解释  \n✓ 代码审查  \n✓ 文件读取  \n✓ 代码执行  \n✓ 测试生成")
        st.button("清空对话", on_click=reset_session)
        with st.expander("Agent 设置"):
            st.number_input("最大执行步骤", min_value=4, max_value=50, value=20, step=1, key="agent_steps")
            st.number_input("保留历史轮数", min_value=1, max_value=20, value=MAX_TURNS, step=1, key="history_turns")
            st.caption("设置立即用于下一次请求；修改设置不会清空对话。")
        options = AgentSettings(recursion_limit=st.session_state.agent_steps, max_turns=st.session_state.history_turns)
        st.session_state.agent.recursion_limit = options.recursion_limit
        st.session_state.memory.max_turns = options.max_turns
        st.caption(f"短期历史最多保留 {options.max_turns} 轮；较早的内容会自动移除。")
        uploaded_file = st.file_uploader(
            "上传代码文件", type=sorted(UPLOAD_EXTENSIONS),
            key=f"code_upload_{st.session_state.upload_generation}",
            help="最多 32 KB，UTF-8 编码。选择文件不会自动分析或执行。",
        )
        st.caption(f"上传大小限制：{MAX_UPLOAD_BYTES // 1024} KB；文件仅在当前会话中使用。")
        if uploaded_file is not None:
            st.text(f"已选择：{uploaded_file.name}")
            st.text(f"文件大小：{uploaded_file.size / 1024:.2f} KB")
            try:
                if uploaded_file.size > MAX_UPLOAD_BYTES:
                    raise ValueError("文件过大：上传代码文件最多 32 KB（32768 字节）。")
                upload = decode_upload(uploaded_file.name, uploaded_file.getvalue())
            except ValueError as exc:
                st.error(str(exc))
                upload_error = True
            except Exception as exc:
                log_failure("upload_ui", exc)
                st.error("无法读取上传文件，请重新选择文件。")
                upload_error = True

    st.title(settings.app_name)
    st.markdown(settings.app_subtitle)
    st.write("通过自然语言分析代码，Agent 可自主读取 workspace 文件、运行代码，并结合当前会话继续回答。")
    if not settings.deepseek_api_key:
        st.info("请在项目根目录 .env 中配置 DEEPSEEK_API_KEY 后开始分析。")

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("请输入代码或分析需求", max_chars=MAX_INPUT_CHARS)
    if prompt:
        if upload_error:
            st.error("上传文件无效，请移除或重新上传后再发送请求。")
            return
        try:
            agent_input = build_upload_context(prompt, upload) if upload else prompt
            conversation = st.session_state.memory.prepare(
                agent_input,
                max_input_chars=MAX_UPLOAD_CONTEXT_CHARS if upload else MAX_INPUT_CHARS,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        user_message = {
            "role": "user",
            "content": f"{prompt}\n\n附件：{upload.name}" if upload else prompt,
        }
        st.session_state.messages.append(user_message)
        render_message(user_message)
        with st.status("Agent 正在分析任务…", expanded=True) as execution_status:
            def display_event(event: dict) -> None:
                # 只接收封装层的公开工具事件，不接触模型消息或内部 Token。
                if event["type"] == "tool_start":
                    st.write(f"🔧 正在调用工具：{event['name']}")
                    with st.expander("工具输入"):
                        st.code(json.dumps(event["input"], ensure_ascii=False)[:2000], language="json")
                elif event["type"] == "tool_end":
                    if event["success"]:
                        st.write(f"✅ {event['name']} 执行成功")
                    else:
                        st.error(f"❌ {event['name']} 执行失败")
                    st.code(event["output_summary"], language="json")
                    st.write("Agent 正在继续分析并生成回答…")

            try:
                with st.spinner("正在等待 Agent 执行结果…"):
                    result = st.session_state.agent.invoke(conversation, on_event=display_event)
            except Exception as exc:
                log_failure("agent_ui", exc)
                result = AgentResult(success=False, error="请求处理失败，请稍后重试或检查服务配置。")
            execution_status.update(
                label="Agent 执行完成" if result.success else "Agent 执行失败",
                state="complete" if result.success else "error",
                expanded=not result.success,
            )
        retained = st.session_state.memory.complete(result)

        assistant_message = {
            "role": "assistant",
            "content": result.answer if result.success else result.error,
            "error": not result.success,
            "tools": [{
                "name": record.name,
                "input": json.dumps(record.input, ensure_ascii=False)[:2000],
                "success": record.success,
                "output_summary": record.output_summary[:1000],
            } for record in result.tool_calls],
        }
        st.session_state.messages.append(assistant_message)
        # 页面展示也限长；模型历史另由 ConversationMemory 管理。
        visible = st.session_state.messages[-options.max_turns * 2:]
        while len(visible) > 2 and len(json.dumps(visible, ensure_ascii=False)) > MAX_HISTORY_CHARS:
            visible = visible[2:]
        if len(json.dumps(visible, ensure_ascii=False)) > MAX_HISTORY_CHARS:
            visible = []
        st.session_state.messages = visible
        render_message(assistant_message)
        if not retained:
            st.warning("本轮内容超过历史预算，未保留上下文。后续问题请重新提供代码或文件路径。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_failure("web_ui", exc)
        st.error("页面处理失败，请清空对话或检查配置后重试。")
