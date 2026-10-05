"""Prompt 回归案例：离线检查接入，真实模型测试需要显式开启。"""

import re

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
import json

from agent import code_agent
from agent.prompts import SYSTEM_PROMPT
from tests.test_agent import ScriptedModel


EXPLANATION_HEADINGS = ["功能概述", "核心变量", "执行流程", "关键代码", "时间复杂度", "空间复杂度", "总结"]
REVIEW_HEADINGS = ["代码功能", "发现的问题", "边界情况", "复杂度", "建议修改方案"]


def test_prompt_is_sent_to_model(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="test response")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    assert code_agent.CodePilotAgent().invoke("解释代码").success
    assert isinstance(model.seen_messages[0][0], SystemMessage)
    assert model.seen_messages[0][0].content == SYSTEM_PROMPT


# 默认测试不联网、不消耗 API 额度。脚本模型不能验证回答结构，所以这里使用真实模型。
@pytest.mark.integration
@pytest.mark.parametrize("case,prompt,headings", [
    ("explain", "请完整解释以下Python代码：\ndef total(numbers):\n    result = 0\n    for number in numbers:\n        result += number\n    return result", EXPLANATION_HEADINGS),
    ("clean-review", "请完整审查下面的Python代码。要求：对两个固定大小整数求和并返回，不涉及输入输出。\ndef add(a, b):\n    return a + b", REVIEW_HEADINGS),
    ("wa-risk", "下面的C++竞赛代码为什么WA？请完整审查。我暂未提供题面和数据范围。\nint square(int n) { return n * n; }", REVIEW_HEADINGS),
    ("runtime-error", "请运行以下Python代码，再完整审查并解释实际错误：\nprint(1 / 0)", REVIEW_HEADINGS),
], ids=["explanation", "no-definite-bug", "conditional-wa-risk", "real-zero-division"])
def test_live_prompt_output(case, prompt, headings):
    result = code_agent.CodePilotAgent().invoke(prompt)
    assert result.success, result.error
    answer = result.answer
    positions = []
    for heading in headings:
        match = re.search(rf"^##\s+{heading}\s*$", answer, re.MULTILINE)
        assert match, f"缺少标题 {heading}：\n{answer}"
        positions.append(match.start())
    assert positions == sorted(positions)

    if case == "clean-review":
        assert "当前代码中没有发现明显的确定性Bug。" in answer
        assert result.tool_calls == []
    elif case == "wa-risk":
        assert "可能存在" in answer
        assert re.search(r"^###\s+\[Warning\]", answer, re.MULTILINE)
        for label in ["位置", "问题", "原因", "影响", "修改建议"]:
            assert re.search(rf"{label}(?:\*\*)?\s*[：:]", answer)
    elif case == "runtime-error":
        assert "ZeroDivisionError" in answer
        assert any(record.name == "run_code" and not record.success for record in result.tool_calls)
        assert any(isinstance(message, ToolMessage) and "ZeroDivisionError" in json.loads(message.content).get("stderr", "") for message in result.messages)
    else:
        assert result.tool_calls == []
