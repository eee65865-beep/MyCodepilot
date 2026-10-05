"""测试生成案例：真实工具观察与可选真实 DeepSeek 演示。"""

import json
import ast

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent import code_agent
from config import WORKSPACE_ROOT
from tools.code_runner import run_code
from tests.test_agent import ScriptedModel, tool_call
from agent.memory import ConversationMemory


SOURCE = (WORKSPACE_ROOT / "examples" / "sum_bug.py").read_text(encoding="utf-8")
CASES = [
    ("normal", "3\n1 2 3\n", "6", "5"),
    ("minimum", "0\n", "0", "0"),
    ("maximum", "10\n" + "100 " * 10 + "\n", "1000", "900"),
    ("special", "3\n-2 0 2\n", "0", "2"),
    ("bug-trigger", "1\n7\n", "7", "0"),
]


@pytest.mark.parametrize("category,stdin,expected,actual", CASES, ids=[c[0] for c in CASES])
def test_demo_original_program(category, stdin, expected, actual):
    result = json.loads(run_code.invoke({"language": "python", "code": SOURCE, "stdin": stdin}))
    assert result["success"]
    assert result["stdout"].strip() == actual
    assert (result["stdout"].strip() == expected) == (category == "minimum")


def test_read_then_execute_observations(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("read_code_file", {"path": "examples/sum_bug.py"}, "read-sum"),
        tool_call("run_code", {"language": "python", "code": SOURCE, "stdin": "3\n1 2 3\n"}, "run-sum"),
        AIMessage(content="Expected: 6\nActual: 5\nResult: FAILED"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = code_agent.CodePilotAgent().invoke("读取 examples/sum_bug.py，生成测试并运行。")
    assert result.success
    assert [record.name for record in result.tool_calls] == ["read_code_file", "run_code"]
    # success=True 是进程成功，不代表输出符合求和契约。
    observation = model.seen_messages[2][-1]
    assert isinstance(observation, ToolMessage)
    assert json.loads(observation.content)["stdout"] == "5\n"
    assert json.loads(observation.content)["success"]


@pytest.mark.integration
@pytest.mark.parametrize("execute", [False, True], ids=["generate-only", "generate-and-verify"])
def test_live_test_generation(execute):
    if execute:
        # 使用完整路径，文件实际位于 workspace/examples 下。
        prompt = "读取 sum_bug.py，生成测试并运行，找出可能的问题。"
    else:
        prompt = "给这段代码生成测试用例，不要运行：\n" + SOURCE
    result = code_agent.CodePilotAgent().invoke(prompt)
    assert result.success, result.error
    for label in ["测试目的", "Input", "Expected Output", "说明"]:
        assert label in result.answer, result.answer
    for category in ["正常", "最小", "最大", "特殊", "Bug"]:
        assert category.lower() in result.answer.lower(), result.answer
    if not execute:
        assert result.tool_calls == []
        return

    names = [record.name for record in result.tool_calls]
    assert "read_code_file" in names and "run_code" in names
    assert names.index("read_code_file") < names.index("run_code")
    assert "Actual" in result.answer and "FAILED" in result.answer
    observations = [json.loads(m.content) for m in result.messages if isinstance(m, ToolMessage)]
    assert any("stdout" in observation and observation["stdout"] for observation in observations)
    assert any(term in result.answer for term in ["第一个", "首个", "首元素", "索引 0", "索引0"])


@pytest.mark.integration
def test_live_report_matches_each_original_program_observation():
    inputs = [case[1] for case in CASES]
    prompt = (
        "读取 examples/sum_bug.py，为以下五个输入生成测试并逐例运行验证。"
        "每个输入独立调用一次 run_code，必须运行读取的完整原代码，不修改。"
        "最终只输出JSON对象：{\"tests\":[{\"input\":\"完整stdin\",\"expected\":\"正确契约输出\","
        "\"actual\":\"工具stdout\",\"result\":\"PASSED或FAILED\"}]}。"
        "数字输出用字符串，不要添加解释或Markdown。输入列表：" + json.dumps(inputs)
    )
    result = code_agent.CodePilotAgent(recursion_limit=30).invoke(prompt)
    assert result.success, result.error
    answer = result.answer.strip()
    if answer.startswith("```"):
        answer = "\n".join(answer.splitlines()[1:-1])
    report = json.loads(answer)["tests"]
    observed = {}
    for record, observation in zip(result.tool_calls, [m for m in result.messages if isinstance(m, ToolMessage)], strict=True):
        if record.name != "run_code":
            continue
        assert ast.dump(ast.parse(record.input["code"])) == ast.dump(ast.parse(SOURCE))
        payload = json.loads(observation.content)
        assert payload["success"]
        observed[tuple(record.input.get("stdin", "").split())] = payload["stdout"].strip()
    assert len(report) == len(CASES) == len(observed)
    rows = {tuple(row["input"].split()): row for row in report}
    assert len(rows) == len(CASES)
    for category, stdin, expected, actual in CASES:
        key = tuple(stdin.split())
        row = rows[key]
        assert row["expected"].strip() == expected
        assert row["actual"].strip() == observed[key] == actual
        assert row["result"] == ("PASSED" if expected == actual else "FAILED")


@pytest.mark.integration
def test_live_memory_followup_and_clear():
    agent = code_agent.CodePilotAgent()
    memory = ConversationMemory()
    first = agent.invoke(memory.prepare("分析这个Python函数：def add(a,b): return a+b。使用固定大小数值假设。"))
    assert first.success, first.error
    assert memory.complete(first)
    second = agent.invoke(memory.prepare("它的时间复杂度呢？请说明函数名。"))
    assert second.success, second.error
    assert "add" in second.answer and "O(1)" in second.answer
    memory.complete(second)
    memory.clear()
    fresh = agent.invoke(memory.prepare("之前我们讨论的是哪个函数？若当前对话没有提供函数请说明无法确定。"))
    assert fresh.success, fresh.error
    assert "add" not in fresh.answer
