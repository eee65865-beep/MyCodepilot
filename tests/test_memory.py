"""验证跨轮上下文、完整工具消息、预算和清空。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.code_agent import AgentResult, CodePilotAgent
from agent.memory import ConversationMemory
from tests.test_agent import ScriptedModel, tool_call
from agent import code_agent
from tools import file_reader


def test_followup_receives_function(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="add 返回两数之和。"), AIMessage(content="固定大小数值相加为 O(1)。")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    agent = CodePilotAgent()
    memory = ConversationMemory()
    memory.complete(agent.invoke(memory.prepare("分析这个Python函数：\ndef add(a,b):\n    return a+b")))
    result = agent.invoke(memory.prepare("它的时间复杂度呢？"))
    memory.complete(result)
    assert result.success
    assert "def add" in model.seen_messages[1][1].content
    assert "add 返回" in model.seen_messages[1][2].content
    assert model.seen_messages[1][-1].content == "它的时间复杂度呢？"


def test_file_followup_and_no_duplicate_records(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("read_code_file", {"path": "examples/bug.py"}),
        AIMessage(content="索引越界。"), AIMessage(content="去掉 range 中的 +1。"),
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    agent = CodePilotAgent()
    memory = ConversationMemory()
    first = agent.invoke(memory.prepare("读取 examples/bug.py 并分析。"))
    memory.complete(first)
    second = agent.invoke(memory.prepare("这个问题怎么修改？"))
    assert second.success
    assert second.tool_calls == []
    assert any(isinstance(m, ToolMessage) and "numbers" in m.content for m in model.seen_messages[-1])


def test_whole_turn_trimming():
    memory = ConversationMemory(max_turns=2)
    old = [HumanMessage(content="old"), tool_call("run_code", {"language": "python", "code": "print(1)"}),
           ToolMessage(content='{"success":true}', tool_call_id="call-1"), AIMessage(content="done")]
    memory.messages = old + [HumanMessage(content="recent"), AIMessage(content="reply")]
    prepared = memory.prepare("new")
    assert [m.content for m in prepared] == ["recent", "reply", "new"]


def test_budget_and_clear():
    memory = ConversationMemory(max_chars=1000)
    memory.prepare("hello")
    assert not memory.complete(AgentResult(success=True, messages=[HumanMessage(content="hello"), AIMessage(content="x" * 2000)]))
    assert memory.messages == []
    memory.prepare("new")
    memory.clear()
    assert memory.messages == []


def test_failed_request_does_not_store_incomplete_calls():
    memory = ConversationMemory()
    prepared = memory.prepare("run")
    memory.complete(AgentResult(success=False, error="network error", messages=prepared + [tool_call("run_code", {})]))
    assert len(memory.messages) == 2
    assert isinstance(memory.messages[-1], AIMessage)
    assert not memory.messages[-1].tool_calls


def test_clear_prevents_old_context_in_next_request(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="old reply"), AIMessage(content="new reply")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    agent = CodePilotAgent()
    memory = ConversationMemory()
    memory.complete(agent.invoke(memory.prepare("old-file.py old question")))
    memory.clear()
    result = agent.invoke(memory.prepare("new question"))
    assert result.success
    assert len(model.seen_messages[-1]) == 2  # SystemMessage + 新的 HumanMessage。
    assert model.seen_messages[-1][-1].content == "new question"
    assert all("old-file.py" not in str(message.content) for message in model.seen_messages[-1])


def test_large_tool_result_is_rejected_before_model_and_history_kept(tmp_path, monkeypatch):
    (tmp_path / "large.py").write_text("#" + "x" * 65000, encoding="utf-8")
    monkeypatch.setattr(file_reader, "WORKSPACE_ROOT", tmp_path)
    model = ScriptedModel(responses=[tool_call("read_code_file", {"path": "large.py"}), AIMessage(content="请缩小文件。")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    memory = ConversationMemory()
    result = CodePilotAgent().invoke(memory.prepare("读取 large.py"))
    assert result.success
    assert memory.complete(result)
    assert not result.tool_calls[0].success
    assert "预算" in model.seen_messages[1][-1].content
    assert all(sum(len(m.model_dump_json()) for m in call) <= memory.max_chars for call in model.seen_messages)
    assert any(isinstance(m, ToolMessage) for m in memory.messages)


def test_model_context_limit_stops_before_request(monkeypatch):
    model = ScriptedModel(responses=[AIMessage(content="should not be requested")])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = CodePilotAgent().invoke("x" * 60000)
    assert not result.success
    assert "上下文超过预算" in result.error
    assert model.seen_messages == []


def test_accumulated_observations_do_not_bypass_context_limit(monkeypatch):
    model = ScriptedModel(responses=[
        tool_call("run_code", {"language": "python", "code": "print('x' * 10000)"}, f"call-{i}")
        for i in range(8)
    ])
    monkeypatch.setattr(code_agent, "create_deepseek_model", lambda: model)
    result = CodePilotAgent(recursion_limit=30).invoke("连续运行测试")
    assert not result.success
    assert "上下文超过预算" in result.error
    assert len(result.tool_calls) > 1
    assert all(record.success for record in result.tool_calls)
    assert all(sum(len(m.model_dump_json()) for m in call) <= 60000 for call in model.seen_messages)
