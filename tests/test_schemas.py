"""业务边界使用真实 Pydantic 模型，不以文件存在作为实现证据。"""

import json

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from models.schemas import AgentResult, AgentSettings, CodeRunInput, FileReadInput
from tools.code_runner import run_code
from tools.file_reader import read_code_file


def test_models_are_used_by_tools_and_agent():
    assert read_code_file.args_schema is FileReadInput
    assert run_code.args_schema is CodeRunInput
    result = AgentResult(success=True, messages=[HumanMessage(content="hello")])
    assert isinstance(result, BaseModel)
    assert AgentResult.model_validate_json(result.model_dump_json()) == result
    assert json.loads(run_code.invoke({"language": "python", "code": "print(3)"}))["stdout"] == "3\n"


@pytest.mark.parametrize("kwargs", [{"recursion_limit": 0}, {"recursion_limit": 51}, {"max_turns": 0}, {"max_turns": 21}])
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        AgentSettings(**kwargs)


def test_result_rejects_invalid_tool_records():
    with pytest.raises(ValidationError):
        AgentResult(success=True, tool_calls=[{"name": "run_code", "success": "wrong"}])


def test_serialization_preserves_tool_call_pair():
    messages = [HumanMessage(content="run"), AIMessage(content="", tool_calls=[
        {"name": "run_code", "args": {"language": "python", "code": "print(1)"}, "id": "run-1", "type": "tool_call"}
    ]), ToolMessage(content='{"stdout":"1"}', tool_call_id="run-1", name="run_code")]
    result = AgentResult(success=True, messages=messages)
    restored = AgentResult.model_validate_json(result.model_dump_json())
    assert restored.messages == messages
    assert restored.messages[1].tool_calls[0]["id"] == restored.messages[2].tool_call_id
