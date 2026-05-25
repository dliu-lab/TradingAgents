"""Tests for the experimental Codex app-server LLM provider."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.codex_app_server_client import (
    CodexAppServerChatModel,
    CodexAppServerProcessClient,
)
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options


@dataclass
class FakeRunner:
    text: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run_turn(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.text


class FakeStdIn:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        pass


class FakeStdOut:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._lines = [json.dumps(message) + "\n" for message in messages]

    def fileno(self) -> int:
        return 0

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class FakeProcess:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.stdin = FakeStdIn()
        self.stdout = FakeStdOut(messages)
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


@pytest.mark.unit
def test_codex_process_client_runs_app_server_turn(monkeypatch):
    process = FakeProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr_123"}}},
            {"id": 3, "result": {"turn": {"id": "turn_123", "status": "inProgress"}}},
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "first "},
            },
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "id": "item_1",
                        "text": "first final",
                    }
                },
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"id": "turn_123", "status": "completed"}},
            },
        ]
    )
    monkeypatch.setattr("select.select", lambda read, _write, _err, _timeout: (read, [], []))
    client = CodexAppServerProcessClient(
        command="codex",
        cwd="/tmp/project",
        process_factory=lambda *_args, **_kwargs: process,
    )

    result = client.run_turn(
        prompt="Summarize NVDA.",
        model="gpt-5.4",
        output_schema={"type": "object"},
    )

    assert result == "first final"
    sent = [json.loads(line) for line in process.stdin.writes]
    assert sent[0]["method"] == "initialize"
    assert sent[1]["method"] == "initialized"
    assert sent[2]["method"] == "thread/start"
    assert sent[3]["method"] == "turn/start"
    assert sent[3]["params"]["approvalPolicy"] == "on-request"
    assert sent[3]["params"]["sandboxPolicy"] == {
        "type": "readOnly",
        "networkAccess": False,
    }
    assert sent[3]["params"]["outputSchema"] == {"type": "object"}
    assert process.terminated is True


@pytest.mark.unit
def test_codex_chat_model_invokes_runner_and_returns_ai_message():
    runner = FakeRunner("Market report")
    llm = CodexAppServerChatModel(model="gpt-5.4", runner=runner, cwd="/tmp/project")

    response = llm.invoke([HumanMessage(content="Analyze NVDA")])

    assert response.content == "Market report"
    assert runner.calls[0]["model"] == "gpt-5.4"
    assert runner.calls[0]["cwd"] == "/tmp/project"
    assert "Analyze NVDA" in runner.calls[0]["prompt"]


@pytest.mark.unit
def test_codex_bind_tools_translates_json_tool_calls_to_ai_message():
    runner = FakeRunner(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_stock_data",
                        "args": {"ticker": "NVDA", "start_date": "2026-01-01"},
                    }
                ]
            }
        )
    )
    llm = CodexAppServerChatModel(model="gpt-5.4", runner=runner)

    response = llm.bind_tools([lambda ticker: ticker]).invoke("Fetch data")

    assert response.content == ""
    assert response.tool_calls == [
        {
            "name": "get_stock_data",
            "args": {"ticker": "NVDA", "start_date": "2026-01-01"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]
    assert "Return JSON only" in runner.calls[0]["prompt"]


@pytest.mark.unit
def test_codex_with_structured_output_parses_pydantic_schema():
    class Pick(BaseModel):
        answer: str

    runner = FakeRunner('{"answer": "hold"}')
    llm = CodexAppServerChatModel(model="gpt-5.4", runner=runner)

    response = llm.with_structured_output(Pick).invoke("Choose")

    assert response == Pick(answer="hold")
    assert runner.calls[0]["output_schema"]["properties"]["answer"]["type"] == "string"


@pytest.mark.unit
def test_codex_provider_is_registered_without_api_key():
    assert get_api_key_env("codex") is None
    assert get_model_options("codex", "quick")

    client = create_llm_client("codex", "gpt-5.4")

    assert client.get_provider_name() == "codex"


@pytest.mark.unit
def test_codex_model_catalog_prefers_codex_chatgpt_models():
    quick_models = [value for _label, value in get_model_options("codex", "quick")]
    deep_models = [value for _label, value in get_model_options("codex", "deep")]

    assert quick_models[0] == "gpt-5.3-codex-spark"
    assert deep_models[0] == "gpt-5.3-codex"
    assert "gpt-5.5" not in quick_models
    assert "gpt-5.5" not in deep_models
