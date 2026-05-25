"""Codex app-server backed LLM provider.

This provider delegates authentication and model access to the official
``codex app-server`` process. It is intentionally implemented as a small
bridge from TradingAgents' LangChain-shaped expectations to Codex's JSON-RPC
app-server protocol.
"""

from __future__ import annotations

import json
import os
import re
import select
import shlex
import subprocess
import time
import uuid
from typing import Any, Callable, Iterable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field

from .base_client import BaseLLMClient
from .validators import validate_model


DEFAULT_CODEX_COMMAND = "codex"
DEFAULT_CODEX_TIMEOUT_SECONDS = 300.0
DEFAULT_CODEX_CLIENT_NAME = "tradingagents_codex_bridge"
DEFAULT_CODEX_CLIENT_TITLE = "TradingAgents Codex Bridge"
DEFAULT_CODEX_CLIENT_VERSION = "0.1.0"
DEFAULT_CODEX_APPROVAL_POLICY = "on-request"

TOOL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "final": {
            "type": "string",
            "description": "The final answer when no tool call is needed.",
        },
        "tool_calls": {
            "type": "array",
            "description": "Tool calls to execute before answering.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "args": {"type": "object", "additionalProperties": True},
                },
                "required": ["name", "args"],
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


class CodexAppServerError(RuntimeError):
    """Raised when the Codex app-server bridge cannot complete a turn."""


class CodexAppServerProcessClient:
    """Small synchronous JSONL client for ``codex app-server`` over stdio."""

    def __init__(
        self,
        *,
        command: str = DEFAULT_CODEX_COMMAND,
        cwd: Optional[str] = None,
        timeout_seconds: float = DEFAULT_CODEX_TIMEOUT_SECONDS,
        client_name: str = DEFAULT_CODEX_CLIENT_NAME,
        client_title: str = DEFAULT_CODEX_CLIENT_TITLE,
        client_version: str = DEFAULT_CODEX_CLIENT_VERSION,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.command = command
        self.cwd = cwd or os.getcwd()
        self.timeout_seconds = timeout_seconds
        self.client_name = client_name
        self.client_title = client_title
        self.client_version = client_version
        self.process_factory = process_factory

    def run_turn(
        self,
        *,
        prompt: str,
        model: str,
        cwd: Optional[str] = None,
        output_schema: Optional[dict[str, Any]] = None,
        effort: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """Run one Codex app-server turn and return the final agent text."""
        proc = self._start_process()
        timeout = timeout_seconds or self.timeout_seconds
        deadline = time.monotonic() + timeout
        try:
            self._initialize(proc, deadline)
            thread_id = self._start_thread(proc, model, deadline)
            turn_id = self._start_turn(
                proc,
                thread_id=thread_id,
                prompt=prompt,
                model=model,
                cwd=cwd or self.cwd,
                output_schema=output_schema,
                effort=effort,
                deadline=deadline,
            )
            return self._read_turn(proc, turn_id=turn_id, deadline=deadline)
        finally:
            self._close_process(proc)

    def _start_process(self) -> Any:
        argv = shlex.split(self.command)
        if not argv:
            raise CodexAppServerError("CODEX_APP_SERVER_COMMAND cannot be empty")
        if argv[-1] != "app-server":
            argv.append("app-server")
        return self.process_factory(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _initialize(self, proc: Any, deadline: float) -> None:
        self._send(
            proc,
            {
                "method": "initialize",
                "id": 1,
                "params": {
                    "clientInfo": {
                        "name": self.client_name,
                        "title": self.client_title,
                        "version": self.client_version,
                    }
                },
            },
        )
        self._read_response(proc, 1, deadline)
        self._send(proc, {"method": "initialized", "params": {}})

    def _start_thread(self, proc: Any, model: str, deadline: float) -> str:
        self._send(
            proc,
            {
                "method": "thread/start",
                "id": 2,
                "params": {
                    "model": model,
                    "sourceKind": "appServer",
                    "ephemeral": True,
                },
            },
        )
        response = self._read_response(proc, 2, deadline)
        thread_id = response.get("result", {}).get("thread", {}).get("id")
        if not thread_id:
            raise CodexAppServerError("Codex app-server did not return a thread id")
        return str(thread_id)

    def _start_turn(
        self,
        proc: Any,
        *,
        thread_id: str,
        prompt: str,
        model: str,
        cwd: str,
        output_schema: Optional[dict[str, Any]],
        effort: Optional[str],
        deadline: float,
    ) -> str:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": cwd,
            "approvalPolicy": DEFAULT_CODEX_APPROVAL_POLICY,
            "sandboxPolicy": {
                "type": "readOnly",
                "access": {"type": "fullAccess"},
            },
            "model": model,
            "summary": "none",
        }
        if effort:
            params["effort"] = effort
        if output_schema:
            params["outputSchema"] = output_schema

        self._send(proc, {"method": "turn/start", "id": 3, "params": params})
        response = self._read_response(proc, 3, deadline)
        turn_id = response.get("result", {}).get("turn", {}).get("id")
        if not turn_id:
            raise CodexAppServerError("Codex app-server did not return a turn id")
        return str(turn_id)

    def _read_turn(self, proc: Any, *, turn_id: str, deadline: float) -> str:
        streamed_text: list[str] = []
        final_text: Optional[str] = None

        while True:
            message = self._read_message(proc, deadline)
            if "id" in message and "method" in message:
                self._decline_server_request(proc, message)
                continue

            method = message.get("method")
            params = message.get("params", {})
            if method == "item/agentMessage/delta":
                delta = params.get("delta")
                if isinstance(delta, str):
                    streamed_text.append(delta)
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                    final_text = item["text"]
            elif method == "error":
                error = params.get("error", {})
                raise CodexAppServerError(
                    error.get("message") or "Codex app-server emitted an error"
                )
            elif method == "turn/completed":
                turn = params.get("turn", {})
                if turn.get("id") not in (None, turn_id):
                    continue
                status = turn.get("status")
                if status == "completed":
                    return final_text if final_text is not None else "".join(streamed_text)
                error = turn.get("error") or {}
                raise CodexAppServerError(
                    error.get("message") or f"Codex turn ended with status {status!r}"
                )

    def _read_response(self, proc: Any, response_id: int, deadline: float) -> dict[str, Any]:
        while True:
            message = self._read_message(proc, deadline)
            if message.get("id") == response_id:
                if "error" in message:
                    error = message["error"]
                    raise CodexAppServerError(
                        error.get("message") or f"Codex request {response_id} failed"
                    )
                return message
            if "id" in message and "method" in message:
                self._decline_server_request(proc, message)

    def _read_message(self, proc: Any, deadline: float) -> dict[str, Any]:
        if proc.stdout is None:
            raise CodexAppServerError("Codex app-server stdout is unavailable")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexAppServerError("Timed out waiting for Codex app-server")

        readable, _, _ = select.select([proc.stdout], [], [], remaining)
        if not readable:
            raise CodexAppServerError("Timed out waiting for Codex app-server")

        line = proc.stdout.readline()
        if not line:
            raise CodexAppServerError("Codex app-server exited without a response")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexAppServerError(f"Invalid JSON from Codex app-server: {line!r}") from exc

    def _send(self, proc: Any, message: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise CodexAppServerError("Codex app-server stdin is unavailable")
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def _decline_server_request(self, proc: Any, message: dict[str, Any]) -> None:
        self._send(proc, {"id": message["id"], "result": "decline"})

    def _close_process(self, proc: Any) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


class CodexAppServerChatModel(BaseChatModel):
    """LangChain chat model backed by a local Codex app-server process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(alias="model")
    runner: Any = Field(default=None, exclude=True)
    command: str = DEFAULT_CODEX_COMMAND
    cwd: Optional[str] = None
    timeout_seconds: float = DEFAULT_CODEX_TIMEOUT_SECONDS
    effort: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "codex-app-server"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._run_messages(
            messages,
            output_schema=kwargs.get("output_schema"),
            prompt_suffix=kwargs.get("prompt_suffix", ""),
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def bind_tools(
        self,
        tools: Iterable[Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return CodexToolBoundRunnable(self, list(tools))

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Runnable[Any, Any]:
        return CodexStructuredRunnable(self, schema)

    def _run_input(
        self,
        input_: Any,
        *,
        output_schema: Optional[dict[str, Any]] = None,
        prompt_suffix: str = "",
    ) -> str:
        if isinstance(input_, list) and all(isinstance(item, BaseMessage) for item in input_):
            messages = input_
        else:
            messages = self._coerce_messages(input_)
        return self._run_messages(
            messages,
            output_schema=output_schema,
            prompt_suffix=prompt_suffix,
        )

    def _run_messages(
        self,
        messages: list[BaseMessage],
        *,
        output_schema: Optional[dict[str, Any]] = None,
        prompt_suffix: str = "",
    ) -> str:
        prompt = _messages_to_prompt(messages)
        if prompt_suffix:
            prompt = f"{prompt}\n\n{prompt_suffix}"
        runner = self.runner or CodexAppServerProcessClient(
            command=self.command,
            cwd=self.cwd,
            timeout_seconds=self.timeout_seconds,
        )
        return runner.run_turn(
            prompt=prompt,
            model=self.model_name,
            cwd=self.cwd or os.getcwd(),
            output_schema=output_schema,
            effort=self.effort,
            timeout_seconds=self.timeout_seconds,
        )

    def _coerce_messages(self, input_: Any) -> list[BaseMessage]:
        if hasattr(input_, "to_messages"):
            return list(input_.to_messages())
        if isinstance(input_, str):
            return [HumanMessage(content=input_)]
        if isinstance(input_, tuple) and len(input_) == 2:
            role, content = input_
            return [HumanMessage(content=f"{role}: {content}")]
        if isinstance(input_, dict):
            role = input_.get("role", "human")
            content = input_.get("content", "")
            return [HumanMessage(content=f"{role}: {content}")]
        if isinstance(input_, list):
            return [_message_like_to_message(item) for item in input_]
        return [HumanMessage(content=str(input_))]


class CodexToolBoundRunnable(Runnable[Any, AIMessage]):
    """Runnable that asks Codex to emit LangChain-compatible tool calls."""

    def __init__(self, model: CodexAppServerChatModel, tools: list[Any]) -> None:
        self.model = model
        self.tool_specs = [_tool_to_spec(tool) for tool in tools]

    def invoke(
        self,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        raw = self.model._run_input(
            input,
            output_schema=TOOL_RESPONSE_SCHEMA,
            prompt_suffix=_tool_prompt_suffix(self.tool_specs),
        )
        payload = _parse_json_object(raw)
        tool_calls = payload.get("tool_calls") or []
        if tool_calls:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": str(call["name"]),
                        "args": call.get("args") or {},
                        "id": str(call.get("id") or f"call_{uuid.uuid4().hex}"),
                    }
                    for call in tool_calls
                ],
            )
        return AIMessage(content=str(payload.get("final") or raw))


class CodexStructuredRunnable(Runnable[Any, Any]):
    """Runnable returned by ``with_structured_output``."""

    def __init__(self, model: CodexAppServerChatModel, schema: type[BaseModel]) -> None:
        self.model = model
        self.schema = schema

    def invoke(
        self,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        schema_json = self.schema.model_json_schema()
        raw = self.model._run_input(
            input,
            output_schema=schema_json,
            prompt_suffix="Return JSON only. Do not include Markdown fences.",
        )
        payload = _parse_json_object(raw)
        return self.schema.model_validate(payload)


class CodexAppServerClient(BaseLLMClient):
    """Client factory wrapper for the Codex app-server provider."""

    provider = "codex"

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        return CodexAppServerChatModel(
            model=self.model,
            command=os.environ.get("CODEX_APP_SERVER_COMMAND", DEFAULT_CODEX_COMMAND),
            cwd=os.environ.get("CODEX_APP_SERVER_CWD") or os.getcwd(),
            timeout_seconds=float(
                os.environ.get(
                    "CODEX_APP_SERVER_TIMEOUT_SECONDS",
                    str(DEFAULT_CODEX_TIMEOUT_SECONDS),
                )
            ),
            effort=self.kwargs.get("reasoning_effort"),
        )

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)


def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.type
        content = _content_to_text(message.content)
        if not content:
            continue
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def _message_like_to_message(value: Any) -> BaseMessage:
    if isinstance(value, BaseMessage):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        role, content = value
        return HumanMessage(content=f"{role}: {content}")
    if isinstance(value, dict):
        role = value.get("role", "human")
        content = value.get("content", "")
        return HumanMessage(content=f"{role}: {content}")
    return HumanMessage(content=str(value))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(str(item.get("text") or item.get("content") or item))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(content)


def _tool_to_spec(tool: Any) -> dict[str, Any]:
    try:
        converted = convert_to_openai_tool(tool)
        function = converted.get("function", converted)
        return {
            "name": function.get("name") or getattr(tool, "name", tool.__name__),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }
    except Exception:
        return {
            "name": getattr(tool, "name", getattr(tool, "__name__", str(tool))),
            "description": getattr(tool, "description", ""),
            "parameters": {},
        }


def _tool_prompt_suffix(tool_specs: list[dict[str, Any]]) -> str:
    return (
        "You may call at most one tool before answering. Return JSON only. "
        "Do not include Markdown fences. If a tool is needed, return "
        '{"tool_calls":[{"id":"call_1","name":"tool_name","args":{...}}]}. '
        'If no tool is needed, return {"final":"your answer"}.\n\n'
        f"Available tools:\n{json.dumps(tool_specs, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not object_match:
            raise CodexAppServerError(f"Codex response was not JSON: {text!r}")
        payload = json.loads(object_match.group(0))

    if not isinstance(payload, dict):
        raise CodexAppServerError(f"Codex response JSON must be an object: {text!r}")
    return payload
