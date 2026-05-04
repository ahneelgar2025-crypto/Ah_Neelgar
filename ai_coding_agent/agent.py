"""AI coding agent powered by OpenAI function calling.

The :class:`Agent` keeps an in-memory conversation history and exposes a
``chat`` method that takes a user message and returns the assistant's
final text reply. During each call the agent runs an inner loop that
executes any tool calls the model requests and feeds the results back
in until the model responds without further tool calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from tools import TOOL_SCHEMAS, ToolError, dispatch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOOL_ITERATIONS = 8
MAX_HISTORY_MESSAGES = 40

SYSTEM_PROMPT = """You are a helpful AI coding assistant embedded in a small \
web app. The user has a sandboxed workspace directory and you have tools to \
read, write, list, and search files inside it, plus a tool to run shell \
commands there.

Guidelines:
- Briefly explain what you are about to do before invoking tools.
- Prefer using tools to inspect the workspace instead of guessing.
- Write clean, idiomatic code and keep edits minimal and focused.
- After making changes, summarize what you changed and why.
- If a request is ambiguous or risky, ask a clarifying question instead \
  of assuming.
- Never attempt to escape the workspace sandbox or run destructive \
  commands without confirmation.
"""


class Agent:
    """Conversational coding agent with tool use."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env "
                "and add your key."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.system_prompt = system_prompt
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history = []

    def _trim_history(self) -> None:
        """Keep history bounded so the context window stays manageable."""
        if len(self.history) > MAX_HISTORY_MESSAGES:
            # Always keep the most recent messages; drop the oldest.
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def _build_messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt}, *self.history]

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            content = json.dumps({"error": f"invalid JSON arguments: {exc}"})
        else:
            try:
                result = dispatch(name, arguments)
                content = json.dumps({"result": result}, default=str)
            except ToolError as exc:
                content = json.dumps({"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("tool %s raised", name)
                content = json.dumps({"error": f"unexpected tool error: {exc}"})

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": name,
            "content": content,
        }

    def chat(self, user_message: str) -> dict[str, Any]:
        """Send ``user_message`` to the model and return the final reply.

        The returned dict has shape::

            {"reply": str, "tool_calls": list[{"name", "arguments", "result"}]}
        """
        if not user_message or not user_message.strip():
            raise ValueError("user_message must be non-empty")

        self.history.append({"role": "user", "content": user_message})
        tool_invocations: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(),
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            self.history.append(assistant_entry)

            if not message.tool_calls:
                self._trim_history()
                return {
                    "reply": message.content or "",
                    "tool_calls": tool_invocations,
                }

            for tool_call in message.tool_calls:
                tool_message = self._execute_tool_call(tool_call)
                self.history.append(tool_message)
                try:
                    parsed = json.loads(tool_message["content"])
                except json.JSONDecodeError:
                    parsed = {"raw": tool_message["content"]}
                tool_invocations.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "result": parsed,
                    }
                )

        # Hit the iteration cap; return whatever the model said last.
        self._trim_history()
        last_text = ""
        for entry in reversed(self.history):
            if entry.get("role") == "assistant" and entry.get("content"):
                last_text = entry["content"]
                break
        return {
            "reply": last_text
            or "(stopped after too many tool calls without a final reply)",
            "tool_calls": tool_invocations,
        }
