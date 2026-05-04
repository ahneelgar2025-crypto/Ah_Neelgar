"""Tool implementations for the AI coding agent.

All filesystem operations and command execution are sandboxed to the
``workspace/`` directory that lives next to this module. Paths supplied
by callers are resolved relative to that directory and validated to
prevent path-traversal escapes.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"
COMMAND_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 20_000
MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB read/write cap to keep things sane


class ToolError(Exception):
    """Raised when a tool fails in a way that should be surfaced to the LLM."""


def ensure_workspace() -> Path:
    """Create the workspace directory if it does not exist and return it."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def _safe_path(relative_path: str) -> Path:
    """Resolve a user-supplied path inside the workspace directory.

    Raises ``ToolError`` if the resolved path escapes the workspace.
    """
    if relative_path is None:
        raise ToolError("path is required")

    cleaned = str(relative_path).strip().lstrip("/\\")
    if not cleaned or cleaned in {".", "./"}:
        return ensure_workspace()

    workspace = ensure_workspace().resolve()
    candidate = (workspace / cleaned).resolve()

    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise ToolError(
            f"path '{relative_path}' is outside the workspace sandbox"
        ) from exc

    return candidate


def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace."""
    target = _safe_path(path)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    if target.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ToolError(f"file too large to read: {path}")
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"file is not UTF-8 text: {path}") from exc


def write_file(path: str, content: str) -> str:
    """Write ``content`` to ``path`` inside the workspace, creating dirs."""
    if content is None:
        content = ""
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ToolError("content exceeds maximum file size")

    target = _safe_path(path)
    if target == ensure_workspace().resolve():
        raise ToolError("refusing to overwrite the workspace root")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def delete_file(path: str) -> str:
    """Delete a file inside the workspace."""
    target = _safe_path(path)
    if target == ensure_workspace().resolve():
        raise ToolError("refusing to delete the workspace root")
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    if target.is_dir():
        raise ToolError(f"refusing to delete directory: {path}")
    target.unlink()
    return f"deleted {path}"


def list_files(directory: str | None = None) -> list[dict]:
    """List files and directories inside the workspace.

    Returns a list of ``{"path", "type", "size"}`` dicts. ``path`` is
    relative to the workspace root.
    """
    workspace = ensure_workspace().resolve()
    base = _safe_path(directory) if directory else workspace
    if not base.exists():
        raise ToolError(f"directory not found: {directory}")
    if not base.is_dir():
        raise ToolError(f"not a directory: {directory}")

    entries: list[dict] = []
    for entry in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        rel = entry.relative_to(workspace).as_posix()
        if entry.is_dir():
            entries.append({"path": rel, "type": "dir", "size": 0})
        else:
            entries.append(
                {"path": rel, "type": "file", "size": entry.stat().st_size}
            )
    return entries


def search_files(query: str, directory: str | None = None) -> list[dict]:
    """grep-like search through text files in the workspace.

    Returns a list of ``{"path", "line", "text"}`` matches (case-insensitive).
    Binary or oversized files are skipped silently.
    """
    if not query:
        raise ToolError("query is required")

    workspace = ensure_workspace().resolve()
    base = _safe_path(directory) if directory else workspace
    if not base.exists() or not base.is_dir():
        raise ToolError(f"directory not found: {directory}")

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results: list[dict] = []
    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                results.append(
                    {
                        "path": file_path.relative_to(workspace).as_posix(),
                        "line": lineno,
                        "text": line.rstrip()[:500],
                    }
                )
                if len(results) >= 200:
                    return results
    return results


def run_command(command: str) -> dict:
    """Run a shell command inside the workspace directory.

    Returns ``{"stdout", "stderr", "returncode", "timed_out"}``. Output
    is truncated to ``MAX_OUTPUT_CHARS`` per stream to bound payload size.
    """
    if not command or not command.strip():
        raise ToolError("command is required")

    workspace = ensure_workspace()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "stdout": (exc.stdout or "")[:MAX_OUTPUT_CHARS]
            if isinstance(exc.stdout, str)
            else "",
            "stderr": (exc.stderr or "")[:MAX_OUTPUT_CHARS]
            if isinstance(exc.stderr, str)
            else "",
            "returncode": -1,
            "timed_out": True,
        }

    return {
        "stdout": completed.stdout[:MAX_OUTPUT_CHARS],
        "stderr": completed.stderr[:MAX_OUTPUT_CHARS],
        "returncode": completed.returncode,
        "timed_out": False,
    }


# JSON-schema definitions of the tools, in the format the OpenAI chat
# completions API expects for function calling.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a UTF-8 text file inside the "
                "workspace. Creates parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories inside the workspace, "
                "optionally scoped to a subdirectory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Optional subdirectory relative to the "
                            "workspace root."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command inside the workspace. Subject to a "
                f"{COMMAND_TIMEOUT_SECONDS}-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Case-insensitive substring search across files in the "
                "workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": (
                            "Optional subdirectory to search within."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]


TOOL_DISPATCH = {
    "read_file": lambda args: read_file(args["path"]),
    "write_file": lambda args: write_file(args["path"], args.get("content", "")),
    "list_files": lambda args: list_files(args.get("directory")),
    "run_command": lambda args: run_command(args["command"]),
    "search_files": lambda args: search_files(
        args["query"], args.get("directory")
    ),
}


def dispatch(name: str, arguments: dict):
    """Execute a tool by name, returning its raw result or raising ``ToolError``."""
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        raise ToolError(f"unknown tool: {name}")
    return handler(arguments or {})
