"""Flask server exposing the AI coding agent and a workspace REST API."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request

from agent import Agent
from tools import (
    ToolError,
    delete_file,
    ensure_workspace,
    list_files,
    read_file,
    run_command,
    write_file,
)

load_dotenv()

app = Flask(__name__)

# In-process rate limiter for /run: at most RUN_MAX_REQUESTS within
# RUN_WINDOW_SECONDS across all clients of this single-process app.
RUN_MAX_REQUESTS = 10
RUN_WINDOW_SECONDS = 60
_run_history: deque[float] = deque()
_run_lock = threading.Lock()

# Lazily instantiate the agent so the app can boot (and serve static files
# / file APIs) even if the OpenAI key is missing — only /chat will fail.
_agent: Agent | None = None
_agent_lock = threading.Lock()


def get_agent() -> Agent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = Agent()
        return _agent


def _seed_workspace() -> None:
    """Drop a sample file into the workspace on first run so the UI isn't empty."""
    workspace = ensure_workspace()
    sample = workspace / "hello.py"
    if not sample.exists():
        sample.write_text(
            'print("Hello from the AI coding agent workspace!")\n',
            encoding="utf-8",
        )


_seed_workspace()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        agent = get_agent()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    try:
        result = agent.chat(message)
    except Exception as exc:  # pragma: no cover - defensive surface
        app.logger.exception("agent chat failed")
        return jsonify({"error": f"agent error: {exc}"}), 500

    return jsonify(result)


@app.route("/chat/reset", methods=["POST"])
def chat_reset():
    try:
        get_agent().reset()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True})


@app.route("/files", methods=["GET"])
def files_index():
    directory = request.args.get("directory")
    try:
        entries = list_files(directory)
    except ToolError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"files": entries})


@app.route("/file/<path:relpath>", methods=["GET", "POST", "DELETE"])
def file_endpoint(relpath: str):
    if request.method == "GET":
        try:
            content = read_file(relpath)
        except ToolError as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify({"path": relpath, "content": content})

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        if "content" not in payload:
            return jsonify({"error": "content is required"}), 400
        try:
            message = write_file(relpath, payload["content"])
        except ToolError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "message": message})

    # DELETE
    try:
        message = delete_file(relpath)
    except ToolError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"ok": True, "message": message})


def _check_rate_limit() -> bool:
    now = time.monotonic()
    cutoff = now - RUN_WINDOW_SECONDS
    with _run_lock:
        while _run_history and _run_history[0] < cutoff:
            _run_history.popleft()
        if len(_run_history) >= RUN_MAX_REQUESTS:
            return False
        _run_history.append(now)
        return True


@app.route("/run", methods=["POST"])
def run_endpoint():
    if not _check_rate_limit():
        return (
            jsonify(
                {
                    "error": (
                        f"rate limit exceeded: max {RUN_MAX_REQUESTS} runs "
                        f"per {RUN_WINDOW_SECONDS} seconds"
                    )
                }
            ),
            429,
        )

    payload = request.get_json(silent=True) or {}
    command = (payload.get("command") or "").strip()
    if not command:
        return jsonify({"error": "command is required"}), 400

    try:
        result = run_command(command)
    except ToolError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
