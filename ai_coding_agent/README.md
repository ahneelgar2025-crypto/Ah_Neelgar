# AI Coding Agent

A simplified AI coding assistant built with Flask and OpenAI. It exposes a
chat UI plus a file browser and a CodeMirror editor over a sandboxed
`workspace/` directory. The agent can read, write, list, search, and
execute commands inside that directory using OpenAI's function-calling
API.

This is a separate Flask app from the face-recognition app at the repo
root. It lives entirely under `ai_coding_agent/`.

## Features

- Chat with an OpenAI-backed coding agent that uses tools to inspect and
  modify a workspace directory.
- File tree, CodeMirror editor (syntax highlighting for several
  languages), and Save/Delete buttons.
- Markdown + code-block rendering of agent replies via marked.js +
  highlight.js.
- A run bar that shells out to the workspace, with output displayed
  inline.
- Path-traversal protection on every filesystem tool, a 30-second
  command timeout, and a simple in-memory rate limit on `/run`.

## Quickstart

```bash
cd ai_coding_agent
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then edit .env and add your OpenAI key
python agent_app.py
```

The server listens on <http://127.0.0.1:5050>. Override the port with
`PORT=… python agent_app.py`.

## Environment variables

| Variable         | Default        | Description                              |
| ---------------- | -------------- | ---------------------------------------- |
| `OPENAI_API_KEY` | _required_     | Your OpenAI API key (read from `.env`).  |
| `OPENAI_MODEL`   | `gpt-4o-mini`  | Any chat model that supports tool use.   |
| `PORT`           | `5050`         | Port for the Flask dev server.           |
| `FLASK_DEBUG`    | `0`            | Set to `1` to enable Flask debug mode.   |

## HTTP API

| Method   | Path                   | Description                                    |
| -------- | ---------------------- | ---------------------------------------------- |
| `GET`    | `/`                    | Serve the chat / editor UI.                    |
| `POST`   | `/chat`                | `{message}` → `{reply, tool_calls}`.           |
| `POST`   | `/chat/reset`          | Clear in-memory conversation history.          |
| `GET`    | `/files`               | List entries in the workspace directory.       |
| `GET`    | `/file/<path>`         | Read a workspace file as text.                 |
| `POST`   | `/file/<path>`         | `{content}` → write/overwrite a file.          |
| `DELETE` | `/file/<path>`         | Delete a workspace file.                       |
| `POST`   | `/run`                 | `{command}` → run a shell command (rate-limited). |

## Agent tools

The agent is given access to these JSON-schema tools (defined in
`tools.py`): `read_file`, `write_file`, `list_files`, `run_command`,
`search_files`. The agent loop runs up to 8 iterations of tool calls
before giving up and returning its last text response.

## Project layout

```
ai_coding_agent/
├── agent_app.py         # Flask routes
├── agent.py             # OpenAI client + agent loop
├── tools.py             # Sandboxed filesystem + shell tools
├── templates/index.html # Chat + editor UI
├── static/
│   ├── style.css
│   └── app.js
├── workspace/           # Created on startup; agent edits live here
├── requirements.txt
├── .env.example
└── README.md
```

## Security notes

This is a learning project. **Do not expose this server to the
internet.** The `run_command` tool executes arbitrary shell commands
inside the workspace directory; the only safeguards are:

- a hard 30-second per-command timeout,
- output capped at 20 KB per stream,
- a simple in-memory rate limit (10 requests per 60 seconds) on `/run`,
- and path-traversal protection on file operations.

There is no real sandbox. Run the app locally, behind `127.0.0.1`, on a
machine you trust.
