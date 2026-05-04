(() => {
  "use strict";

  const fileTreeEl = document.getElementById("file-tree");
  const refreshFilesBtn = document.getElementById("refresh-files");
  const currentFileEl = document.getElementById("current-file");
  const saveBtn = document.getElementById("save-file");
  const deleteBtn = document.getElementById("delete-file");
  const commandInput = document.getElementById("command-input");
  const runBtn = document.getElementById("run-command");
  const runOutput = document.getElementById("run-output");
  const chatLog = document.getElementById("chat-log");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const resetChatBtn = document.getElementById("reset-chat");

  const editor = CodeMirror.fromTextArea(document.getElementById("editor"), {
    lineNumbers: true,
    theme: "material-darker",
    indentUnit: 4,
    tabSize: 4,
    mode: "python",
  });

  let currentFile = null;
  let dirty = false;

  if (window.marked && window.hljs) {
    window.marked.setOptions({
      highlight: (code, lang) => {
        try {
          if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
          }
          return hljs.highlightAuto(code).value;
        } catch (_) {
          return code;
        }
      },
    });
  }

  function modeForPath(path) {
    const ext = path.split(".").pop().toLowerCase();
    switch (ext) {
      case "py":
        return "python";
      case "js":
      case "json":
        return "javascript";
      case "html":
      case "htm":
        return "htmlmixed";
      case "css":
        return "css";
      case "md":
        return "markdown";
      default:
        return "text/plain";
    }
  }

  async function jsonFetch(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      data = null;
    }
    if (!response.ok) {
      const message = (data && data.error) || `request failed (${response.status})`;
      throw new Error(message);
    }
    return data;
  }

  async function loadFiles() {
    try {
      const data = await jsonFetch("/files");
      renderFileTree(data.files || []);
    } catch (err) {
      fileTreeEl.innerHTML = `<li class="error">${escapeHtml(err.message)}</li>`;
    }
  }

  function renderFileTree(entries) {
    fileTreeEl.innerHTML = "";
    if (!entries.length) {
      const empty = document.createElement("li");
      empty.textContent = "(empty)";
      empty.style.color = "var(--text-dim)";
      empty.style.cursor = "default";
      fileTreeEl.appendChild(empty);
      return;
    }
    for (const entry of entries) {
      const li = document.createElement("li");
      li.textContent = entry.path;
      li.title = entry.path;
      li.classList.add(entry.type === "dir" ? "dir" : "file");
      if (entry.path === currentFile) {
        li.classList.add("active");
      }
      if (entry.type === "file") {
        li.addEventListener("click", () => openFile(entry.path));
      }
      fileTreeEl.appendChild(li);
    }
  }

  async function openFile(path) {
    if (dirty && !confirm("Discard unsaved changes?")) {
      return;
    }
    try {
      const data = await jsonFetch(`/file/${encodeURI(path)}`);
      currentFile = path;
      dirty = false;
      currentFileEl.textContent = path;
      editor.setOption("mode", modeForPath(path));
      editor.setValue(data.content || "");
      saveBtn.disabled = false;
      deleteBtn.disabled = false;
      [...fileTreeEl.children].forEach((li) =>
        li.classList.toggle("active", li.textContent === path)
      );
    } catch (err) {
      alert(`Failed to open ${path}: ${err.message}`);
    }
  }

  editor.on("change", () => {
    dirty = true;
  });

  saveBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    saveBtn.disabled = true;
    try {
      await jsonFetch(`/file/${encodeURI(currentFile)}`, {
        method: "POST",
        body: JSON.stringify({ content: editor.getValue() }),
      });
      dirty = false;
      flashStatus(`Saved ${currentFile}`);
    } catch (err) {
      alert(`Save failed: ${err.message}`);
    } finally {
      saveBtn.disabled = false;
    }
  });

  deleteBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    if (!confirm(`Delete ${currentFile}?`)) return;
    try {
      await jsonFetch(`/file/${encodeURI(currentFile)}`, { method: "DELETE" });
      currentFile = null;
      dirty = false;
      currentFileEl.textContent = "No file selected";
      editor.setValue("");
      saveBtn.disabled = true;
      deleteBtn.disabled = true;
      await loadFiles();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  });

  refreshFilesBtn.addEventListener("click", () => loadFiles());

  runBtn.addEventListener("click", runCommand);
  commandInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runCommand();
    }
  });

  async function runCommand() {
    const command = commandInput.value.trim();
    if (!command) return;
    runBtn.disabled = true;
    runOutput.textContent = `$ ${command}\n(running...)`;
    try {
      const data = await jsonFetch("/run", {
        method: "POST",
        body: JSON.stringify({ command }),
      });
      const parts = [`$ ${command}`];
      if (data.stdout) parts.push(data.stdout);
      if (data.stderr) parts.push(`[stderr]\n${data.stderr}`);
      parts.push(
        data.timed_out
          ? "[timed out]"
          : `[exit ${data.returncode}]`
      );
      runOutput.textContent = parts.join("\n");
      await loadFiles();
    } catch (err) {
      runOutput.textContent = `$ ${command}\nerror: ${err.message}`;
    } finally {
      runBtn.disabled = false;
    }
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    sendChat();
  });
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  resetChatBtn.addEventListener("click", async () => {
    if (!confirm("Clear chat history?")) return;
    try {
      await jsonFetch("/chat/reset", { method: "POST" });
      chatLog.innerHTML = "";
    } catch (err) {
      alert(`Reset failed: ${err.message}`);
    }
  });

  async function sendChat() {
    const message = chatInput.value.trim();
    if (!message) return;
    chatInput.value = "";
    appendBubble("user", escapeHtml(message));
    const pending = appendBubble("agent", "<em>thinking…</em>");
    try {
      const data = await jsonFetch("/chat", {
        method: "POST",
        body: JSON.stringify({ message }),
      });
      pending.remove();
      if (data.tool_calls && data.tool_calls.length) {
        for (const call of data.tool_calls) {
          appendToolBubble(call);
        }
      }
      const reply = data.reply || "(no reply)";
      const html = window.marked
        ? window.marked.parse(reply)
        : escapeHtml(reply);
      appendBubble("agent", html, false);
      await loadFiles();
      if (currentFile) {
        // Refresh current file in case the agent edited it.
        try {
          const fresh = await jsonFetch(`/file/${encodeURI(currentFile)}`);
          if (fresh.content !== editor.getValue()) {
            const cursor = editor.getCursor();
            editor.setValue(fresh.content);
            editor.setCursor(cursor);
            dirty = false;
          }
        } catch (_) {
          /* file may have been deleted */
        }
      }
    } catch (err) {
      pending.remove();
      appendBubble("error", escapeHtml(err.message));
    }
  }

  function appendBubble(kind, html) {
    const div = document.createElement("div");
    div.className = `bubble ${kind}`;
    div.innerHTML = html;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    if (window.hljs) {
      div.querySelectorAll("pre code").forEach((block) => {
        try {
          window.hljs.highlightElement(block);
        } catch (_) {
          /* ignore */
        }
      });
    }
    return div;
  }

  function appendToolBubble(call) {
    const summary = `${call.name}(${truncate(call.arguments || "", 120)})`;
    let resultText = "";
    if (call.result && Object.prototype.hasOwnProperty.call(call.result, "error")) {
      resultText = `error: ${call.result.error}`;
    } else if (call.result && Object.prototype.hasOwnProperty.call(call.result, "result")) {
      resultText = stringifyResult(call.result.result);
    } else {
      resultText = JSON.stringify(call.result);
    }
    const html =
      `<div class="tool-summary">tool · ${escapeHtml(call.name)}</div>` +
      `<div>${escapeHtml(summary)}</div>` +
      `<div style="margin-top:4px;color:var(--text-dim);">${escapeHtml(
        truncate(resultText, 1500)
      )}</div>`;
    appendBubble("tool", html);
  }

  function stringifyResult(value) {
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch (_) {
      return String(value);
    }
  }

  function truncate(text, max) {
    text = String(text);
    return text.length > max ? `${text.slice(0, max)}…` : text;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function flashStatus(text) {
    const prev = currentFileEl.textContent;
    currentFileEl.textContent = text;
    setTimeout(() => {
      currentFileEl.textContent = currentFile || prev;
    }, 1200);
  }

  loadFiles();
})();
