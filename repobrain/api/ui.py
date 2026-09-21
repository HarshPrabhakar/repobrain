from __future__ import annotations

from fastapi.responses import HTMLResponse


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RepoBrain</title>
<style>
:root {
  color-scheme: dark;
  --bg:#08111f;
  --panel:#0f1b2d;
  --panel2:#132239;
  --border:#243956;
  --text:#eef5ff;
  --muted:#94a9c4;
  --accent:#67e8f9;
  --accent2:#a78bfa;
  --good:#4ade80;
  --bad:#fb7185;
  --warn:#facc15;
}
*{box-sizing:border-box}
body{
  margin:0;background:
  radial-gradient(circle at 15% 0%,rgba(103,232,249,.08),transparent 24%),
  radial-gradient(circle at 100% 100%,rgba(167,139,250,.08),transparent 28%),
  var(--bg);
  color:var(--text);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
button,input,textarea{font:inherit}
.shell{width:min(1500px,calc(100% - 28px));margin:auto;padding:22px 0 36px}
.topbar{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:44px;height:44px;border:1px solid var(--border);border-radius:14px;display:grid;place-items:center;font-weight:800;background:linear-gradient(135deg,rgba(103,232,249,.16),rgba(167,139,250,.18))}
.brand h1{margin:0;font-size:1.12rem}
.brand p{margin:3px 0 0;color:var(--muted);font-size:.82rem}
.actions{display:flex;flex-wrap:wrap;gap:8px}
button,.btn{
  border:1px solid var(--border);background:var(--panel2);color:var(--text);
  padding:9px 13px;border-radius:10px;cursor:pointer;text-decoration:none;transition:.15s
}
button:hover,.btn:hover{border-color:#426186;transform:translateY(-1px)}
button:disabled{opacity:.45;cursor:not-allowed;transform:none}
.primary{background:linear-gradient(135deg,rgba(103,232,249,.14),rgba(167,139,250,.15))}
.panel{background:rgba(15,27,45,.96);border:1px solid var(--border);border-radius:16px}
.statusbar{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.metric{padding:14px}
.metric small{display:block;color:var(--muted);margin-bottom:6px}
.metric strong{font-size:1.25rem}
.workspace{display:grid;grid-template-columns:390px minmax(0,1fr);gap:14px;align-items:start}
.sidebar{padding:16px;position:sticky;top:12px}
.sidebar h2,.chat h2{margin:0;font-size:1rem}
.section-note{color:var(--muted);font-size:.78rem}
.field{margin-top:14px}
label{display:block;font-size:.78rem;color:var(--muted);margin-bottom:7px}
input,textarea{
  width:100%;border:1px solid var(--border);background:#091424;color:var(--text);
  border-radius:10px;padding:11px 12px;outline:none
}
input:focus,textarea:focus{border-color:#4e749c}
.repo-actions{display:flex;gap:8px;margin-top:10px}
.repo-status{margin-top:14px;padding:12px;border:1px solid var(--border);border-radius:12px;background:rgba(9,20,36,.6)}
.row{display:flex;justify-content:space-between;gap:10px;margin-top:7px;font-size:.79rem}
.row:first-child{margin-top:0}
.row span:first-child{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;word-break:break-all}
.badge{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--border);border-radius:999px;padding:5px 9px;font-size:.75rem;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--warn)}
.dot.good{background:var(--good)}
.dot.bad{background:var(--bad)}
.chat{min-height:720px;display:flex;flex-direction:column}
.chat-head{padding:16px 18px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:12px;align-items:center}
.chat-meta{display:flex;gap:8px;flex-wrap:wrap}
.messages{flex:1;padding:20px;overflow-y:auto;max-height:620px}
.empty{text-align:center;color:var(--muted);padding:70px 20px}
.msg{display:flex;margin:12px 0}
.msg.user{justify-content:flex-end}
.bubble{max-width:min(860px,88%);padding:12px 14px;border-radius:14px;line-height:1.55;white-space:pre-wrap}
.user .bubble{background:linear-gradient(135deg,rgba(103,232,249,.15),rgba(167,139,250,.14));border:1px solid rgba(103,232,249,.25)}
.assistant .bubble{background:#0a1626;border:1px solid var(--border)}
.meta-line{margin-top:8px;color:var(--muted);font-size:.72rem}
.citations{margin-top:9px;display:flex;flex-wrap:wrap;gap:6px}
.citation{border:1px solid var(--border);border-radius:999px;padding:4px 8px;color:#bdd1ea;font-size:.72rem}
.composer{border-top:1px solid var(--border);padding:14px}
.composer-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:9px}
textarea{min-height:54px;max-height:160px;resize:vertical}
.hint{color:var(--muted);font-size:.72rem;margin-top:7px}
.toast{position:fixed;right:20px;bottom:20px;background:#0c1828;border:1px solid var(--border);padding:11px 14px;border-radius:10px;display:none;max-width:420px;box-shadow:0 16px 45px rgba(0,0,0,.35)}
.toast.bad{border-color:rgba(251,113,133,.45)}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.22);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:7px}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:950px){
  .workspace{grid-template-columns:1fr}
  .sidebar{position:static}
  .statusbar{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:560px){
  .statusbar{grid-template-columns:1fr}
  .topbar{align-items:flex-start;flex-direction:column}
}
</style>
</head>
<body>
<main class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="logo">RB</div>
      <div>
        <h1>RepoBrain</h1>
        <p>Inspect a local repository and ask grounded questions about its code.</p>
      </div>
    </div>
    <div class="actions">
      <button id="refreshBtn">Refresh</button>
      <a class="btn" href="/docs" target="_blank">API Docs</a>
    </div>
  </header>

  <section class="statusbar">
    <div class="panel metric"><small>Service</small><strong id="serviceStatus">Checking</strong></div>
    <div class="panel metric"><small>Loaded repositories</small><strong id="repoCount">0</strong></div>
    <div class="panel metric"><small>Active sessions</small><strong id="sessionCount">0</strong></div>
    <div class="panel metric"><small>API version</small><strong>11.2</strong></div>
  </section>

  <section class="workspace">
    <aside class="panel sidebar">
      <div style="display:flex;justify-content:space-between;gap:10px;align-items:center">
        <div>
          <h2>Repository</h2>
          <div class="section-note">Use a local folder path on this PC.</div>
        </div>
        <span class="badge"><span id="repoDot" class="dot"></span><span id="repoBadge">Not loaded</span></span>
      </div>

      <div class="field">
        <label for="repoPath">Local repository path</label>
        <input id="repoPath" type="text" placeholder="D:\Projects\my-repository" />
      </div>

      <div class="repo-actions">
        <button id="openRepoBtn" class="primary" style="flex:1">Open repository</button>
        <button id="closeRepoBtn" disabled>Close</button>
      </div>

      <div id="buildMessage" class="hint"></div>

      <div id="repoInfo" class="repo-status">
        <div class="row"><span>Runtime</span><span id="runtimeId" class="mono">—</span></div>
        <div class="row"><span>Fingerprint</span><span id="fingerprint" class="mono">—</span></div>
        <div class="row"><span>Files</span><span id="files">—</span></div>
        <div class="row"><span>Symbols</span><span id="symbols">—</span></div>
        <div class="row"><span>Relationships</span><span id="relationships">—</span></div>
        <div class="row"><span>Graph</span><span id="graph">—</span></div>
        <div class="row"><span>Embedding</span><span id="embedding">—</span></div>
        <div class="row"><span>Device</span><span id="device">—</span></div>
        <div class="row"><span>LLM</span><span id="llm">—</span></div>
      </div>

      <div class="field">
        <label>Conversation</label>
        <div class="repo-actions">
          <button id="newSessionBtn" style="flex:1" disabled>Start session</button>
          <button id="clearBtn" disabled>Clear</button>
          <button id="newConversationBtn" disabled>New</button>
        </div>
      </div>

      <div class="repo-status">
        <div class="row"><span>Session</span><span id="sessionId" class="mono">—</span></div>
        <div class="row"><span>Conversation</span><span id="conversationId" class="mono">—</span></div>
        <div class="row"><span>Turn</span><span id="turnNumber">0</span></div>
        <div class="row"><span>Current focus</span><span id="currentFocus" class="mono">—</span></div>
        <div class="row"><span>Relation focus</span><span id="relationFocus" class="mono">—</span></div>
      </div>
    </aside>

    <section class="panel chat">
      <div class="chat-head">
        <div>
          <h2>Ask RepoBrain</h2>
          <div class="section-note">Answers are generated from repository evidence and structural graph facts.</div>
        </div>
        <div class="chat-meta">
          <span class="badge"><span id="sessionDot" class="dot"></span><span id="sessionBadge">No session</span></span>
        </div>
      </div>

      <div id="messages" class="messages">
        <div class="empty">
          <strong>Open a repository to begin.</strong><br /><br />
          Enter a local folder path on the left. RepoBrain will build its scanner, AST,
          symbol resolution, BM25, semantic index, knowledge graph, and grounded answer layer.
        </div>
      </div>

      <div class="composer">
        <div class="composer-row">
          <textarea id="question" placeholder="Try: who calls _calculate_sha256?" disabled></textarea>
          <button id="askBtn" class="primary" disabled>Ask</button>
        </div>
        <div class="hint">
          Enter to send · Shift+Enter for a new line · Structural caller/callee questions use deterministic graph truth.
        </div>
      </div>
    </section>
  </section>
</main>

<div id="toast" class="toast"></div>

<script>
const $ = id => document.getElementById(id);

const state = {
  repositoryRoot: null,
  runtimeId: null,
  sessionId: null,
  conversationId: null,
  busy: false,
};

function shortId(value) {
  if (!value) return "—";
  return value.length > 20 ? `${value.slice(0,9)}…${value.slice(-7)}` : value;
}

function toast(message, bad=false) {
  const el = $("toast");
  el.textContent = message;
  el.className = bad ? "toast bad" : "toast";
  el.style.display = "block";
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.style.display = "none", 4200);
}

async function api(path, options={}) {
  const response = await fetch(path, {
    headers: {"Content-Type":"application/json"},
    ...options,
  });

  let body = null;
  try { body = await response.json(); } catch {}

  if (!response.ok) {
    const message = body?.message || `HTTP ${response.status}`;
    const error = new Error(message);
    error.payload = body;
    error.status = response.status;
    throw error;
  }

  return body;
}

function setBusy(busy, text="") {
  state.busy = busy;
  $("openRepoBtn").disabled = busy;
  $("askBtn").disabled = busy || !state.sessionId;
  $("question").disabled = busy || !state.sessionId;
  $("newSessionBtn").disabled = busy || !state.runtimeId;
  $("clearBtn").disabled = busy || !state.sessionId;
  $("newConversationBtn").disabled = busy || !state.sessionId;
  $("closeRepoBtn").disabled = busy || !state.runtimeId;
  $("buildMessage").innerHTML = busy ? `<span class="spinner"></span>${text}` : "";
}

function setRepositoryStatus(loaded) {
  $("repoDot").className = loaded ? "dot good" : "dot";
  $("repoBadge").textContent = loaded ? "Loaded" : "Not loaded";
}

function setSessionStatus(active) {
  $("sessionDot").className = active ? "dot good" : "dot";
  $("sessionBadge").textContent = active ? "Active" : "No session";
}

function renderState(s) {
  $("conversationId").textContent = shortId(s?.conversation_id);
  $("turnNumber").textContent = s?.turn_number ?? 0;
  $("currentFocus").textContent = s?.current_qualified_name || "—";
  $("relationFocus").textContent = s?.relationship_qualified_name || "—";
}

function clearMessages(message="Session ready. Ask a question about the repository.") {
  $("messages").innerHTML = `<div class="empty">${message}</div>`;
}

function addMessage(role, text, meta="", citations=[]) {
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const citationHtml = citations?.length
    ? `<div class="citations">${citations.map(c => {
        const loc = c.relative_path
          ? `${c.relative_path}${c.start_line ? `:${c.start_line}` : ""}`
          : (c.qualified_name || c.source_qualified_name || "");
        return `<span class="citation">${escapeHtml(c.citation_id)}${loc ? " · " + escapeHtml(loc) : ""}</span>`;
      }).join("")}</div>`
    : "";

  wrapper.innerHTML = `
    <div class="bubble">
      ${escapeHtml(text)}
      ${citationHtml}
      ${meta ? `<div class="meta-line">${escapeHtml(meta)}</div>` : ""}
    </div>
  `;

  $("messages").appendChild(wrapper);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    $("serviceStatus").textContent = health.ready ? "Ready" : "Not ready";
    $("repoCount").textContent = health.loaded_repositories ?? 0;
    $("sessionCount").textContent = health.active_sessions ?? 0;
  } catch {
    $("serviceStatus").textContent = "Offline";
  }
}

async function loadRepositoryDiagnostics() {
  if (!state.repositoryRoot) return;

  try {
    const d = await api("/repositories/diagnostics", {
      method:"POST",
      body:JSON.stringify({repository_root:state.repositoryRoot}),
    });

    if (!d) return;

    const rt = d.runtime || {};
    const diag = d.runtime_diagnostics || {};

    $("runtimeId").textContent = shortId(rt.runtime_id);
    $("fingerprint").textContent = shortId(rt.fingerprint_value);
    $("files").textContent = diag.files_discovered ?? "—";
    $("symbols").textContent = diag.symbols ?? "—";
    $("relationships").textContent = diag.relationships ?? "—";
    $("graph").textContent =
      (diag.graph_nodes != null && diag.graph_edges != null)
        ? `${diag.graph_nodes} nodes / ${diag.graph_edges} edges`
        : "—";
    $("embedding").textContent =
      diag.embedding_model
        ? `${diag.embedding_model} (${diag.embedding_dimension ?? "?"}d)`
        : "—";
    $("device").textContent = diag.embedding_device ?? "—";
    $("llm").textContent = diag.llm_model ?? "—";
  } catch (error) {
    console.error(error);
  }
}

async function openRepository() {
  const repositoryRoot = $("repoPath").value.trim();

  if (!repositoryRoot) {
    toast("Enter a local repository path first.", true);
    return;
  }

  setBusy(true, "Building repository intelligence. This can take a moment...");

  try {
    const runtime = await api("/repositories/open", {
      method:"POST",
      body:JSON.stringify({repository_root:repositoryRoot}),
    });

    state.repositoryRoot = repositoryRoot;
    state.runtimeId = runtime.runtime_id;
    state.sessionId = null;
    state.conversationId = null;

    setRepositoryStatus(true);
    setSessionStatus(false);
    $("runtimeId").textContent = shortId(runtime.runtime_id);
    $("fingerprint").textContent = shortId(runtime.fingerprint_value);
    $("sessionId").textContent = "—";
    $("conversationId").textContent = "—";
    renderState(null);

    await loadRepositoryDiagnostics();
    await refreshHealth();

    setBusy(false);
    $("newSessionBtn").disabled = false;

    clearMessages("Repository loaded. Start a session to ask questions.");
    toast("Repository loaded successfully.");
  } catch (error) {
    setBusy(false);
    setRepositoryStatus(false);
    toast(error.message, true);
  }
}

async function createSession() {
  if (!state.repositoryRoot) return;

  setBusy(true, "Creating investigation session...");

  try {
    const session = await api("/sessions", {
      method:"POST",
      body:JSON.stringify({repository_root:state.repositoryRoot}),
    });

    state.sessionId = session.session_id;
    state.conversationId = session.conversation_id;

    $("sessionId").textContent = shortId(session.session_id);
    $("conversationId").textContent = shortId(session.conversation_id);
    renderState(session);
    setSessionStatus(true);

    setBusy(false);
    clearMessages("Session ready. Ask RepoBrain about this repository.");
    $("question").focus();
    await refreshHealth();
  } catch (error) {
    setBusy(false);
    toast(error.message, true);
  }
}

async function askQuestion() {
  const query = $("question").value.trim();

  if (!query || !state.sessionId || !state.repositoryRoot || state.busy) return;

  $("question").value = "";
  addMessage("user", query);
  setBusy(true, "RepoBrain is inspecting repository evidence...");

  try {
    const result = await api("/query/ask", {
      method:"POST",
      body:JSON.stringify({
        repository_root:state.repositoryRoot,
        session_id:state.sessionId,
        query,
      }),
    });

    const meta = [
      result.intent ? `intent: ${result.intent}` : "",
      result.model_name ? `model: ${result.model_name}` : "",
      `grounded: ${result.grounded ? "yes" : "no"}`,
    ].filter(Boolean).join(" · ");

    addMessage(
      "assistant",
      result.answer_text,
      meta,
      result.citations || []
    );

    renderState(result.state);

    setBusy(false);
    $("question").focus();
    await refreshHealth();
  } catch (error) {
    setBusy(false);

    if (error.payload?.code === "SESSION_STALE") {
      state.sessionId = null;
      setSessionStatus(false);
      $("sessionId").textContent = "—";
      $("conversationId").textContent = "—";
      toast("Repository changed. Start a new session against the rebuilt runtime.", true);
      return;
    }

    toast(error.message, true);
  }
}

async function clearSession() {
  if (!state.sessionId) return;

  setBusy(true, "Clearing investigation context...");

  try {
    const session = await api("/sessions/clear", {
      method:"POST",
      body:JSON.stringify({
        repository_root:state.repositoryRoot,
        session_id:state.sessionId,
      }),
    });

    renderState(session);
    clearMessages("Investigation context cleared. Conversation identity was preserved.");
    setBusy(false);
  } catch (error) {
    setBusy(false);
    toast(error.message, true);
  }
}

async function newConversation() {
  if (!state.sessionId) return;

  setBusy(true, "Starting a new conversation...");

  try {
    const session = await api("/sessions/new", {
      method:"POST",
      body:JSON.stringify({
        repository_root:state.repositoryRoot,
        session_id:state.sessionId,
      }),
    });

    state.conversationId = session.conversation_id;
    $("conversationId").textContent = shortId(session.conversation_id);
    renderState(session);
    clearMessages("New conversation started in the same application session.");
    setBusy(false);
  } catch (error) {
    setBusy(false);
    toast(error.message, true);
  }
}

async function closeRepository() {
  if (!state.repositoryRoot) return;

  setBusy(true, "Closing repository runtime...");

  try {
    await api("/repositories/close", {
      method:"POST",
      body:JSON.stringify({repository_root:state.repositoryRoot}),
    });

    state.repositoryRoot = null;
    state.runtimeId = null;
    state.sessionId = null;
    state.conversationId = null;

    setRepositoryStatus(false);
    setSessionStatus(false);

    ["runtimeId","fingerprint","files","symbols","relationships","graph","embedding","device","llm","sessionId","conversationId"].forEach(id => {
      $(id).textContent = "—";
    });

    renderState(null);
    clearMessages("Repository closed. Open another repository to continue.");
    setBusy(false);
    await refreshHealth();
  } catch (error) {
    setBusy(false);
    toast(error.message, true);
  }
}

$("openRepoBtn").addEventListener("click", openRepository);
$("newSessionBtn").addEventListener("click", createSession);
$("askBtn").addEventListener("click", askQuestion);
$("clearBtn").addEventListener("click", clearSession);
$("newConversationBtn").addEventListener("click", newConversation);
$("closeRepoBtn").addEventListener("click", closeRepository);
$("refreshBtn").addEventListener("click", async () => {
  await refreshHealth();
  await loadRepositoryDiagnostics();
});

$("repoPath").addEventListener("keydown", event => {
  if (event.key === "Enter") openRepository();
});

$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});

refreshHealth();
setInterval(refreshHealth, 5000);
</script>
</body>
</html>
"""


def dashboard_response() -> HTMLResponse:
    return HTMLResponse(
        content=DASHBOARD_HTML,
        status_code=200,
    )
