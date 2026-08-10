// rag UI - vanilla JS, no framework.
//
// Tabs: Ask | Ingest | Library (P2).
//   Ask:     chat-style ask with topic filter + inline [n] citations.
//   Ingest:  file picker + drag-drop. Uploads to /api/ingest.
//   Library: read-only view of sources / citations / eval runs / stats.

const $ = (id) => document.getElementById(id);

// ---- state ----------------------------------------------------------------

const state = {
  files: [],            // queued File objects for ingest
  currentSessionId: null,  // active session ID, or null for stateless
};

// ---- markdown renderer (minimal, no deps) --------------------------------
//
// Handles: **bold**, *italic*, `code`, ```fenced```, bullet lists, [n]
// citation markers (kept as anchors for click-to-source).
// Doesn't try to be CommonMark - just good enough for the LLM's output.

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

function renderInline(s) {
  s = s.replace(/`([^`]+)`/g, (_, code) => `<code>${escapeHtml(code)}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, (_, t) => `<strong>${t}</strong>`);
  s = s.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, (_, t) => `<em>${t}</em>`);
  s = s.replace(/\[(\d+)\]/g, (_, n) => `<a class="cite" data-n="${n}" href="#cite-${n}">[${n}]</a>`);
  return s;
}

function renderMarkdown(text) {
  const parts = text.split(/(```[\s\S]*?```)/g);
  const out = [];
  for (const part of parts) {
    if (part.startsWith("```") && part.endsWith("```")) {
      const inner = part.slice(3, -3).replace(/^.*\n/, "");
      out.push(`<pre><code>${escapeHtml(inner)}</code></pre>`);
    } else {
      const lines = part.split("\n");
      let buf = [];
      let inList = false;
      const flush = () => {
        if (!buf.length) return;
        out.push(`<p>${renderInline(buf.join("<br>"))}</p>`);
        buf = [];
      };
      for (const line of lines) {
        const m = line.match(/^[-*]\s+(.*)/);
        if (m) {
          if (!inList) { flush(); inList = true; out.push("<ul>"); }
          out.push(`<li>${renderInline(m[1])}</li>`);
        } else {
          if (inList) { out.push("</ul>"); inList = false; }
          buf.push(line);
        }
      }
      if (inList) out.push("</ul>");
      flush();
    }
  }
  return out.join("");
}

// ---- chat -----------------------------------------------------------------

function addMessage(role, bodyHtml, citations = []) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="role">${role}</div><div class="body">${bodyHtml}</div>`;
  if (citations.length) {
    const list = document.createElement("div");
    list.className = "cite-list";
    for (const c of citations) {
      const item = document.createElement("div");
      item.className = "cite-item";
      item.id = `cite-${c.n}`;
      const head = document.createElement("div");
      head.className = "cite-head";
      head.innerHTML = `<strong>[${c.n}]</strong> ${escapeHtml(c.source_path)} :: ${escapeHtml(c.section)} <span class="score">score=${c.score}</span>`;
      const text = document.createElement("div");
      text.className = "cite-text";
      text.textContent = c.text || "";
      text.addEventListener("click", () => text.classList.toggle("expanded"));
      item.appendChild(head);
      item.appendChild(text);
      list.appendChild(item);
    }
    div.appendChild(list);
  }
  const empty = $("chat").querySelector(".empty");
  if (empty) empty.remove();
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

async function ask(query, topic, sessionId) {
  const body = { query };
  if (topic) body.topic = topic;
  if (sessionId) body.session_id = sessionId;
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`ask failed: ${res.status} ${await res.text()}`);
  return res.json();
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("query").value.trim();
  if (!q) return;
  addMessage("user", escapeHtml(q));
  $("query").value = "";
  const thinking = addMessage("assistant", '<span class="thinking">thinking...</span>');
  try {
    const result = await ask(q, $("topic").value, state.currentSessionId);
    thinking.querySelector(".body").innerHTML = renderMarkdown(result.answer);
    const list = document.createElement("div");
    list.className = "cite-list";
    for (const c of result.citations) {
      const item = document.createElement("div");
      item.className = "cite-item";
      item.id = `cite-${c.n}`;
      const head = document.createElement("div");
      head.className = "cite-head";
      head.innerHTML = `<strong>[${c.n}]</strong> ${escapeHtml(c.source_path)} :: ${escapeHtml(c.section)} <span class="score">score=${c.score}</span>`;
      const text = document.createElement("div");
      text.className = "cite-text";
      text.textContent = c.text || "";
      text.addEventListener("click", () => text.classList.toggle("expanded"));
      item.appendChild(head);
      item.appendChild(text);
      list.appendChild(item);
    }
    thinking.appendChild(list);
    $("chat").scrollTop = $("chat").scrollHeight;
  } catch (err) {
    thinking.querySelector(".body").innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
  }
});

// ---- sessions (P2: conversation memory) -------------------------------------

// Render a list of historical turns into the chat pane.
function renderTurns(turns) {
  $("chat").innerHTML = "";
  if (!turns || turns.length === 0) {
    $("chat").innerHTML =
      '<div class="empty"><p>Ask a question. Citations will appear below the answer.</p>' +
      '<p class="hint">This session is empty — go ahead and chat.</p></div>';
    return;
  }
  for (const t of turns) {
    // citations may be stored as a JSON string or already-parsed array.
    let cites = [];
    if (t.citations) {
      if (typeof t.citations === "string") {
        try { cites = JSON.parse(t.citations); } catch (_) { cites = []; }
      } else {
        cites = t.citations;
      }
    }
    // Normalise citation shape for addMessage.
    const normCites = (cites || []).map((c) => ({
      n: c.n ?? c.rank ?? 0,
      source_path: c.source_path ?? "",
      section: c.section ?? "",
      text: c.text ?? "",
      score: c.score ?? 0,
    }));
    addMessage("user", escapeHtml(t.query || ""));
    addMessage("assistant", renderMarkdown(t.answer || ""), normCites);
  }
  $("chat").scrollTop = $("chat").scrollHeight;
}

// Load all turns for a session and display them.
async function loadTurns(sessionId) {
  try {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`);
    if (!res.ok) throw new Error(`${res.status}`);
    const turns = await res.json();
    renderTurns(turns);
  } catch (err) {
    // Non-fatal: just show an empty chat.
    renderTurns([]);
    console.warn("loadTurns:", err.message);
  }
}

// Populate the session dropdown from /api/sessions.
async function loadSessions() {
  try {
    const res = await fetch("/api/sessions?limit=30");
    if (!res.ok) return;
    const sessions = await res.json();
    const sel = $("sessionSelect");
    // Remember the current selection.
    const prev = sel.value;
    // Wipe all options except the placeholder.
    while (sel.options.length > 1) sel.remove(1);
    for (const s of sessions) {
      const opt = document.createElement("option");
      opt.value = s.id;
      const ts = (s.updated_at || "").replace("T", " ").slice(0, 16);
      const title = s.title || "(untitled)";
      opt.textContent = `${ts}  ${title}  (${s.turn_count} turn${s.turn_count !== 1 ? "s" : ""})`;
      sel.appendChild(opt);
    }
    // Restore selection if still valid.
    if (prev && [...sel.options].some((o) => o.value === prev)) {
      sel.value = prev;
    }
  } catch (_) { /* best-effort */ }
}

// Start a brand-new session: generate a UUID, create it via the API, and
// activate it (clearing the chat for a fresh conversation).
async function newSession() {
  const id = (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36));
  try {
    await fetch(`/api/sessions/${encodeURIComponent(id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
  } catch (_) { /* best-effort — the server will auto-create on record_turn anyway */ }
  state.currentSessionId = id;
  $("sessionSelect").value = id;   // won't match any option, but signals "active"
  renderTurns([]);
  await loadSessions();   // repopulate so the new session appears in the list
  // Select the new session in the dropdown (it should now be first after reload).
  $("sessionSelect").value = id;
}

// Switch to an existing session: set it as current and restore its turns.
async function switchToSession(sessionId) {
  if (!sessionId) {
    state.currentSessionId = null;
    renderTurns([]);
    return;
  }
  state.currentSessionId = sessionId;
  await loadTurns(sessionId);
}

$("newSessionBtn").addEventListener("click", () => newSession());

$("sessionSelect").addEventListener("change", () => {
  switchToSession($("sessionSelect").value);
});

$("clearBtn").addEventListener("click", () => {
  if (state.currentSessionId) {
    // Clear just the chat pane, stay in the current session.
    renderTurns([]);
  } else {
    $("chat").innerHTML =
      '<div class="empty"><p>Ask a question. Citations will appear below the answer.</p></div>';
  }
});

// ---- eval panel -----------------------------------------------------------

$("evalBtn").addEventListener("click", async () => {
  $("evalPanel").classList.remove("hidden");
  $("evalOutput").textContent = "running...";
  try {
    const res = await fetch("/api/eval");
    if (!res.ok) throw new Error(`eval failed: ${res.status}`);
    const m = await res.json();
    let out = `recall@dense : ${m.recall_at_dense ?? "-"}\nrecall@5     : ${m.recall_at_5}\nMRR          : ${m.mrr}\n`;
    if (m.faithfulness_proxy !== undefined) out += `faithfulness  : ${m.faithfulness_proxy}\n`;
    out += `\nper-question (${m.n_questions}):\n`;
    for (const r of m.per_question) {
      out += `  ${r.recall_at_5 >= 1 ? "OK" : r.recall_at_5 > 0 ? "~" : "X"} ${r.question}\n`;
    }
    $("evalOutput").textContent = out;
  } catch (err) {
    $("evalOutput").textContent = `error: ${err.message}`;
  }
});

$("evalClose").addEventListener("click", () => $("evalPanel").classList.add("hidden"));

// ---- tabs (P2) ------------------------------------------------------------

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    $(`tab-${target}`).classList.add("active");
    if (target === "library") refreshLibrary();
  });
});

// ---- ingest (P2) ----------------------------------------------------------

const dropzone = $("dropzone");
const filePicker = $("filePicker");
const ingestList = $("ingestList");
const ingestBtn = $("ingestBtn");
const ingestClear = $("ingestClear");
const ingestStatus = $("ingestStatus");
const ingestResult = $("ingestResult");

function humanSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function addFiles(files) {
  // Filter to supported extensions.
  const allowed = new Set([".pdf", ".md", ".markdown", ".py", ".epub"]);
  for (const f of files) {
    const name = (f.name || "").toLowerCase();
    const dot = name.lastIndexOf(".");
    const ext = dot >= 0 ? name.slice(dot) : "";
    if (!allowed.has(ext)) continue;
    // De-dupe by name + size.
    if (state.files.some((g) => g.name === f.name && g.size === f.size)) continue;
    state.files.push(f);
  }
  renderIngestList();
}

function removeFile(idx) {
  state.files.splice(idx, 1);
  renderIngestList();
}

function renderIngestList() {
  ingestList.innerHTML = "";
  state.files.forEach((f, idx) => {
    const row = document.createElement("div");
    row.className = "ingest-item";
    row.innerHTML = `
      <span class="name">${escapeHtml(f.name)}</span>
      <span class="size">${humanSize(f.size)}</span>
      <button class="remove" title="remove">&times;</button>
    `;
    row.querySelector(".remove").addEventListener("click", () => removeFile(idx));
    ingestList.appendChild(row);
  });
  ingestBtn.disabled = state.files.length === 0;
  ingestClear.disabled = state.files.length === 0;
  ingestStatus.textContent = state.files.length
    ? `${state.files.length} file(s) queued`
    : "";
}

function clearFiles() {
  state.files = [];
  ingestResult.className = "ingest-result";
  ingestResult.textContent = "";
  renderIngestList();
}

dropzone.addEventListener("click", () => filePicker.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    filePicker.click();
  }
});
filePicker.addEventListener("change", () => {
  if (filePicker.files && filePicker.files.length) {
    addFiles(Array.from(filePicker.files));
    filePicker.value = "";  // allow re-picking the same file
  }
});
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
    addFiles(Array.from(e.dataTransfer.files));
  }
});

ingestClear.addEventListener("click", clearFiles);

ingestBtn.addEventListener("click", async () => {
  if (state.files.length === 0) return;
  ingestBtn.disabled = true;
  ingestClear.disabled = true;
  ingestStatus.textContent = "uploading + embedding (this may take a while)...";
  ingestResult.className = "ingest-result";
  ingestResult.textContent = "";
  const fd = new FormData();
  for (const f of state.files) fd.append("files", f, f.name);
  try {
    const res = await fetch("/api/ingest", { method: "POST", body: fd });
    const j = await res.json();
    if (!res.ok) {
      ingestResult.className = "ingest-result err";
      ingestResult.textContent = `error: ${j.detail || res.status}`;
    } else {
      ingestResult.className = "ingest-result ok";
      const skipped = (j.skipped || []).length
        ? `\nskipped (unsupported extension or read failure): ${(j.skipped || []).join(", ")}`
        : "";
      ingestResult.textContent =
        `ingested ${j.ingested} file(s) -> ${j.chunks} chunk(s) in the index.\n` +
        `types: ${(j.types || []).join(", ") || "-"}${skipped}`;
      // Clear the queue on success.
      state.files = [];
      renderIngestList();
      // Refresh topics so the Ask dropdown picks up new content.
      loadTopics();
    }
  } catch (err) {
    ingestResult.className = "ingest-result err";
    ingestResult.textContent = `error: ${err.message}`;
  } finally {
    ingestBtn.disabled = state.files.length === 0;
    ingestClear.disabled = state.files.length === 0;
    ingestStatus.textContent = state.files.length
      ? `${state.files.length} file(s) queued`
      : "done";
  }
});

// ---- library (P2) ---------------------------------------------------------

async function refreshLibrary() {
  $("libraryStats").textContent = "loading...";
  $("librarySources").textContent = "loading...";
  $("libraryCitations").textContent = "loading...";
  $("libraryEval").textContent = "loading...";
  try {
    const [statsRes, sourcesRes, citationsRes, evalRes] = await Promise.all([
      fetch("/api/stats").then((r) => r.ok ? r.json() : null),
      fetch("/api/sources").then((r) => r.ok ? r.json() : []),
      fetch("/api/citations").then((r) => r.ok ? r.json() : []),
      fetch("/api/eval-runs").then((r) => r.ok ? r.json() : []),
    ]);
    renderStats(statsRes);
    renderSources(sourcesRes);
    renderCitations(citationsRes);
    renderEvalRuns(evalRes);
  } catch (err) {
    $("libraryStats").textContent = `error: ${err.message}`;
  }
}

function renderStats(s) {
  if (!s) { $("libraryStats").textContent = "(no metadata DB)"; return; }
  $("libraryStats").innerHTML =
    `<span class="num">${s.total_sources ?? 0}</span> sources  ` +
    `<span class="num">${s.total_citations ?? 0}</span> citations  ` +
    `<span class="num">${s.total_eval_runs ?? 0}</span> eval runs`;
}

function renderSources(rows) {
  if (!rows.length) {
    $("librarySources").innerHTML = '<div class="library-row empty">(no sources yet)</div>';
    return;
  }
  $("librarySources").innerHTML = rows.map((r) => {
    const ts = (r.ingested_at || "").replace("T", " ").slice(0, 19);
    const topic = r.topic || "-";
    return `<div class="library-row">
      <span class="ts">${escapeHtml(ts)}</span>
      <span class="topic">${escapeHtml(topic)}</span>
      <span>chunks=${r.chunk_count}</span>
      <span>${escapeHtml(r.source_path)}</span>
    </div>`;
  }).join("");
}

function renderCitations(rows) {
  if (!rows.length) {
    $("libraryCitations").innerHTML = '<div class="library-row empty">(no citations yet)</div>';
    return;
  }
  $("libraryCitations").innerHTML = rows.map((r) => {
    const ts = (r.asked_at || "").replace("T", " ").slice(0, 19);
    const q = r.query || "";
    const qShort = q.length > 50 ? q.slice(0, 47) + "..." : q;
    return `<div class="library-row">
      <span class="ts">${escapeHtml(ts)}</span>
      <span>rank=${r.rank}</span>
      <span>chunk=${(r.chunk_id || "").slice(0, 8)}</span>
      <span>q=${escapeHtml(qShort)}</span>
    </div>`;
  }).join("");
}

function renderEvalRuns(rows) {
  if (!rows.length) {
    $("libraryEval").innerHTML = '<div class="library-row empty">(no eval runs yet)</div>';
    return;
  }
  $("libraryEval").innerHTML = rows.map((r) => {
    const ts = (r.ran_at || "").replace("T", " ").slice(0, 19);
    const rd = r.recall_at_dense != null ? ` recall@dense=${r.recall_at_dense}` : "";
    return `<div class="library-row">
      <span class="ts">${escapeHtml(ts)}</span>
      <span>n=${r.n_questions}</span>
      <span>recall@5=${r.recall_at_5}</span>
      <span>mrr=${r.mrr}</span>
      <span>${rd}</span>
    </div>`;
  }).join("");
}

// ---- health + topics on load ---------------------------------------------

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("not ready");
    const j = await res.json();
    if (j.ready) {
      $("statusDot").className = "status-dot ready";
      $("statusText").textContent = `ready - ${j.url}`;
    } else {
      $("statusDot").className = "status-dot error";
      $("statusText").textContent = "not ready";
    }
  } catch {
    $("statusDot").className = "status-dot error";
    $("statusText").textContent = "server unreachable";
  }
}

async function loadTopics() {
  try {
    const res = await fetch("/api/topics");
    if (!res.ok) return;
    const topics = await res.json();
    const sel = $("topic");
    // Wipe any existing options except the "all" placeholder.
    while (sel.options.length > 1) sel.remove(1);
    for (const t of topics) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      sel.appendChild(opt);
    }
  } catch { /* fine - no topics if collection is empty */ }
}

checkHealth();
loadTopics();
loadSessions();
setInterval(checkHealth, 30000);
