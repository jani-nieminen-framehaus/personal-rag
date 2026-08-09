// rag UI — vanilla JS, no framework. ~150 lines.

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const form = $("form");
const queryInput = $("query");
const topicSelect = $("topic");
const statusDot = $("statusDot");
const statusText = $("statusText");
const evalPanel = $("evalPanel");
const evalOutput = $("evalOutput");

// -- markdown renderer (minimal, no deps) ----------------------------------
//
// Handles: **bold**, *italic*, `code`, ```fenced```, bullet lists, [n]
// citation markers (kept as anchors for click-to-source).
// Doesn't try to be CommonMark — just good enough for the LLM's output.

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

function renderInline(s) {
  // code first (so we don't bold inside code)
  s = s.replace(/`([^`]+)`/g, (_, code) => `<code>${escapeHtml(code)}</code>`);
  // bold
  s = s.replace(/\*\*([^*]+)\*\*/g, (_, t) => `<strong>${t}</strong>`);
  // italic
  s = s.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, (_, t) => `<em>${t}</em>`);
  // citation marker [n] -> clickable anchor
  s = s.replace(/\[(\d+)\]/g, (_, n) => `<a class="cite" data-n="${n}" href="#cite-${n}">[${n}]</a>`);
  return s;
}

function renderMarkdown(text) {
  // Split on fenced code blocks first.
  const parts = text.split(/(```[\s\S]*?```)/g);
  const out = [];
  for (const part of parts) {
    if (part.startsWith("```") && part.endsWith("```")) {
      const inner = part.slice(3, -3).replace(/^.*\n/, "");  // drop language tag line
      out.push(`<pre><code>${escapeHtml(inner)}</code></pre>`);
    } else {
      // paragraphs and bullet lists
      const lines = part.split("\n");
      let buf = [];
      let inList = false;
      const flush = () => {
        if (!buf.length) return;
        const html = buf.join("<br>");
        out.push(`<p>${renderInline(html)}</p>`);
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

// -- chat state -----------------------------------------------------------

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
  // replace empty placeholder
  const empty = chat.querySelector(".empty");
  if (empty) empty.remove();
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addError(msg) {
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = msg;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// -- ask ------------------------------------------------------------------

async function ask(query, topic) {
  const body = { query };
  if (topic) body.topic = topic;
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`ask failed: ${res.status} ${t}`);
  }
  return res.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;
  addMessage("user", escapeHtml(q));
  queryInput.value = "";
  const thinking = addMessage("assistant", '<span class="thinking">thinking…</span>');
  try {
    const result = await ask(q, topicSelect.value);
    thinking.querySelector(".body").innerHTML = renderMarkdown(result.answer);
    // attach citations to the same message
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
    chat.scrollTop = chat.scrollHeight;
  } catch (err) {
    thinking.querySelector(".body").innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
  }
});

// -- clear / eval --------------------------------------------------------

$("clearBtn").addEventListener("click", () => {
  chat.innerHTML = '<div class="empty"><p>Ask a question. Citations will appear below the answer.</p></div>';
});

$("evalBtn").addEventListener("click", async () => {
  evalPanel.classList.remove("hidden");
  evalOutput.textContent = "running…";
  try {
    const res = await fetch("/api/eval");
    if (!res.ok) throw new Error(`eval failed: ${res.status}`);
    const m = await res.json();
    let out = `recall@dense : ${m.recall_at_dense ?? "-"}\nrecall@5     : ${m.recall_at_5}\nMRR          : ${m.mrr}\n`;
    if (m.faithfulness_proxy !== undefined) out += `faithfulness  : ${m.faithfulness_proxy}\n`;
    out += `\nper-question (${m.n_questions}):\n`;
    for (const r of m.per_question) {
      out += `  ${r.recall_at_5 >= 1 ? "✓" : r.recall_at_5 > 0 ? "·" : "✗"} ${r.question}\n`;
    }
    evalOutput.textContent = out;
  } catch (err) {
    evalOutput.textContent = `error: ${err.message}`;
  }
});

$("evalClose").addEventListener("click", () => evalPanel.classList.add("hidden"));

// -- health + topics on load ---------------------------------------------

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("not ready");
    const j = await res.json();
    if (j.ready) {
      statusDot.className = "status-dot ready";
      statusText.textContent = `ready · ${j.url}`;
    } else {
      statusDot.className = "status-dot error";
      statusText.textContent = "not ready";
    }
  } catch {
    statusDot.className = "status-dot error";
    statusText.textContent = "server unreachable";
  }
}

async function loadTopics() {
  try {
    const res = await fetch("/api/topics");
    if (!res.ok) return;
    const topics = await res.json();
    for (const t of topics) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      topicSelect.appendChild(opt);
    }
  } catch { /* fine — no topics if collection is empty */ }
}

checkHealth();
loadTopics();
setInterval(checkHealth, 30_000);  // refresh status every 30s
