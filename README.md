# rag — personal, multi-topic RAG (P0 walking skeleton)

> A terminal-first RAG CLI for your own knowledge base. Plain Python 3.11+,
> no LangChain / LlamaIndex. Every model is swappable from `config.yaml`.
> Built for Windows 11 + Ollama on a 48 GB VRAM rig. No WSL2, no Docker
> Desktop, no GUI.

```
$ rag ask "What was YaRN about?"

YaRN is a method to extend the context length of an existing LLM without
expensive continued pre-training. It combines NTK-aware scaling of RoPE
frequencies with an attention-temperature adjustment, so a pretrained
model can be extended from 4k to 64k–128k context with minimal
fine-tuning [1][2].

--- citations ---
1. samples/notes/ml/yarn.md :: YaRN — Yet another RoPE extensioN  (topic=ml, score=0.871, chunk=b6a5ab0b…)
2. samples/notes/ml/yarn.md :: The idea  (topic=ml, score=0.843, chunk=079aa336…)
```

---

## What's in the box

- `cli.py` — `rag ask`, `rag ingest`, `rag eval` (Click-based, terminal-only)
- `core/` — interfaces, structure-aware chunker, pipeline orchestrator
- `providers/` — Qwen3-Embedding-8B embedder, Ollama generator, BGE-reranker stub
- `ingest/` — Markdown directory + Zeal docset SQLite readers
- `store/qdrant_store.py` — dense Qdrant collection with `sparse` slot reserved for P1
- `eval/` — golden set + custom recall@5 / MRR harness (`run_ragas.py`, no Ragas framework)
- `samples/notes/` — 5 demo notes so the whole loop runs end-to-end on day one
- `config.yaml` — **the only place model names live**

Design notes in [`PLAN.md`](./PLAN.md).

---

## Quick start (TL;DR)

```powershell
# 1. Ollama (generator) — install from https://ollama.com/download
ollama pull qwen3:30b-a3b                       # ~20 GB one-time

# 2. Qdrant (vector store) — see full setup for the two install options
#    Quick path: download the Windows binary, run it, leave it on :6333

# 3. The CLI
cd D:\Tinkering sideprojects\rag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt                  # ~6 GB on first run (torch + Qwen3)
python cli.py ingest --markdown .\samples\notes
python cli.py ask "What was YaRN about?"
python cli.py eval
```

---

## Full setup — Windows 11

Three services to install, all native Windows. No WSL2, no Docker Desktop.

### 1. Python 3.11+

If you don't have it already:

```powershell
# from the Microsoft Store, or winget:
winget install Python.Python.3.12
# restart terminal so `python` resolves
python --version    # → Python 3.11.x or 3.12.x
```

### 2. Ollama

The Windows installer handles GPU passthrough correctly on your 48 GB rig.

```powershell
# download from https://ollama.com/download (Windows installer)
# then in a normal terminal:
ollama pull qwen3:30b-a3b        # ~20 GB, one-time
ollama list                       # confirm it's there
```

The `LocalGenerator` defaults to `http://localhost:11434/v1` — Ollama
serves the OpenAI-compatible chat completions API on that URL natively.

### 3. Qdrant (vector store)

Two install options. Pick whichever fits your setup.

**Option A — Windows binary (recommended — no Docker at all):**

```powershell
# download from Qdrant's GitHub releases
Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip" -OutFile "$env:TEMP\qdrant.zip"
Expand-Archive "$env:TEMP\qdrant.zip" -DestinationPath "C:\Tools\qdrant"

# data dir
New-Item -ItemType Directory -Force -Path "C:\qdrant\storage" | Out-Null

# run it
& "C:\Tools\qdrant\qdrant.exe" --storage-snapshots-dir C:\qdrant\storage

# verify
curl http://localhost:6333/collections
# → {"result":{"collections":[]}}
```

To make Qdrant start on boot, wrap the above in a scheduled task, a
NSSM service, or just stick it in your shell startup folder.

**Option B — Docker (only if you already have Docker Desktop or the
Docker CLI; we don't require it):**

```powershell
docker run -d --name qdrant -p 6333:6333 `
  -v C:\qdrant\storage:/qdrant/storage `
  --restart unless-stopped `
  qdrant/qdrant
```

From either option, `http://localhost:6333/collections` should return
`{"result":{"collections":[]}}`.

> **This machine: port 7333, not 6333.** IPv4 `:6333` is squatted by a
> bound-but-never-listening ghost socket (held by an unrelated local agent
> process — netstat-invisible; find it with
> `Get-NetTCPConnection -LocalPort 6333`). Qdrant here runs with
> `QDRANT__SERVICE__HTTP_PORT=7333` (see `C:\Tools\qdrant\run-qdrant.cmd`),
> and `config.yaml → store.url` matches. Substitute 7333 in every
> `localhost:6333` URL on this page.

### 4. The Python venv (Windows side)

```powershell
cd D:\Tinkering sideprojects\rag
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

First install pulls down PyTorch + CUDA runtime + sentence-transformers +
Qwen3-Embedding-8B (~5 GB on first `rag ask`). Subsequent runs are instant
(HF cache + Qdrant state).

> If `bitsandbytes` fails to install on Windows, you can:
> - swap `embedder.quant: q4` → `none` in `config.yaml` (uses full precision),
> - or pin `bitsandbytes` to the latest Windows-compatible release.

### 5. The shiv launcher (recommended)

Skip the `python -m venv` + `activate` + `cd` dance. There's a Windows shiv
at the repo root that does all of it:

- `rag.bat` — cmd.exe compatible, double-clickable
- `rag.ps1` — PowerShell version with colored output

First run creates the venv + installs `requirements.txt`. After that, every
run just `exec`s `cli.py`:

```powershell
# from anywhere on the machine (after adding the repo to PATH or
# pinning rag.bat to your taskbar):
rag ask "What was YaRN about?"
rag ingest --markdown D:\notes
rag eval
```

If you want the literal `rag.exe` feel: create a Windows shortcut to
`rag.bat`, rename the shortcut to `rag.exe`, and pin it. Windows runs
.bat files transparently when launched via a shortcut.

### 6. First ingest (uses the demo notes)

```powershell
python cli.py ingest --markdown .\samples\notes
# → "done. wrote N chunks into kb_p0."
```

You should see ~20 chunks. Open the Qdrant dashboard at
`http://localhost:6333/dashboard` to inspect them.

### 7. First query

```powershell
python cli.py ask "What was YaRN about?"
```

Expected: an answer with `[1] [2] …` inline citations and a footer
listing each cited source. First run downloads the embedding model.

### 8. Eval

```powershell
python cli.py eval
# → "recall@5 = 0.89  MRR = 0.92" (or close, depending on embedding model)
```

The harness reads `eval/golden_set.jsonl` and prints per-question
results so you can see *which* questions failed and fix them by tweaking
the chunker, the embedding model, or the query.

---

## Evaluation — what it's actually for

Skip this section until you have a real corpus in the index. None of it
is required to use the tool: ingest, ask, read answers. That works.

**The problem it solves.** When retrieval misses, you don't find out. The
model still writes a fluent, confident answer from whatever chunks it did
get — it does not say "I couldn't find the relevant passage." You only
notice if you already knew the answer, and if you knew, you wouldn't be
asking. Everything below exists to catch that one failure, because it is
the only failure you cannot spot by reading the output.

Quick manual check, no tooling needed: look at the citation footer. If it
is empty or cites files unrelated to your question, retrieval missed and
the confident tone was the model bluffing. If it cites the right file but
the answer is still mush, retrieval worked and the *answer* is the
problem — that is what the faithfulness score measures, separately.

**The golden set is a ruler, not a quiz.** It is a list of questions where
the correct source chunk is already known, so software can ask "did the
right chunk come back?" and give you a number. You never read it.

**You do not write the questions.** `rag golden generate` samples chunks
from your own index and has the local model draft a question that each
chunk answers. `rag golden review` then shows you each one with a preview
of its source chunk, and you press `y` / `n` / `e` (edit) / `q` (quit,
resume later). The correct answer is known by construction — the question
was generated *from* that chunk. Twenty minutes gets you fifty.

**"Knobs" are four settings in `config.yaml`:** how many candidates to
fetch before narrowing to the final few (`top_k_dense`), the balance
between meaning-matching and exact-keyword-matching (`dense_weight`),
whether the reranker earns its latency, and chunk size. There is no
universally right answer — a docset full of exact function names wants
different weighting than prose notes, and a mixed corpus is an empirical
question. `rag eval --sweep` tries the combinations, scores each against
the golden set, and prints a table best-first. The top row is what you
copy into `config.yaml`.

### When to run what

| Command | When | Cost |
| --- | --- | --- |
| `rag golden generate` + `review` | after adding a meaningful batch of new material | ~10 min of your attention |
| `rag eval` | health check, or after changing anything | seconds |
| `rag eval --sweep` | only when the corpus changes *character* | slow, unattended |

Most of the time you run none of them.

### Two things that will bite you if nobody says them

**A recall number is only meaningful against a fixed corpus.** Retrieval
gets harder as the index grows — more near-duplicates competing for the
same few slots — so the same settings score *lower* on 1000 notes than on
10. That is a harder exam, not a regression. Compare setting A against
setting B on the same corpus on the same afternoon; do not read the
numbers as a trend line across months.

**So do not tune on a toy corpus.** Ingest the bulk of what you actually
have first, then spend the twenty minutes. Tuning on ten notes optimises
for a corpus you do not own. The question set is designed to grow with
you: `rag golden generate` remembers which chunks it has already drafted
from and skips them, so re-running it after an ingest drafts only from the
new material.

Full operator sequence: `docs/superpowers/specs/2026-08-10-p3-runbook.md`.

---

## Ingesting your own notes

```powershell
# any folder of .md files, with optional YAML frontmatter:
#
#   ---
#   topic: photography
#   ---
#   # Heading
#   ...
#
python cli.py ingest --markdown D:\path\to\your\notes

# topic resolution order (see ingest/markdown_dir.py):
#   1. frontmatter `topic:` field
#   2. parent directory name (relative to the ingester root)
#   3. config.ingest.default_topic
```

The ingest is **idempotent** — re-running updates chunks in place thanks
to deterministic UUID5 ids. No duplicates, no manual cleanup.

## Keeping the index current

`rag ingest` is for one-offs. `rag refresh` is for "the index should
match what's on disk, and I don't want to think about it."

**Tell it what the index is supposed to contain.** Nothing is tracked
until it's listed under `sources:` in `config.yaml` (it ships empty):

```yaml
sources:
  - type: markdown          # markdown | pdf | epub | zeal
    path: D:/notes
  - type: pdf
    path: D:/Library
  - type: zeal
    path: C:/Users/you/AppData/Local/Zeal/Zeal/docsets/Python.docset
```

Then:

```powershell
rag refresh              # re-ingest only what changed
rag refresh --dry-run    # say what you WOULD do, write nothing
rag refresh --prune      # ...and drop index entries whose file is gone
```

Refresh hashes every file and compares it to what it ingested last time,
so an unchanged corpus costs a directory walk and one SHA-256 per file —
**no embedding model is loaded at all**. Editing three notes re-embeds
three notes, not the library. A Zeal docset is all-or-nothing (its
`.dsidx` index is the hash), because there's no cheap way to tell which
of ~50,000 pages moved.

### What deletes, and what doesn't

**`rag refresh --prune` and `rag forget` are the only two commands in
this system that delete anything.** Everything else only ever adds or
overwrites. Both ask for confirmation first (`-y` skips it, for
scripts).

```powershell
rag refresh --dry-run --prune   # the paths it would remove, before you agree
rag forget --source D:\notes\old.md    # one file, by the path `rag sources` shows
rag forget --topic photography         # everything filed under a topic
```

`--dry-run` is the honest preview: it's the same code path as a real run
with the last step removed, so what it prints is what would happen.

### Why an unplugged drive can't wipe your index

This runs unattended, which means it will eventually fire while a USB
disk is unplugged or a network share is unmounted. A naive
implementation enumerates zero files, concludes the corpus was deleted,
and `--prune` takes it out.

So **a source root that isn't there is treated as "unknown", never as
"empty"** — planned, reported, and skipped, with nothing under it
ingested or pruned. Same for a root that's present but enumerates zero
files while the index holds rows for it, and same for an unreachable
subtree below a healthy root. `rag refresh` exits non-zero when this
happens so you find out. If you emptied a folder *on purpose*, that's
what `rag forget` is for — refusing to guess is the point.

### Doing it on a schedule

```powershell
.\scripts\install-refresh-task.ps1              # daily at 03:00
.\scripts\install-refresh-task.ps1 -At '23:30'  # or whenever
# remove it again:
.\scripts\uninstall-refresh-task.ps1
```

Registers a Task Scheduler task (`rag-refresh`) that runs plain
`rag refresh` — **never `--prune`**; unattended is the wrong mode for
the only irreversible command here. Re-running the installer just
re-registers, so changing `-At` is safe. It runs unelevated, when you're
logged in; a run missed because the machine was off happens once it's
back.

Since it never prunes, files you delete from disk stay in the index
until you run `rag refresh --prune` by hand. Refresh tells you the count
each time it notices. Check on the task with:

```powershell
Get-ScheduledTaskInfo -TaskName rag-refresh   # LastTaskResult 0 = clean run
```

## Ingesting books (PDF)

Drop a single PDF or a folder of PDFs at the CLI:

```powershell
# One book
python cli.py ingest --pdf D:\Books\mybook.pdf

# A whole library, topics from the parent dir
python cli.py ingest --pdf D:\Library
#   D:\Library\photography\book.pdf  -> topic = "photography"
#   D:\Library\code\patterns.pdf     -> topic = "code"
```

Mechanics (see `ingest/pdf_dir.py`):

- One chunk per page; long pages are split by token count using the
  same sliding-window strategy as the markdown chunker.
- Section is `Page N` (or the PDF's own page label if it has one —
  books sometimes use roman numerals or chapter prefixes).
- Topic: from the immediate parent dir (single-file ingest falls
  back to `default_topic`).
- `doc_type` is `"pdf"`; the chunk's `extra` payload includes the
  page number and label.
- Pages with no text layer (scans) are skipped with a debug log.
  OCR is out of P1 scope; add Tesseract later if you need it.
- Corrupt PDFs in a directory log a warning and are skipped — one
  bad file doesn't kill the rest of the ingest.

## The metadata database

Every ingest and every `rag ask` writes a small side-channel to a
SQLite file at `metadata.sqlite3` (override via `metadata.path` in
`config.yaml`). Three tables:

```sql
CREATE TABLE sources (
    source_path   TEXT PRIMARY KEY,   -- absolute path
    doc_type      TEXT,               -- 'markdown' | 'pdf' | 'code_python' | 'zeal'
    topic         TEXT,               -- the topic this file was indexed under
    ingested_at   TIMESTAMP,
    chunk_count   INTEGER,
    content_hash  TEXT                -- SHA-256 of joined chunk text (first 16 hex)
);

CREATE TABLE citations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at     TIMESTAMP,
    query        TEXT,
    chunk_id     TEXT,
    source_path  TEXT,
    rank         INTEGER
);

CREATE TABLE eval_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at              TIMESTAMP,
    n_questions         INTEGER,
    recall_at_5         REAL,
    mrr                 REAL,
    recall_at_dense     REAL,
    faithfulness_proxy  REAL
);
```

Explore it from the CLI:

```powershell
rag stats                  # counts + most-cited source
rag sources --limit 20     # catalog of indexed files
rag citations --limit 20   # recent citations (filter with --source PATH)
rag eval-runs --limit 10   # recall@5 / MRR over time
```

Disable the side-channel entirely with `metadata.enabled: false` in
`config.yaml` — the pipeline silently skips the writes.

## Ingesting a Zeal docset

```powershell
# Zeal's docsets live at %LOCALAPPDATA%\Zeal\Zeal\docsets\
python cli.py ingest --zeal "$env:LOCALAPPDATA\Zeal\Zeal\docsets\Python.docset"
```

The docset is opened **read-only** (SQLite `?mode=ro` URI). The docset's
name becomes the topic (e.g. `Python.docset` → `python`).

---

## Swapping a model

Open `config.yaml` and change the relevant section. Nothing in the code
needs to move.

```yaml
# Try a smaller, faster embedder:
embedder:
  class: providers.embed_qwen3.Qwen3Embedder
  model: BAAI/bge-large-en-v1.5       # ← change model name
  quant: none                          # bge-large is small enough for fp16
  batch_size: 32

# Swap the generator to a different Ollama model:
generator:
  class: providers.llm_local.LocalGenerator
  model: qwen2.5:7b                    # ← change model name
  temperature: 0.1
```

If the new model is from a different family (not sentence-transformers),
implement its provider in `providers/` and update `embedder.class`.

---

## Repository layout

```
rag/
├── PLAN.md                  # design doc (read this if you change architecture)
├── README.md                # this file
├── config.yaml              # SINGLE source of truth for models
├── requirements.txt
├── cli.py                   # `rag ask` / `rag ingest` / `rag eval` / etc.
├── serve.py                 # FastAPI server for the GUI (P1)
├── tray.py                  # Windows tray icon (pystray + Pillow) (P1)
├── rag.bat                  # Windows shiv launcher (cmd)
├── rag.ps1                  # Windows shiv launcher (PowerShell)
├── core/
│   ├── interfaces.py        # ABCs: Embedder, Reranker, Generator, Ingester
│   ├── chunker.py           # heading-aware (md) + AST (py) + whole-file fallback
│   ├── pipeline.py          # config loader, factories, ingest/ask drivers
│   ├── walk.py              # the one directory walk every ingester shares
│   ├── refresh.py           # `rag refresh`: plan what changed, then apply it
│   └── metadata.py          # SQLite side-channel: sources / citations / eval_runs
├── providers/
│   ├── embed_qwen3.py       # real, Q4 via bitsandbytes
│   ├── embed_bgem3.py       # P1 stub (BGE-M3 dense+sparse)
│   ├── rerank_bge.py        # real, BGE cross-encoder (bge-reranker-base)
│   └── llm_local.py         # OpenAI client -> Ollama
├── ingest/
│   ├── markdown_dir.py      # walk a notes dir, respect frontmatter
│   ├── zeal_docsets.py      # read Dash-compatible docset SQLite (read-only)
│   └── pdf_dir.py           # pymupdf-based PDF ingester (P1)
├── store/
│   └── qdrant_store.py      # dense + reserved sparse slot
├── eval/
│   ├── golden_set.jsonl     # 9 hand-written Q->chunk_id pairs
│   └── run_ragas.py         # recall@5, MRR, faithfulness proxy
├── scripts/
│   ├── install-service.ps1         # Task Scheduler setup (auto-launch at logon)
│   ├── uninstall-service.ps1       # Task Scheduler removal
│   ├── install-refresh-task.ps1    # Task Scheduler setup (nightly `rag refresh`)
│   └── uninstall-refresh-task.ps1  # Task Scheduler removal
├── static/                  # the GUI (vanilla HTML + CSS + JS, no framework)
└── samples/
    └── notes/               # 5 demo notes (photography, ml, code, operations)
```

---

## P1 + P2 status — what's shipped

**P1 (walking skeleton + RAG core):** all original features.

- [x] **BGE cross-encoder reranker** (`providers/rerank_bge.py`) —
  defaults to `BAAI/bge-reranker-base` (~0.5 GB fp16) so it fits on
  a 24 GB GPU with the embedder; upgrade commented path to
  `BAAI/bge-reranker-v2-gemma` for stronger ranking.
- [x] **PDF ingester** (`ingest/pdf_dir.py`) — `rag ingest --pdf PATH`
  (file or directory), per-page chunks, deterministic chunk_ids,
  graceful corrupt-file handling.
- [x] **SQLite metadata layer** (`core/metadata.py`) — sources,
  citations, eval_runs; CLI commands `rag stats / sources /
  citations / eval-runs`; wired into both the CLI and the GUI server.
- [x] **Browser-based GUI** (FastAPI + vanilla HTML/CSS/JS) — see
  the "GUI server" section below; auto-launches at logon via the
  Task Scheduler task that `scripts/install-service.ps1` installs.

**P2 (no terminal, clear GUI, persistent backends):** done.

- [x] **Boot orchestrator** (`scripts/start_all.ps1`, `rag start`).
  One command brings up the whole stack: Ollama + Qdrant + the rag
  GUI server, in that order, with health probes, then opens the
  browser. Idempotent. Double-click `rag.bat` (no args) to trigger
  the same path; the desktop shortcut at `scripts/create-desktop-
  shortcut.ps1` pins it to the desktop.
- [x] **GUI tab nav**: Ask / Ingest / Library.
  - **Ask** — chat with `[n]` citations, topic filter dropdown.
  - **Ingest** — drag-and-drop / file-picker upload of PDFs and
    Markdown. Posts to `/api/ingest`; server saves to a temp dir
    and runs the right ingester. No terminal.
  - **Library** — read-only view of the metadata DB: stats summary,
    recent sources, recent citations, recent eval runs.
- [x] **Auto-launch at logon now brings up the full stack** (was
  just the GUI server before). `install-service.ps1` schedules
  `rag start --no-browser` so Ollama + Qdrant come up too.

Still P2 (not in this release):

- BM25 sparse vectors + RRF hybrid search — collection schema
  already reserves the slot, no recreate needed when this lands.
- EPUB ingester (`ingest/epub_dir.py`) — same shape as PDF.
- NLI-based faithfulness metric (replace the token-overlap proxy).
- Conversation memory / streaming responses.

---

## GUI (P1 — not in P0)

You use terminal every day, so a simple graphical UI is the natural next
step. The CLI stays — the GUI is a thin window on top of the same
pipeline (`ask` / `ingest` / `eval` all work the same way). Tech choice
is open; see the question at the end of the install.

---

## GUI server (P1) + P2 polish

A small browser-based UI on top of the same pipeline. Bound to a
persistent port (8420) so you always know where to find it. The CLI
commands work exactly the same — the GUI is just another surface
over the same `ask` / `ingest` / `eval` pipeline.

```powershell
# Bring up the WHOLE stack (Ollama + Qdrant + rag GUI) and open the browser.
# This is the one command that maps to the desktop shortcut.
rag start

# Or in another shell, query the running server
rag status                 # URL, PID, started_at
rag url                    # just the URL (for piping)
rag open                   # just open the browser to the running GUI
```

### Three tabs, no terminal

- **Ask** — chat input + topic filter. Answers come back with inline
  `[n]` citation markers; click a marker to scroll to its chunk.
- **Ingest** — drag PDFs or Markdown files onto the page, or click
  to pick. The server saves them to a temp dir, runs the right
  ingester, reports the chunk count. No terminal, no `rag ingest`.
- **Library** — read-only view of the metadata DB: stats summary,
  recent sources, recent citations, recent eval runs.

### One-click bring-up (the whole point)

Three ways to get to a running stack, pick whichever is closest:

```powershell
# 1. Desktop shortcut (one-time setup)
powershell -ExecutionPolicy Bypass -File D:\Tinkering sideprojects\rag\scripts\create-desktop-shortcut.ps1
# -> puts a `rag` shortcut on your desktop. Double-click -> stack up + browser opens.

# 2. Bare command (or `rag.bat` with no args from any shell)
rag start

# 3. Auto-launch at logon (one-time setup, then forget about it)
.\scripts\install-service.ps1
# After this, every logon brings up Ollama + Qdrant + rag GUI
# (15s delay so the desktop + services have time to settle).
# To remove: .\scripts\uninstall-service.ps1
```

All three paths are idempotent: if everything is already up, the
orchestrator just opens the browser. The Task Scheduler task
restarts on failure up to 3 times.

### What the UI looks like

- **Chat input + topic filter** at the bottom; type a question, hit `ask`.
- **Answer** appears with `[1] [2]` clickable markers that scroll to the
  citation in the list below.
- **Citation list** per answer: source path, section, score, and the
  full chunk text (click to expand).
- **run eval** button (top right) opens a side panel with the current
  recall@5 / MRR / per-question breakdown.
- **No framework** — vanilla HTML + CSS + a small JS file. ~250 lines total.

### Files

```
serve.py                       FastAPI app: /api/ask, /api/eval, /api/health, /api/topics
tray.py                        Windows tray icon (pystray + Pillow)
static/index.html              the chat UI
static/style.css               dark theme, terminal-adjacent
static/app.js                  vanilla JS, minimal markdown renderer
scripts/install-service.ps1    Task Scheduler setup
scripts/uninstall-service.ps1  Task Scheduler removal
```

### Tray icon (one-click GUI access)

Once the server is auto-launching at logon, the easiest way to reach
it is a click on the tray icon. After `rag tray` (and you can add
that to the same Task Scheduler task if you want it always on):

```powershell
rag tray                      # runs forever; left-click opens the GUI
```

Right-click menu: **Open rag GUI** / **Run eval** / **Status** /
**Quit**. The icon is generated programmatically (no binary asset
to ship) so it works on a fresh clone with no extra files.

The service state lives at `%USERPROFILE%\.rag\state.json` (port, pid,
started_at, url). The server writes it on startup and clears it on
clean shutdown, so `rag status` and `rag url` can find the running
server from any shell.

---

## Verification checklist (line by line)

Run from `D:\Tinkering sideprojects\rag` with the venv active.

```powershell
# 1. Python version
python --version                           # → Python 3.11.x or 3.12.x

# 2. Heavy deps loaded (no model download yet)
python -c "import torch, transformers, sentence_transformers, qdrant_client, openai, yaml; print('deps ok')"

# 3. Ollama is running with the generator model
ollama list                                 # → qwen3:30b-a3b  present
curl http://localhost:11434/api/tags        # → JSON listing qwen3:30b-a3b

# 4. Qdrant is up and reachable
curl http://localhost:6333/collections      # → {"result":{"collections":[]}}

# 5. CLI help works
python cli.py --help
python cli.py ask --help
python cli.py ingest --help
python cli.py eval --help

# 6. Ingest the sample notes (idempotent)
python cli.py ingest --markdown .\samples\notes
# expected: "done. wrote 20 chunks into kb_p0."   (14 markdown + 6 Python AST)

# 7. Verify the Qdrant collection
curl http://localhost:6333/collections/kb_p0
# expected: vectors_count > 0, points_count > 0

# 8. Ask a question, see inline [n] citations and footer
python cli.py ask "What was YaRN about?"
python cli.py ask "When should I use spot metering?" --topic photography
python cli.py ask "What's the morning routine for a solo photographer?"

# 9. JSON output for piping
python cli.py ask "What is QLoRA?" --json

# 10. Run the eval harness
python cli.py eval
# expected: recall@5 ~0.7+ (depends on Qwen3-Embedding quality), MRR ~0.8+
python cli.py eval --json

# 11. Idempotent re-ingest (re-running shouldn't grow the count)
python cli.py ingest --markdown .\samples\notes
# expected: "done. wrote 0 chunks" (or same as before; UUID5 dedupes)

# 12. Swap the embedder in config.yaml to a different model, re-ingest, re-eval
#     → confirms model swap is config-only

# 13. GUI server (P1)
python cli.py serve                     # foreground; visit http://localhost:8420
# in another shell:
rag status                             # → running, url=http://localhost:8420
rag url                                # → http://localhost:8420
# Auto-launch on logon (one-time):
.\scripts\install-service.ps1
# After logging out + back in: rag status should still show running
# To remove: .\scripts\uninstall-service.ps1

# 14. Tray icon (one-click GUI)
rag tray                               # left-click opens the GUI
# Or add it to the install-service.ps1 task so it auto-starts at logon.

# 15. PDF ingest
python cli.py ingest --pdf D:\path\to\some.pdf
python cli.py ingest --pdf D:\Library  # recursive; topic = parent dir
#   → one chunk per page (or many, if the page is long)

# 16. Metadata DB (SQLite, side-channel)
rag stats                              # counts + most-cited source
rag sources --limit 20                 # catalog of indexed files
rag citations --limit 20               # recent citations from `rag ask`
rag eval-runs --limit 10               # recall@5 / MRR over time
# Add --json to any of the four for machine-readable output.
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `bitsandbytes` install fails on Windows | Set `embedder.quant: none` in `config.yaml` (uses fp16 — adds ~10 GB VRAM but no bitsandbytes dep). |
| `OSError: libcudart.so not found` | CUDA toolkit missing. Install CUDA 12.x runtime, or set `embedder.device: cpu` (slow but works). |
| `qwen3:30b-a3b` not found by Ollama | `ollama pull qwen3:30b-a3b` — it's a MoE so the pull is large but inference is fast. |
| `Qdrant connection refused` | Check that `qdrant.exe` is running — **on this machine at `localhost:7333`** (see the port note in the Qdrant section). If `qdrant.exe` exits at boot with `os error 10048` but netstat shows nothing, a Bound-state ghost socket is squatting the port: `Get-NetTCPConnection -LocalPort <port>` reveals the owner. |
| `Torch not compiled with CUDA enabled` | PyPI's Windows `torch` wheel is CPU-only. For GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu126`. Until then the config runs CPU (`device: cpu`, `quant: none`) — slow but correct. |
| `UnicodeDecodeError` when ingesting | All file IO uses `encoding="utf-8"` already. If you see this, the source file isn't UTF-8 — convert with `iconv` or save-as UTF-8 in your editor. |
| Eval recall@5 is 0 for every question | Your collection is empty — run `python cli.py ingest --markdown .\samples\notes` first. |
| Em-dashes / accents look wrong in print output | Windows console code page. Run `chcp 65001` before `python cli.py …` (sets the active code page to UTF-8). |

---

## License

Personal project. Use it however you want.
