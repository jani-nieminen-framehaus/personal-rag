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
ollama pull qwen3:35b-a3b                       # ~20 GB one-time

# 2. Qdrant (vector store) — see full setup for the two install options
#    Quick path: download the Windows binary, run it, leave it on :6333

# 3. The CLI
cd D:\Tinkering sideprojects\rag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt                  # ~6 GB on first run (torch + Qwen3)
python cli.py ingest --markdown .\samples
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
ollama pull qwen3:35b-a3b        # ~20 GB, one-time
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
python cli.py ingest --markdown .\samples
# → "done. wrote N chunks into kb_p0."
```

You should see ~15 chunks. Open the Qdrant dashboard at
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

## Ingesting books (P1 — PDF / EPUB)

P0 only ingests Markdown and Zeal. For books (PDFs, EPUBs) the P1 ingester
will live at `ingest/pdf_dir.py` and `ingest/epub_dir.py` — same `Ingester`
ABC, same chunking pipeline.

Where to put the books is your call — the ingester takes a path. Two
natural patterns:

- **Inside the repo**, mirroring the notes tree: `library/<topic>/<author>/<title>.pdf`
- **External**, with the ingester pointed at any folder:
  `python cli.py ingest --pdf D:\Library\Photography`

Topic resolution will reuse the frontmatter pattern, falling back to the
parent directory. The recommended stack for parsing:

- **PDF**: `pymupdf` (fitz) — handles complex layouts, scans + OCR via
  Tesseract if you need it.
- **EPUB**: `ebooklib` — already extracts the chapter structure, which
  maps cleanly onto our heading-aware chunker.

## The metadata database (P1)

The current P0 keeps everything inside Qdrant's payload (chunk text,
source path, topic, section, doc_type). That works for a few thousand
chunks; the moment you want to ask things like "what did I cite most
often last month" or "which sources have I never retrieved", you'll
want a separate metadata store.

**SQLite is the right answer** for this:

- One file, in the repo (`metadata.sqlite3`). Trivial to back up, copy,
  inspect with `sqlite3` CLI.
- Zero infrastructure — no service to run alongside Qdrant.
- Mature, fast, predictable. Your Framehaus ledger already uses it.

**Not Postgres** unless you already have it running for something else.
A solo RAG over thousands of chunks doesn't need a server, and the
operational cost of "remember to back up the DB, run migrations, monitor
it" isn't worth it at this scale.

The P1 schema will be ~50 lines:

```sql
CREATE TABLE sources (         -- one row per ingested file
    source_path TEXT PRIMARY KEY,
    doc_type TEXT,
    topic TEXT,
    ingested_at TIMESTAMP,
    chunk_count INT,
    etag_or_hash TEXT           -- to detect "file changed, re-ingest"
);

CREATE TABLE citations (       -- append-only log of every [n] the model emitted
    id INTEGER PRIMARY KEY,
    asked_at TIMESTAMP,
    query TEXT,
    chunk_id TEXT,
    rank INT
);

CREATE TABLE eval_runs (       -- eval harness results
    id INTEGER PRIMARY KEY,
    ran_at TIMESTAMP,
    n_questions INT,
    recall_at_5 REAL,
    mrr REAL
);
```

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
├── cli.py                   # `rag ask` / `rag ingest` / `rag eval`
├── rag.bat                  # Windows shiv launcher (cmd)
├── rag.ps1                  # Windows shiv launcher (PowerShell)
├── core/
│   ├── interfaces.py        # ABCs: Embedder, Reranker, Generator, Ingester
│   ├── chunker.py           # heading-aware (md) + AST (py) + whole-file fallback
│   └── pipeline.py          # config loader, factories, ingest/ask drivers
├── providers/
│   ├── embed_qwen3.py       # real, Q4 via bitsandbytes
│   ├── embed_bgem3.py       # P1 stub
│   ├── rerank_bge.py        # P1 stub
│   └── llm_local.py         # OpenAI client → Ollama
├── ingest/
│   ├── markdown_dir.py      # walk a notes dir, respect frontmatter
│   └── zeal_docsets.py      # read Dash-compatible docset SQLite (read-only)
├── store/
│   └── qdrant_store.py      # dense + reserved sparse slot
├── eval/
│   ├── golden_set.jsonl     # 9 hand-written Q→chunk_id pairs
│   └── run_ragas.py         # recall@5, MRR, faithfulness proxy
└── samples/
    └── notes/               # 5 demo notes (photography, ml, code, operations)
```

---

## P1 roadmap (not in P0)

- BGE-reranker-v2 (provider file already stubbed) — flip `reranker.class`
- BM25 sparse vectors + RRF hybrid search — collection schema already reserves the slot
- PDF / EPUB ingester (`ingest/pdf_dir.py` + `ingest/epub_dir.py`)
- Simple graphical UI — see "GUI" section below
- SQLite metadata layer (sources, citations, eval_runs)
- NLI-based faithfulness metric (replace the token-overlap proxy)
- LangSmith-style trace logging

---

## GUI (P1 — not in P0)

You use terminal every day, so a simple graphical UI is the natural next
step. The CLI stays — the GUI is a thin window on top of the same
pipeline (`ask` / `ingest` / `eval` all work the same way). Tech choice
is open; see the question at the end of the install.

---

## GUI server (P1)

A small browser-based UI on top of the same pipeline. Bound to a
persistent port (8420) so you always know where to find it. The CLI
commands work exactly the same — the GUI is just another surface
over the same `ask` / `ingest` / `eval` pipeline.

```powershell
# Start the server (foreground, blocking)
rag serve
# → http://localhost:8420

# Or in another shell, query the running server
rag status                 # URL, PID, started_at
rag url                    # just the URL (for piping)
```

### Auto-launch on logon (the whole point)

The first time, run `install-service.ps1` once and forget about it. After
that, every time you log in to Windows, the rag GUI starts in the
background on port 8420.

```powershell
# Run ONCE to set up the auto-launch (Task Scheduler task)
.\scripts\install-service.ps1

# Or from anywhere:
powershell -ExecutionPolicy Bypass -File D:\Tinkering sideprojects\rag\scripts\install-service.ps1

# To remove the auto-launch:
.\scripts\uninstall-service.ps1
```

The task triggers at user logon (15s delay, so Ollama + Qdrant have
time to be reachable), restarts on failure up to 3 times, and runs
in the interactive session so your browser can open `localhost:8420`.

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
serve.py                  FastAPI app: /api/ask, /api/eval, /api/health, /api/topics
static/index.html         the chat UI
static/style.css          dark theme, terminal-adjacent
static/app.js             vanilla JS, minimal markdown renderer
scripts/install-service.ps1    Task Scheduler setup
scripts/uninstall-service.ps1  Task Scheduler removal
```

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
ollama list                                 # → qwen3:35b-a3b  present
curl http://localhost:11434/api/tags        # → JSON listing qwen3:35b-a3b

# 4. Qdrant is up and reachable
curl http://localhost:6333/collections      # → {"result":{"collections":[]}}

# 5. CLI help works
python cli.py --help
python cli.py ask --help
python cli.py ingest --help
python cli.py eval --help

# 6. Ingest the sample notes (idempotent)
python cli.py ingest --markdown .\samples
# expected: "done. wrote 16 chunks into kb_p0."   (15 markdown + 1 Python AST)

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
python cli.py ingest --markdown .\samples
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
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `bitsandbytes` install fails on Windows | Set `embedder.quant: none` in `config.yaml` (uses fp16 — adds ~10 GB VRAM but no bitsandbytes dep). |
| `OSError: libcudart.so not found` | CUDA toolkit missing. Install CUDA 12.x runtime, or set `embedder.device: cpu` (slow but works). |
| `qwen3:35b-a3b` not found by Ollama | `ollama pull qwen3:35b-a3b` — it's a MoE so the pull is large but inference is fast. |
| `Qdrant connection refused` | Check that `qdrant.exe` is running (or `docker ps` if you used the Docker option). Service should listen on `localhost:6333`. |
| `UnicodeDecodeError` when ingesting | All file IO uses `encoding="utf-8"` already. If you see this, the source file isn't UTF-8 — convert with `iconv` or save-as UTF-8 in your editor. |
| Eval recall@5 is 0 for every question | Your collection is empty — run `python cli.py ingest --markdown .\samples` first. |
| Em-dashes / accents look wrong in print output | Windows console code page. Run `chcp 65001` before `python cli.py …` (sets the active code page to UTF-8). |

---

## License

Personal project. Use it however you want.
