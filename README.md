# rag — personal, multi-topic RAG (P0 walking skeleton)

> A RAG CLI and desktop window for your own knowledge base. Plain Python 3.11+,
> no LangChain / LlamaIndex. Every model is swappable from `config.yaml`.
> Built for Linux + Ollama. `./rag` opens the desktop window.

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

- `cli.py` — `rag ask`, `rag ingest`, `rag eval`, `rag gui`
- `gui.py` — desktop window (Ask, Ingest, Library) over the same pipeline
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

```bash
# 1. Ollama (generator) — https://ollama.com/download
ollama pull qwen3:14b

# 2. Qdrant on PATH. The committed store.url is http://localhost:7333.
#    Qdrant's own default port is 6333. Match config.yaml to the port you run.

# 3. The CLI and the window
cd /path/to/personal-rag
./rag ingest --markdown ./samples/notes
./rag ask "What was YaRN about?"
./rag                  # Ollama + Qdrant + the desktop window
```

---

## Full setup

`./rag` with no arguments runs `rag start`: Ollama, Qdrant, then the
desktop window (Ask, Ingest, Library). The window calls the pipeline
directly. `rag serve` still serves the page in `static/` on port 8420.
If `.venv` already exists, install the new dependency once:
`.venv/bin/pip install -r requirements.txt`.

### 1. Python 3.11+

```bash
cd /path/to/personal-rag
./rag
```

The first run creates `.venv` and installs `requirements.txt`. Later runs
just start the stack.

With an NVIDIA GPU, install the CUDA wheel before the rest of the
requirements, then confirm CUDA survived the requirements install:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False`, install the CUDA wheel again after
`requirements.txt`. With no GPU, set `embedder.device` and
`reranker.device` to `cpu`, and `embedder.quant` to `none`, in
`config.yaml`.

### 2. Ollama

Install from https://ollama.com/download, then:

```bash
ollama pull qwen3:14b
```

`rag start` probes `http://localhost:11434/api/tags` and runs
`ollama serve` when nothing is listening.

### 3. Qdrant

Put the Linux `qdrant` binary on `PATH` (or set `QDRANT_BIN`). `rag start`
reads `store.url` from `config.yaml` and probes that URL. The committed
file points at port **7333**. Qdrant's normal port is **6333**. Set `store.url` to the port
you actually run. When the script starts Qdrant itself, storage is
`~/.local/share/rag/qdrant` and the HTTP port matches `store.url`.

### 4. Open the GUI

```bash
./rag start          # or just ./rag
# Ask / Ingest / Library in the desktop window
```

Running it again is safe: Ollama and Qdrant that are already up are
left alone, and the window opens. `./rag start --no-browser` starts the
server at http://127.0.0.1:8420 and leaves the window closed.

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

```bash
# any folder of .md files, with optional YAML frontmatter:
#
#   ---
#   topic: photography
#   ---
#   # Heading
#   ...
#
./rag ingest --markdown ~/notes

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
    path: ~/notes
  - type: pdf
    path: ~/Library
  - type: zeal
    path: ~/.local/share/Zeal/Zeal/docsets/Python.docset
```

Then:

```bash
./rag refresh              # re-ingest only what changed
./rag refresh --dry-run    # say what you WOULD do, write nothing
./rag refresh --prune      # ...and drop index entries whose file is gone
```

Refresh hashes every file and compares it to what it ingested last time,
so an unchanged corpus costs a directory walk and one SHA-256 per file —
**no embedding model is loaded at all**. Editing three notes re-embeds
three notes, not the library. A Zeal docset is all-or-nothing (its
`.dsidx` index is the hash), because there's no cheap way to tell which
of ~50,000 pages moved.

### What deletes, and what doesn't

**`rag refresh --prune` and `rag forget` are the only two commands that
delete *part* of the index** — selected rows, chosen deliberately. Both
ask for confirmation first (`-y` skips it, for scripts).

The one command that deletes *all* of it is **`rag ingest --recreate`**,
which drops the whole collection before re-ingesting. It asks first too,
naming the collection it is about to drop, and nothing in this README
tells you to run it. Because the collection is gone, it also empties the
`sources` catalog — those rows would otherwise claim chunks that no
longer exist, and the next `rag refresh` would believe them and report
the index up to date. Sources you didn't name on that command line are
re-ingested by the next refresh.

A plain `rag refresh` removes one thing, and it's worth knowing which: a
**changed** file's old chunks, deleted and replaced by the re-ingested
ones (chunk ids are positional, so the leftovers would keep answering
queries with text you deleted). If that file's ingest fails, or the run
is killed part-way, that one file can be left short in the index until
the next successful refresh — which retries it, and names it in the
summary. One file failing costs one file; its neighbours are untouched.

```bash
./rag refresh --dry-run --prune   # the paths it would remove, before you agree
./rag forget --source ~/notes/old.md   # one file, by the path `rag sources` shows
./rag forget --topic photography       # everything filed under a topic
```

`--dry-run` is the honest preview: the same code path as a real run with
the last step removed. Read it as an upper bound — a real run deletes no
more than it listed, and possibly fewer, because a source whose ingest
fails that run is skipped by the prune.

**`rag forget` erases the index entry, not the file.** If the file is
still on disk under something listed in `sources:`, the next `rag
refresh` finds it missing from the catalog, calls it *new*, and indexes
it again — a nightly task quietly undoing your deletion. `forget` warns
when that applies and names the source. To make an erasure stick, take
that path out of `sources:` in `config.yaml` (or delete/move the files
themselves) as well.

### Why an unplugged drive can't wipe your index

This runs unattended, which means it will eventually fire while a USB
disk is unplugged or a network share is unmounted. A naive
implementation enumerates zero files, concludes the corpus was deleted,
and `--prune` takes it out.

So **a source root that isn't there is treated as "unknown", never as
"empty"** — planned, reported, and skipped, with nothing under it
ingested or pruned. Same for a root that's present but enumerates zero
files while the index holds rows for it, and same for an unreachable
subtree below a healthy root.

A **missing root** also makes `rag refresh` exit non-zero, so a scheduled
run flags it rather than passing silently. A **refusal to prune** is the
safety property working as designed, not a failure: it exits 0 and prints
the reason it wouldn't. If you emptied a folder *on purpose*, that's what
`rag forget` is for — refusing to guess is the point.

### Doing it on a schedule

```cron
0 3 * * * /path/to/personal-rag/rag refresh >> ~/.rag/refresh.log 2>&1
```

That runs plain `rag refresh` — **never `--prune`**; unattended is the
wrong mode for the only irreversible command here. A run missed because
the machine was off waits until the next time the clock matches.

Since it never prunes, files you delete from disk stay in the index
until you run `rag refresh --prune` by hand. Refresh tells you the count
each time it notices.

A refresh that fires while the server is up is picked up without a
restart: the refresh leaves a stamp beside the metadata DB, and the
server rebuilds its BM25 vocabulary on the next hybrid query rather than
answering out of one that predates your new documents.

Everything the run printed goes to `~/.rag/refresh.log` (appended). That
is where to look when a nightly run fails — which source failed, and
which files it left for the next run.

## Ingesting books (PDF)

Drop a single PDF or a folder of PDFs at the CLI:

```bash
# One book
./rag ingest --pdf ~/Books/mybook.pdf

# A whole library, topics from the parent dir
./rag ingest --pdf ~/Library
#   ~/Library/photography/book.pdf  -> topic = "photography"
#   ~/Library/code/patterns.pdf     -> topic = "code"
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
`config.yaml`). A relative path is resolved against `config.yaml`'s own
directory, so `rag` finds the same database from whatever directory you
run it in — a cwd-relative one would create an empty database next to
you, and `rag refresh` would then call your whole corpus new. Three
tables:

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

```bash
./rag stats                  # counts + most-cited source
./rag sources --limit 20     # catalog of indexed files
./rag citations --limit 20   # recent citations (filter with --source PATH)
./rag eval-runs --limit 10   # recall@5 / MRR over time
```

Disable the side-channel entirely with `metadata.enabled: false` in
`config.yaml` — the pipeline silently skips the writes.

## Ingesting a Zeal docset

```bash
# Zeal docsets typically live under ~/.local/share/Zeal/Zeal/docsets/
./rag ingest --zeal ~/.local/share/Zeal/Zeal/docsets/Python.docset
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
├── cli.py                   # `rag ask` / `rag ingest` / `rag gui` / etc.
├── gui.py                   # desktop window: Ask, Ingest, Library
├── serve.py                 # browser page, `rag serve`
├── rag                      # launcher. No args runs `rag start`
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
│   ├── pdf_dir.py           # pymupdf-based PDF ingester
│   └── epub_dir.py          # EPUB ingester
├── store/
│   └── qdrant_store.py      # dense + reserved sparse slot
├── eval/
│   ├── golden_set.jsonl     # 9 hand-written Q->chunk_id pairs
│   └── run_ragas.py         # recall@5, MRR, faithfulness proxy
├── scripts/
│   └── start_all.sh         # Ollama + Qdrant, then the desktop window
├── static/                  # browser page served by `rag serve`
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
- [x] **Desktop window** (`gui.py`, `rag gui`) — Ask, Ingest, and
  Library over the same pipeline. `./rag` opens it.
- [x] **Browser page** (`serve.py` + `static/`) — `rag serve` on port 8420.
  `./rag start --no-browser` starts that server and leaves the window closed.

**Boot:** `scripts/start_all.sh` via `rag start`. Ollama and Qdrant come
up if they are down, then the desktop window opens. The window loads the
models itself, so that path does not also start `rag serve`.

---

## Desktop window

`./rag` is the window. Ask, Ingest, and Library call `ask`, `ingest`, and
`MetadataStore` directly. Ollama and Qdrant stay separate processes.

```bash
./rag                       # Ollama + Qdrant + the window
./rag gui                   # window only; backends must already be up
./rag start --no-browser    # server at http://127.0.0.1:8420, no window
./rag status                # URL, PID, started_at of the server
./rag open                  # browser to the running server
```

The server state file is `~/.rag/state.json` (`port`, `pid`, `started_at`,
`url`). `rag serve` writes it on startup and clears it on a clean shutdown.

Eval stays a CLI command. The Library view reads past runs.

---

## Verification checklist

Run from the repo.

```bash
# 1. Python
python3 --version                          # 3.11+

# 2. Ollama has the generator
ollama list                                # qwen3:14b
curl http://localhost:11434/api/tags

# 3. Qdrant. Use the port in config.yaml (committed value is 7333).
curl http://localhost:7333/collections

# 4. CLI
./rag ask --help
./rag ingest --markdown ./samples/notes
./rag ask "What was YaRN about?"
./rag eval

# 5. Desktop window
./rag

# 6. Browser page, if you want it
./rag serve                                # http://127.0.0.1:8420
./rag status
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `bitsandbytes` will not install | Set `embedder.quant: none` in `config.yaml` (fp16, more VRAM). |
| `OSError: libcudart.so not found` | CUDA runtime missing, or set `embedder.device: cpu` and `embedder.quant: none`. |
| `qwen3:14b` not found by Ollama | `ollama pull qwen3:14b` |
| `Qdrant connection refused` | Start `qdrant`, or let `./rag` start it. The URL is `store.url` in `config.yaml` (committed value `http://localhost:7333`). |
| `Torch not compiled with CUDA enabled` | `pip install torch --index-url https://download.pytorch.org/whl/cu126`, then install `requirements.txt` again and check `torch.cuda.is_available()`. |
| `UnicodeDecodeError` when ingesting | File IO is UTF-8. Convert the source with `iconv`. |
| Eval recall@5 is 0 for every question | The collection is empty. `./rag ingest --markdown ./samples/notes` first. |

---

## License

Personal project. Use it however you want.
