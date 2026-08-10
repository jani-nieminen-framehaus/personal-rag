# P3.1 — Quality & Eval: design

> Status: approved 2026-08-10. First sub-project of the P3 roadmap.
> Prereq: `fix/p2-audit` branch (verified-audit fix waves A–C, 137 tests green).

## Context

P2 shipped EPUB ingest, BM25 hybrid search, NLI faithfulness, and conversation
memory. A verified audit then hardened correctness (true weighted RRF, hybrid
cache invalidation, honest exit codes) and structure (VectorStore ABC,
service_state, chunking_params). P3 expands the RAG from a tinkering project
into a daily tool for both personal and business use.

Decisions from the 2026-08-10 brainstorm:

- **Daily drivers:** personal knowledge recall, business memory, agent/app
  knowledge layer (MCP + REST). Legal/GDPR reference is an occasional but
  valued edge case.
- **Business sources (later sub-projects):** Framehaus app SQLite DBs,
  Obsidian vault, documents on disk. No IMAP/email ingestion.
- **Pain points:** answer quality/trust, ingest friction, GUI roughness.
  Latency is explicitly NOT a pain point.
- **Sequencing: trust first.** Tune retrieval on a measured baseline before
  pouring in new corpora.

## P3 roadmap (each sub-project gets its own spec → plan → build cycle)

1. **Quality & Eval** — this spec.
2. **Living index** — incremental ingest via stored content hashes,
   `rag refresh`, scheduled task, per-source/per-client erasure
   (`rag forget`).
3. **Business memory** — read-only Framehaus SQLite ingester (outreach,
   promise, ledger rows → templated text chunks, client-scoped topics),
   Obsidian/documents ingest config.
4. **Agent layer** — thin MCP server over the existing pipeline (ask/search/
   status tools) for Claude sessions; Framehaus agents use existing REST.
5. **GUI QoL** — history (sessions/turns already recorded since P2), topic
   filters, citation previews.

PLAN.md gets a status refresh as part of P3.1: P0-era content marked
historical, P2-shipped noted, this roadmap linked.

## P3.1 goal

Measured, trusted retrieval over the real corpora. Success criteria:

- ≥50 curated golden questions against the live index.
- A recorded baseline (recall@5, MRR, faithfulness w/ method tag).
- A tuned config committed with its measured delta — or the honest finding
  that current defaults were already optimal.
- Reranker default decided by numbers, not vibes.

## Design

### 1. Golden set over the real corpora

New files (both gitignored — they quote the personal corpus):

- `eval/golden_candidates.jsonl` — machine-generated candidates.
- `eval/golden_real.jsonl` — curated set used for real evals.

Candidate generation (`eval/golden_gen.py` + `rag golden generate`):

- Sample N chunks from the live index via `VectorStore.iter_payloads()`
  (optionally `--topic`-filtered, stratified across topics).
- Draft one question per sampled chunk with the local Generator (Ollama).
  Provenance is preserved: the sampled chunk_id becomes the reference, so
  every candidate is relevant-by-construction.
- Candidate schema (synthetic example):
  `{"question": "…", "relevant_chunk_ids": ["<uuid>"], "status": "candidate",
    "source_path": "…", "section": "…", "preview": "first ~200 chars"}`
- Requires Ollama up; fails with an actionable message if not.

Curation (`rag golden review`):

- Interactive CLI pass over candidates: show question + preview;
  keys y (accept) / n (reject) / e (edit question text) / q (quit, resume
  later). Accepted rows move to `golden_real.jsonl` with `status` dropped;
  rejected rows are marked in the candidates file so re-runs skip them.
- `rag golden stats` prints counts (candidates/accepted/rejected, per topic).

The samples-based golden set (`eval/golden_set.jsonl` + `build_golden.py`)
stays as CI smoke — unchanged.

### 2. Eval sweep (`rag eval --sweep`)

- Query-time grid, no re-ingestion: `dense_weight` {0.3, 0.5, 0.7} ×
  `top_k_dense` {20, 40} × `top_k_final` {5} × reranker {passthrough, bge}.
  Grid is config-overridable (`eval.sweep` block in config.yaml).
- Per-combo: run the golden set, report recall@5 / MRR / faithfulness.
  Combo failures are isolated (logged, marked failed, sweep continues).
- Results table printed best-first; every run recorded in `eval_runs` with
  the new `params` column.
- Heavy mode (`rag eval --sweep-chunking`, explicit opt-in): re-ingests into
  scratch collections `kb_tune_<param-hash>` to grid over `target_tokens`;
  scratch collections deleted afterwards; the live collection is never
  touched.

### 3. Metadata schema migration (additive)

- `eval_runs` gains `params TEXT` (JSON) and `faithfulness_method TEXT`.
- Versioned via `PRAGMA user_version` (currently unset/0 → becomes 1);
  migration runs in `_init_schema`, additive `ALTER TABLE` only, no data
  rewrite. `rag eval-runs` output includes the new fields.

### 4. Defaults decided by data

- Eval runs default to NLI faithfulness (config `eval.nli: true`); the
  lexical fallback remains for machines without the NLI model and is always
  tagged via `faithfulness_method` (shipped in Wave A).
- If the sweep shows the BGE reranker earns its latency, flip config.yaml
  default from passthrough to `rerank_bge`; either way the decision and
  numbers go in the commit message.
- While touching `rerank_bge.py`: replace the verified-broken
  swallow-then-retry `except Exception` around `rank()` with a one-time
  `hasattr` capability check + logged fallback (audit P7).

## Error handling

- Golden generation: Ollama unreachable → clear error, exit 1, nothing
  written. Malformed LLM output for a candidate → skip that chunk, log,
  continue.
- Review: interrupted sessions resume (status field is the state).
- Sweep: per-combo isolation; a combo that errors is reported as failed in
  the results table, never aborts the sweep. Scratch collections are cleaned
  up in a finally block.

## Testing

- Unit: sweep grid construction (config parsing, combo expansion), golden
  candidate lifecycle (generate → review transitions → resume), migration
  (user_version 0 → 1 on an existing DB file, idempotent re-open), rerank_bge
  capability-check fallback.
- The eval math (recall/MRR/NLI tagging) is already covered by existing
  tests; sweep reuses `run()` unchanged.
- No test may require Ollama, Qdrant, or GPU models — all mocked, matching
  the existing offline-suite convention.

## Out of scope (deferred to later P3 sub-projects)

Incremental/scheduled ingest, `rag forget`, Framehaus SQLite ingester, MCP
server, GUI changes, chunker changes beyond the sweep's scratch-collection
experiments.
