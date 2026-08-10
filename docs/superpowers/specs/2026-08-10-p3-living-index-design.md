# P3.2 — Living index: design

> Status: approved 2026-08-10. Second sub-project of the P3 roadmap.
> Prereq: P3.1 merged (`main` @ 15313d2, 204 tests passing).

## Context

P3.1 gave the project a way to *measure* retrieval. P3.2 addresses the
friction Jani actually named: "re-ingesting manually, remembering flags, no
watch-mode — I want it to just stay current."

Today ingest is invocation-driven: `rag ingest --markdown D:/notes` indexes a
tree and forgets it happened. There is no record of what the index is
*supposed* to contain, so nothing can bring it up to date. Every refresh is a
full re-ingest: re-parsing every PDF and every Zeal page, re-embedding every
chunk on the GPU, for a corpus where three files changed.

Decision from the 2026-08-10 brainstorm: full scope — `rag refresh`, a
scheduled task, and `rag forget` — but no file-watcher daemon (another
background process that can die silently on a machine already running Qdrant,
Ollama and the GUI task).

## Goal

The index tracks its sources without being told to. Success criteria:

- `rag refresh` re-ingests only files whose content actually changed, and
  costs approximately nothing when nothing changed.
- A scheduled task keeps it current unattended.
- `rag forget` can erase one source or one topic, completely, on demand.
- No path can silently delete data: prune and forget both require an explicit
  flag or confirmation.

## Design

### 1. The index remembers its sources

`config.yaml` gains a `sources:` list — the record of what the index is
supposed to contain:

```yaml
sources:
  - type: markdown          # markdown | pdf | epub | zeal
    path: D:/notes
  - type: pdf
    path: D:/papers
```

`rag ingest --markdown X` keeps working unchanged for one-offs. `rag refresh`
operates on the configured list only — it never guesses.

### 2. Change detection: raw-file hashing

The existing `sources.content_hash` is computed from the *chunked* text, so
comparing it requires having already parsed and chunked the file — useless for
skipping work on a 50,000-page docset or a stack of PDFs.

Schema **v2** therefore adds `sources.file_hash`: a hash of the raw file
bytes, via a new `hash_file(path)` helper alongside the existing `hash_text`.
Refresh reads the file, hashes it, compares, and skips on a match — no
parsing, no chunking, no embedding.

The migration follows v1's tolerant pattern exactly: check `PRAGMA
table_info` per column, skip if present, set `user_version` regardless, so a
half-migrated database self-repairs. Rows written before this upgrade have a
NULL `file_hash` and are re-ingested once, after which they are tracked.

Zeal docsets have no per-file granularity worth chasing (one SQLite index,
tens of thousands of pages), so the docset's `.dsidx` file is hashed and the
docset is treated as a single all-or-nothing unit.

### 3. Shared file walk

`rag refresh` must enumerate a source's files without instantiating its
ingester. That walk currently exists three times, near-identically, inside
`markdown_dir.py`, `pdf_dir.py` and `epub_dir.py` — flagged by the original
audit as cross-cutting finding #9 and deferred. Extract it now, because
refresh needs it: `core/walk.py::iter_source_files(root, suffixes, skip_hidden)`,
with all three ingesters switched over.

### 4. `rag refresh`

For each configured source: enumerate files, hash each, compare against the
`sources` table, and collect the new and changed ones. If none, log and exit
0 without loading the embedder at all — the common case must be nearly free,
including no GPU model load.

Otherwise, construct the source's ingester restricted to just those files (a
new optional `only_paths` constructor parameter on the directory ingesters,
which the extracted walk honours) and run the normal ingest path.

`--prune` additionally removes chunks for files recorded in `sources` under a
configured root that no longer exist on disk, via `store.delete_by_source`,
and deletes their metadata rows. Never automatic. `--dry-run` prints what
would change without touching anything.

Because the corpus changed, refresh invalidates the hybrid IDF cache — the
same requirement ingest already satisfies.

### 5. `rag forget`

`rag forget --source <path>` and `rag forget --topic <name>` erase the
matching chunks and their `sources` rows. Confirmation prompt unless `--yes`,
matching the `--recreate` gate. Topic erasure resolves to the set of source
paths carrying that topic and deletes each, so one code path does the work.

This is the GDPR-shaped capability: once business data lands in P3.3, "remove
everything belonging to this client" must be one command, not an archaeology
project.

### 6. Scheduled task

`scripts/install-refresh-task.ps1` and its uninstall counterpart, registering
a `rag-refresh` task that runs `rag refresh` on a timer. Reuses
`scripts/_config.ps1` for the shared constants, and registers at
`-RunLevel Limited` — the P2 audit found the existing task requesting
elevation it does not need.

### 7. Eval history records index size

Small fix folded in here because refresh already computes it: each eval run
records the index point count in its `params`. Without it, a recall trend
across months reads as degradation when it is really just a growing corpus —
the trap now documented in the README. `params` is JSON, so no migration.

## Error handling

- A source path in config that does not exist: warn, skip that source,
  continue with the others, exit non-zero at the end. One bad entry must not
  block the rest.
- An unreadable file during hashing: warn, skip that file, continue.
- Ingest failure mid-refresh: the existing partial-metadata behaviour applies
  (record what landed, propagate). Refresh reports which sources completed.
- `--prune` with a source root that is entirely missing (unplugged drive,
  unmounted share) must NOT delete that source's chunks. Treat "root absent"
  as "unknown", not "empty" — otherwise an unmounted drive silently wipes the
  index.

## Testing

Unit coverage for: hash-skip (unchanged file is not re-ingested), change
detection (edited file is), new-file pickup, prune behaviour including the
missing-root guard, dry-run touching nothing, forget by source and by topic,
the v2 migration including the half-migrated case, and the extracted walk
against all three ingesters.

No test may require Ollama, Qdrant, GPU models, or the network — the existing
convention. The "nothing changed" path must be proven not to construct an
embedder at all.

## Out of scope

File-watcher daemon; ingesting new source types; the Framehaus SQLite
ingester (P3.3); anything in `docs/superpowers/specs/2026-08-10-p3-followups.md`
not named above.
