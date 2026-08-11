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
- No path can silently delete data *without saying so*: prune and forget both
  require an explicit flag or confirmation, and every other removal is
  reported at the time. A plain refresh does remove one thing — a changed
  file's old chunks — see "As-built deltas" §4, which amends this criterion.

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

---

## As-built deltas

> Added 2026-08-11, after the whole-branch review. The dominant failure mode
> on this branch was the plan being wrong and the code being right — and the
> false documentation that review found had been written against this spec's
> claims rather than against the code's behaviour. These are the places the
> shipped implementation differs from the design above. The code is the
> authority; this section is here so nobody has to discover that twice.

### 1. The Zeal marker row — absent from the spec entirely

The spec says a docset is hashed by its `.dsidx` and treated as one unit, and
stops there. It does not say what the plan compares that hash *against*.
Nothing did: `ZealIngester` writes one `sources` row per PAGE, so no row was
ever keyed by the `.dsidx` file. `plan_refresh` therefore found no record for
it, called the docset "new", and re-ingested a docset that can hold ~50,000
pages — on every single refresh, forever, which is precisely the perpetual
work this feature exists to abolish.

The implementation adds a marker row: `refresh.record_zeal_marker` records the
`.dsidx` as a source row in its own right (`doc_type: zeal-docset`,
`chunk_count: 0`), written after a successful ingest and never before. It is
public because a docset can also arrive through a manual `rag ingest --zeal`,
which must leave the same marker or the next refresh rebuilds from scratch.

### 2. The ownership model — absent from the spec entirely

The spec's `--prune` is "recorded files under a configured root that no longer
exist on disk". That definition is unsafe with more than one source. A parent
source claims rows belonging to a nested source whose own root is unreachable,
and deletes files that are sitting intact behind an unmounted link; and a
markdown source sweeps up a docset's rows, which name synthetic per-page paths
that never exist on disk and so look, to a bare existence check, exactly like
50,000 deleted files.

As built, every recorded row belongs to exactly ONE configured source — the
longest matching root (`_assign_owners`, and `owning_source` for callers
outside the plan) — and on top of ownership a source may only report a row
vanished if the row's suffix is one its own enumeration could have produced
(`_prunable`, `SUFFIXES`; Zeal is deliberately absent, so a docset never
prunes anything). Rows under no configured root are owned by nobody and can
never be pruned, so dropping a source from `config.yaml` does not silently
delete it. Two independent locks, because the cost of being wrong is a deleted
corpus.

### 3. Per-source error isolation, not propagation

Under "Error handling" the spec says an ingest failure mid-refresh keeps the
existing partial-metadata behaviour and *propagates*. It does not, and should
not: this runs unattended, and a single corrupt PDF must not mean that no
later source refreshed, that the prune never ran, and that a stale IDF map was
left over chunks already written. As built, a source that raises is recorded
in `SourcePlan.error`, the remaining sources still run, the hybrid cache is
still invalidated, and the CLI exits non-zero off `RefreshPlan.has_errors`
after printing which sources failed. A missing source root exits non-zero the
same way. Same contract the spec wanted for a bad config entry — "one bad
entry must not block the rest" — applied to failures too.

Isolation goes one level finer than the source since the branch review: within
a source, each CHANGED file's delete+ingest is its own unit, so one poison
file costs one file. See §4.

### 4. Delete-before-reingest — and criterion 4, amended

Nothing in the spec says a refresh deletes. It does, and it must: chunk ids
are deterministic on position, so re-ingesting a changed file overwrites
same-position chunks but ORPHANS every chunk whose heading was renamed or
whose index no longer exists once the file shrank. Nothing prunes those, so
deleted text stays queryable and keeps coming back in citations. A changed
file's old chunks are therefore removed before it is re-ingested.

So success criterion 4 as written — "No path can silently delete data" — was
never true of the shipped code. What is true:

> A plain `rag refresh` removes exactly one class of thing: the old chunks of
> a file whose content changed, replaced by the re-ingested ones. If that
> file's ingest fails, or the run is killed, that ONE file can be left short
> in the index until the next successful refresh — which retries it, because
> the failed run forgets its hash, and which names it in the summary and in
> `%USERPROFILE%\.rag\refresh.log`. Its neighbours are untouched: the
> delete+ingest pair is isolated per file. Everything else that deletes —
> `--prune`, `forget`, `ingest --recreate` — requires an explicit flag and a
> confirmation, and `--recreate` also empties the `sources` catalog, because
> rows claiming chunks in a dropped collection would make the next refresh
> report an index it had lost as up to date.

The word doing the work is "silently". Deletion is not the property to
promise; visibility and containment are.

### 5. Smaller drifts, for the record

- `rag forget` is undone by the next refresh when the erased file is still on
  disk under a configured source — refresh sees it missing from the catalog
  and calls it new. The spec's framing ("erasure must be one command") does
  not hold on its own; `forget` now warns and names the source, and the
  README says so. An exclusion mechanism remains unbuilt and unpromised.
- `metadata.path` is resolved against `config.yaml`'s directory, not the cwd,
  so `rag` run from anywhere finds the same catalog rather than creating an
  empty one beside it.
- Cache invalidation crosses processes: the corpus stamp beside the metadata
  DB is how a long-lived `rag serve` learns that the nightly refresh moved the
  corpus under it. The spec only asked for in-process invalidation, which
  reaches nobody who is actually serving queries.
