# P3.1 — known follow-ups

Everything below was found by review, judged non-blocking, and deliberately
left. Recorded here so it is not rediscovered from scratch. Nothing here
affects correctness of the shipped features; the branch was reviewed as
"ready to merge" with these open.

## Parked by the final re-review

- **`eval/golden_gen.py`** — a payload with no `chunk_id` is skipped with no
  log line, folding a corruption case into the same branch as the benign
  "already seen, resume" case. One-line `log.warning` closes it. Same class
  of silent skip that the malformed-payload fix removed elsewhere.
- **`cli.py`** — `rag eval --sweep --no-hybrid` silently ignores the flag.
  The sweep owns the `hybrid` axis and pins it to `True`, so the behaviour is
  defensible, but a silently-ignored flag is still a trap. Either forward it
  or reject the combination.
- **`core/pipeline.py`** — the `query_sparse is None` dense fallback has no
  direct test; only the `except Exception` fallback does. They share
  `_post_filter` and `fetch_k`, so the risk is inspection-level.
- **`core/pipeline.py`** — the hybrid dense fallbacks post-filter
  `search_dense` output instead of calling `store.search_with_filter`, which
  pushes the predicate server-side. For a topic holding under 1/4 of the
  corpus, hybrid can return fewer candidates than dense-only for the same
  query. Documented as intentional in the `OVERFETCH` comment ("honest rather
  than confounded"), but worth revisiting if topic-scoped hybrid recall
  disappoints.

## Deferred during the task loop

- **`core/metadata.py`** — `PRAGMA user_version` is set via f-string
  interpolation of a module constant; PRAGMA cannot bind parameters, but the
  reason deserves a comment since f-string-into-SQL is a pattern reviewers
  reflexively flag.
- **`eval/run_ragas.py`** — `recall_at_5` hardcodes k=5 while `top_k_final`
  is a swept, config-reachable axis. A `top_k_final > 5` would under-report
  under a column labelled `recall@5`. Either derive k from `top_k_final` or
  document that the axis must stay at 5.
- **`eval/run_ragas.py`** — `extra_params` can shadow a base `params` key, so
  a careless sweep driver could record a `params` row that misdescribes the
  run that produced it. Intentional per the plan and documented in the
  docstring.
- **`eval/sweep.py`** — a reranker whose *construction* throws is never
  cached, so every combination naming it retries the load. On the
  out-of-memory path `config.yaml` warns about, that is several slow repeats.
- **`eval/sweep.py`** — no test covers `format_table` with a failed
  (None-metric) row, and none pins the `cfg={"eval": None}` case the
  defensive config lookup exists for.
- **`cli.py`** — `run_sweep` / `run_ragas.run` are not wrapped in
  `try/finally`, so an exception skips `metadata.close()`. The process exits
  immediately after and SQLite's WAL recovers on next open, so this is
  cosmetic today.
- **`cli.py`** — the `if metadata is not None: metadata.close()` guard is
  duplicated across the two return paths.
- **Task 5 / `cli.py`** — a `save_rows` failure inside the curation
  `finally` could mask a propagating `KeyboardInterrupt`. Needs a disk error
  during a Ctrl-C to bite.
- **`eval/golden_gen.py` CLI path** — `rag golden generate` has no automated
  coverage beyond a static signature check; the underlying module is well
  covered and the CLI wrapper is thin.
- **`scripts/uninstall-service.ps1`** — uses `Get-Process -Id` plus a
  `ProcessName -match 'python'` check rather than the `Test-PidAlive` helper
  the consolidation created, so a third liveness idiom exists. Also
  `(Read-StateJson).pid` relies on `$null.pid` returning `$null`, which
  breaks the moment anyone adds `Set-StrictMode`.
- **`scripts/start_all.ps1`** — the header comment describes the scheduled
  task as running `python -m cli serve`; `install-service.ps1` actually
  registers `-m cli start --no-browser`, which re-enters `start_all.ps1`.
  Task Scheduler's default `IgnoreNew` policy swallows the re-entry, then the
  outer script waits 120 s for a server nobody spawned and exits 1.
  Pre-existing; the comment describes the design that would have been right.
- **`core/pipeline.py`** — `_hybrid_cache` is a module global not keyed by
  collection. Harmless today because `--sweep` and `--sweep-chunking` are
  mutually exclusive branches, but it is latent cross-collection
  contamination if they are ever composed in one process.

## Known trap, accepted

- **`eval/golden_set.jsonl`** (git-tracked, samples-based smoke set) carries
  no `relevant_refs`, so it cannot be used with `match_mode="section"`.
  Regenerating it would bake one machine's absolute paths into a committed
  file, which is the worse trade. Nothing runs it in section mode today —
  the only section-mode caller is `rag eval --sweep-chunking`, which the
  runbook points at `golden_real.jsonl`. The failure is loud if anyone tries:
  every combination raises, the table prints, and the command exits 1.
  The clean long-term fix is to normalise both sides of the section key
  inside `run_ragas` so the repo-relative form also matches.

## Not automated

There is no CI. "CI smoke" means the local pytest suite plus the GUI's
`/api/eval`. Nothing enforces the test baseline automatically.
