# P3.1 Quality & Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measured, trusted retrieval — a curated golden set over the real corpora, an eval sweep that tunes query-time knobs against it, and defaults decided by recorded numbers.

**Architecture:** Three new eval-side modules (`eval/golden_gen.py`, `eval/golden_review.py`, `eval/sweep.py`) driven by new `rag golden …` / `rag eval --sweep` CLI surfaces; one additive SQLite migration (`eval_runs.params`, `eval_runs.faithfulness_method` via `PRAGMA user_version`); `dense_weight` threaded through `ask()` so the sweep can vary it; a section-level match mode so chunking sweeps stay comparable across chunk-boundary changes.

**Tech Stack:** Existing stack only — click, qdrant-client (mocked in tests), sqlite3, pytest. No new dependencies.

## Global Constraints

- Repo: `D:\Tinkering sideprojects\rag`, branch `fix/p2-audit` (or its successor after merge).
- Test command: `".venv_tests\Scripts\python.exe" -m pytest tests/ -q` from repo root. Suite must stay green (baseline: 137 passed, 1 skipped — the ebooklib skip is expected).
- No test may require Ollama, Qdrant, GPU models, or the network — mock everything (existing convention).
- `eval/golden_candidates.jsonl` and `eval/golden_real.jsonl` are personal data: they MUST be gitignored (Task 4) and never committed.
- The GateGuard hook denies the FIRST Edit/Write per file per session — retry the identical call once.
- Commit after every task with the footers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_019QXgDZsgWNRnwg2ijiRocm`
- REALITY NOTE vs spec: config.yaml already defaults `reranker.class` to `providers.rerank_bge.BgeReranker`. The sweep therefore VALIDATES the reranker default (passthrough vs bge axis); flip only if passthrough measures ≥ bge.

---

### Task 1: eval_runs schema migration (params + faithfulness_method)

**Files:**
- Modify: `core/metadata.py` (`_SCHEMA` untouched; add `_migrate()`; extend `record_eval_run` and `get_eval_runs`)
- Test: `tests/test_metadata_migration.py` (create)

**Interfaces:**
- Consumes: existing `MetadataStore(path)`, `_init_schema()` running `executescript(_SCHEMA)` under `self._lock`.
- Produces: `record_eval_run(..., params: dict | None = None, faithfulness_method: str | None = None)`; `get_eval_runs()` rows now include `params` (decoded dict or None) and `faithfulness_method`. `PRAGMA user_version` == 1 after open. Later tasks (3, 7) rely on these exact keyword names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metadata_migration.py
"""eval_runs migration: user_version 0 -> 1 adds params + faithfulness_method."""
from __future__ import annotations

import sqlite3

from core.metadata import MetadataStore


def _columns(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def test_fresh_db_is_version_1_with_new_columns(tmp_path):
    db = tmp_path / "meta.sqlite3"
    md = MetadataStore(str(db))
    md.close()
    cols = _columns(str(db), "eval_runs")
    assert "params" in cols and "faithfulness_method" in cols
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    conn.close()


def test_reopen_is_idempotent(tmp_path):
    db = tmp_path / "meta.sqlite3"
    MetadataStore(str(db)).close()
    MetadataStore(str(db)).close()  # second open must not ALTER again / raise


def test_record_and_read_params_roundtrip(tmp_path):
    md = MetadataStore(str(tmp_path / "meta.sqlite3"))
    md.record_eval_run(
        n_questions=5, recall_at_5=0.8, mrr=0.7,
        params={"dense_weight": 0.5, "reranker": "bge"},
        faithfulness_method="nli",
    )
    runs = md.get_eval_runs()
    md.close()
    assert runs[0]["params"] == {"dense_weight": 0.5, "reranker": "bge"}
    assert runs[0]["faithfulness_method"] == "nli"


def test_record_without_params_stays_none(tmp_path):
    md = MetadataStore(str(tmp_path / "meta.sqlite3"))
    md.record_eval_run(n_questions=1, recall_at_5=1.0, mrr=1.0)
    runs = md.get_eval_runs()
    md.close()
    assert runs[0]["params"] is None
    assert runs[0]["faithfulness_method"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv_tests\Scripts\python.exe" -m pytest tests/test_metadata_migration.py -q`
Expected: FAIL (`params` not in columns / TypeError on unexpected kwarg).

- [ ] **Step 3: Implement**

In `core/metadata.py`: add `import json` to the imports if absent. In `__init__`, immediately after the `self._init_schema()` call, add `self._migrate()`. Then:

```python
_LATEST_SCHEMA_VERSION = 1


class MetadataStore:
    # ... existing ...

    def _migrate(self) -> None:
        """Additive migrations, versioned via PRAGMA user_version.

        v0 -> v1: eval_runs gains params (JSON) + faithfulness_method.
        _SCHEMA stays at the v0 shape so fresh and existing DBs take the
        exact same path through here.
        """
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                self._conn.execute("ALTER TABLE eval_runs ADD COLUMN params TEXT")
                self._conn.execute("ALTER TABLE eval_runs ADD COLUMN faithfulness_method TEXT")
                self._conn.execute("PRAGMA user_version = 1")
```

Extend `record_eval_run` — new signature and INSERT:

```python
    def record_eval_run(
        self,
        n_questions: int,
        recall_at_5: float,
        mrr: float,
        recall_at_dense: float | None = None,
        faithfulness_proxy: float | None = None,
        params: dict | None = None,
        faithfulness_method: str | None = None,
    ) -> None:
        """Append one row to `eval_runs`. The eval harness calls this once per run."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO eval_runs
                    (ran_at, n_questions, recall_at_5, mrr, recall_at_dense,
                     faithfulness_proxy, params, faithfulness_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_iso(),
                    int(n_questions),
                    float(recall_at_5),
                    float(mrr),
                    None if recall_at_dense is None else float(recall_at_dense),
                    None if faithfulness_proxy is None else float(faithfulness_proxy),
                    None if params is None else json.dumps(params, ensure_ascii=False),
                    faithfulness_method,
                ),
            )
```

Extend `get_eval_runs` SELECT to `..., faithfulness_proxy, params, faithfulness_method FROM eval_runs ...` and decode after the dict-zip:

```python
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in rows:
            r["params"] = json.loads(r["params"]) if r.get("params") else None
        return rows
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `".venv_tests\Scripts\python.exe" -m pytest tests/ -q` — all green.

- [ ] **Step 5: Commit** — `feat(metadata): eval_runs migration v1 — params + faithfulness_method`

---

### Task 2: Thread dense_weight through ask()

**Files:**
- Modify: `core/pipeline.py` (`ask()` signature; `_retrieve_hybrid()` signature; the hard-coded `dense_weight=0.5` at the `store.search_hybrid(...)` call)
- Test: `tests/test_audit_fixes.py` (append one test)

**Interfaces:**
- Consumes: `ask(query, *, embedder, store, reranker, generator, top_k_dense=20, top_k_final=5, topic=None, metadata=None, hybrid=False)` (verify exact current signature by reading it first — keyword names must not change).
- Produces: `ask(..., dense_weight: float = 0.5)` and `_retrieve_hybrid(query, qvec, store, top_k, topic, dense_weight)`. Tasks 3 and 6 pass `dense_weight` by keyword.

- [ ] **Step 1: Write the failing test** (append to `tests/test_audit_fixes.py`)

```python
def test_ask_threads_dense_weight_to_store(monkeypatch):
    """The sweep varies dense_weight; ask() must pass it to search_hybrid."""
    from unittest.mock import MagicMock
    from core import pipeline

    pipeline._hybrid_cache.update({"vocab": {"kw": 0}, "idf": {"kw": 1.0}, "built": True})
    store = MagicMock()
    store.collection = "kb"
    store.search_hybrid.return_value = []
    store.search_dense.return_value = []
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]

    pipeline.ask(
        "kw question", embedder=embedder, store=store,
        reranker=MagicMock(), generator=MagicMock(),
        hybrid=True, dense_weight=0.7,
    )
    assert store.search_hybrid.call_args.kwargs["dense_weight"] == 0.7
```

(The query must contain a vocab term — `kw` — so `_build_query_sparse_vector` doesn't fall back to dense. If `ask()`'s current positional/keyword shape differs, adapt the call but keep the assertion.)

- [ ] **Step 2: Run to verify failure** — TypeError: unexpected keyword `dense_weight`.

- [ ] **Step 3: Implement** — add `dense_weight: float = 0.5` to `ask()`'s keyword params, pass it to `_retrieve_hybrid(...)`; add the parameter to `_retrieve_hybrid(query, qvec, store, top_k, topic, dense_weight: float = 0.5)` and replace the hard-coded `dense_weight=0.5` in its `store.search_hybrid(...)` call with `dense_weight=dense_weight`.

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `feat(pipeline): thread dense_weight through ask/_retrieve_hybrid`

---

### Task 3: run_ragas.run() gains hybrid / dense_weight / extra_params / match_mode

**Files:**
- Modify: `eval/run_ragas.py` (`run()` signature + ask call + per-question matching + record_eval_run call)
- Test: `tests/test_eval_sweep_support.py` (create)

**Interfaces:**
- Consumes: Task 1's `record_eval_run(params=..., faithfulness_method=...)`; Task 2's `ask(..., hybrid=..., dense_weight=...)`.
- Produces: `run(golden_path, embedder, store, reranker, generator, top_k_dense=20, top_k_final=5, metadata=None, nli_faithfulness=None, hybrid=False, dense_weight=0.5, match_mode="chunk_id", extra_params=None) -> dict`. `match_mode="section"` matches on `(source_path, section)` pairs from golden `relevant_refs`. Tasks 6 and 8 call this exact signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_sweep_support.py
"""run() extensions for the sweep: params recording + section match mode."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from eval import run_ragas


def _golden(tmp_path, row):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return p


def _result(citations):
    r = MagicMock()
    r.answer = "a"
    r.citations = citations
    r.dense_hits = [{"chunk_id": c["chunk_id"]} for c in citations]
    return r


def test_extra_params_recorded(monkeypatch, tmp_path):
    golden = _golden(tmp_path, {"question": "q", "relevant_chunk_ids": ["c1"]})
    cite = {"chunk_id": "c1", "text": "t", "source_path": "/a.md", "section": "S"}
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _result([cite]))
    metadata = MagicMock()

    run_ragas.run(
        golden, embedder=MagicMock(), store=MagicMock(), reranker=MagicMock(),
        generator=None, metadata=metadata,
        extra_params={"dense_weight": 0.7, "reranker": "bge"},
    )
    kwargs = metadata.record_eval_run.call_args.kwargs
    assert kwargs["params"]["dense_weight"] == 0.7
    assert kwargs["params"]["top_k_dense"] == 20  # base params merged in


def test_hybrid_and_dense_weight_forwarded(monkeypatch, tmp_path):
    golden = _golden(tmp_path, {"question": "q", "relevant_chunk_ids": ["c1"]})
    seen = {}

    def fake_ask(*a, **kw):
        seen.update(kw)
        return _result([{"chunk_id": "c1", "text": "t",
                         "source_path": "/a.md", "section": "S"}])

    monkeypatch.setattr(run_ragas, "ask_pipeline", fake_ask)
    run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                  reranker=MagicMock(), generator=None,
                  hybrid=True, dense_weight=0.3)
    assert seen["hybrid"] is True
    assert seen["dense_weight"] == 0.3


def test_section_match_mode(monkeypatch, tmp_path):
    """chunk_ids drift across chunking configs; section match must not."""
    golden = _golden(tmp_path, {
        "question": "q",
        "relevant_chunk_ids": ["stale-id-from-other-chunking"],
        "relevant_refs": [{"source_path": "/a.md", "section": "S"}],
    })
    cite = {"chunk_id": "fresh-id", "text": "t", "source_path": "/a.md", "section": "S"}
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _result([cite]))

    m_chunk = run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                            reranker=MagicMock(), generator=None)
    m_sect = run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                           reranker=MagicMock(), generator=None,
                           match_mode="section")
    assert m_chunk["recall_at_5"] == 0.0   # stale id no longer exists
    assert m_sect["recall_at_5"] == 1.0    # section survives re-chunking
```

- [ ] **Step 2: Run to verify failure** — TypeError on unexpected kwargs.

- [ ] **Step 3: Implement in `eval/run_ragas.py`**

Extend `run()`'s signature exactly as in Produces. Inside the loop:

```python
        if match_mode == "section":
            refs = item.get("relevant_refs") or []
            rel = [f'{r["source_path"]}::{r["section"]}' for r in refs]
            retrieved_ids = [f'{c["source_path"]}::{c["section"]}' for c in result.citations]
            dense_ids = retrieved_ids  # dense_hits carry no section; reuse citations
        else:
            rel = item["relevant_chunk_ids"]
            retrieved_ids = [c["chunk_id"] for c in result.citations]
            dense_ids = [c["chunk_id"] for c in result.dense_hits]
```

(Existing recall/MRR helpers consume the string lists unchanged.) Forward `hybrid=hybrid, dense_weight=dense_weight` in the `ask_pipeline(...)` call. Build the recorded params once before `record_eval_run`:

```python
    run_params = {"top_k_dense": top_k_dense, "top_k_final": top_k_final,
                  "hybrid": hybrid, "dense_weight": dense_weight,
                  "match_mode": match_mode}
    if extra_params:
        run_params.update(extra_params)
```

and pass `params=run_params, faithfulness_method=summary.get("faithfulness_method")` to `metadata.record_eval_run(...)`. Also add `"params": run_params` to the returned summary. In `load_golden`, permit the optional `relevant_refs` key (list of dicts with `source_path`+`section`) — validate shape only when present:

```python
            refs = obj.get("relevant_refs")
            if refs is not None and (
                not isinstance(refs, list)
                or not all(isinstance(r, dict) and "source_path" in r and "section" in r for r in refs)
            ):
                raise ValueError(f"line {ln}: relevant_refs must be a list of "
                                 "{source_path, section} objects")
```

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `feat(eval): run() supports hybrid/dense_weight, section match, recorded params`

---

### Task 4: Golden candidate generation (eval/golden_gen.py) + gitignore

**Files:**
- Create: `eval/golden_gen.py`
- Modify: `.gitignore` (append `eval/golden_candidates.jsonl` and `eval/golden_real.jsonl`)
- Test: `tests/test_golden_gen.py` (create)

**Interfaces:**
- Consumes: `VectorStore.iter_payloads()` yielding `(point_id, payload)`; `Generator.generate(prompt: str) -> str`.
- Produces: `sample_chunks(store, n, topics=None, seed=1337) -> list[dict]` (payload dicts); `draft_question(generator, text) -> str | None`; `generate_candidates(store, generator, out_path: Path, n=100, topics=None, seed=1337) -> int` (count written, appends JSONL, skips chunk_ids already present in the file). Candidate row schema (Task 5 consumes it): `{"question", "relevant_chunk_ids": [id], "relevant_refs": [{"source_path", "section"}], "topic", "preview", "status": "candidate"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_golden_gen.py
"""Candidate generation: stratified sampling, LLM drafting, resume-safe append."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from eval import golden_gen


def _payload(cid, topic="notes"):
    return {"chunk_id": cid, "text": f"Some passage text for {cid}.",
            "source_path": f"/src/{cid}.md", "topic": topic,
            "doc_type": "markdown", "section": f"Sec {cid}"}


def _store(payloads):
    store = MagicMock()
    store.iter_payloads.return_value = iter([(p["chunk_id"], p) for p in payloads])
    return store


def test_sample_chunks_stratifies_across_topics():
    payloads = [_payload(f"a{i}", "alpha") for i in range(10)] + \
               [_payload(f"b{i}", "beta") for i in range(10)]
    picked = golden_gen.sample_chunks(_store(payloads), n=6)
    topics = [p["topic"] for p in picked]
    assert len(picked) == 6
    assert topics.count("alpha") == 3 and topics.count("beta") == 3


def test_sample_chunks_topic_filter():
    payloads = [_payload("a1", "alpha"), _payload("b1", "beta")]
    picked = golden_gen.sample_chunks(_store(payloads), n=5, topics=["beta"])
    assert [p["chunk_id"] for p in picked] == ["b1"]


def test_draft_question_strips_and_rejects_empty():
    gen = MagicMock()
    gen.generate.return_value = "  What is exposure compensation?  \n"
    assert golden_gen.draft_question(gen, "text") == "What is exposure compensation?"
    gen.generate.return_value = "   "
    assert golden_gen.draft_question(gen, "text") is None


def test_generate_candidates_appends_and_resumes(tmp_path):
    out = tmp_path / "cands.jsonl"
    gen = MagicMock()
    gen.generate.return_value = "A question?"
    store = _store([_payload("c1"), _payload("c2")])
    n1 = golden_gen.generate_candidates(store, gen, out, n=2)
    assert n1 == 2
    # Re-run with the same corpus: everything already present -> 0 new rows.
    store2 = _store([_payload("c1"), _payload("c2")])
    n2 = golden_gen.generate_candidates(store2, gen, out, n=2)
    assert n2 == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["status"] == "candidate"
    assert rows[0]["relevant_refs"][0]["section"] == "Sec c1"


def test_generate_candidates_skips_failed_drafts(tmp_path):
    out = tmp_path / "cands.jsonl"
    gen = MagicMock()
    gen.generate.side_effect = ["", "Good question?"]
    n = golden_gen.generate_candidates(_store([_payload("c1"), _payload("c2")]), gen, out, n=2)
    assert n == 1
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `eval/golden_gen.py`**

```python
"""Golden-set candidate generation over the LIVE index.

Samples chunks via VectorStore.iter_payloads(), drafts one question per
chunk with the local Generator, and appends resume-safe candidate rows.
Provenance is the point: the sampled chunk_id is the reference, so every
candidate is relevant-by-construction. Output files are gitignored —
they quote the personal corpus.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from core.interfaces import Generator, VectorStore

log = logging.getLogger(__name__)

PREVIEW_CHARS = 200
PASSAGE_CAP = 4000  # keep the drafting prompt bounded

QUESTION_PROMPT = """You are building an evaluation set for a retrieval system.
Write ONE natural question that the following passage clearly answers.
Reply with the question only - no preamble, no quotes.

Passage:
{passage}
"""


def sample_chunks(
    store: VectorStore,
    n: int,
    topics: list[str] | None = None,
    seed: int = 1337,
) -> list[dict]:
    """Round-robin across topics so no corpus dominates the golden set."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for _pid, payload in store.iter_payloads():
        topic = payload.get("topic", "default")
        if topics and topic not in topics:
            continue
        if payload.get("text"):
            by_topic[topic].append(payload)

    rng = random.Random(seed)
    for bucket in by_topic.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    buckets = sorted(by_topic)  # deterministic topic order
    i = 0
    while len(picked) < n and any(by_topic[t] for t in buckets):
        topic = buckets[i % len(buckets)]
        if by_topic[topic]:
            picked.append(by_topic[topic].pop())
        i += 1
    return picked


def draft_question(generator: Generator, text: str) -> str | None:
    q = generator.generate(QUESTION_PROMPT.format(passage=text[:PASSAGE_CAP])).strip()
    if not q:
        return None
    return q.splitlines()[0].strip() or None


def _existing_chunk_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.update(json.loads(line).get("relevant_chunk_ids", []))
    return ids


def generate_candidates(
    store: VectorStore,
    generator: Generator,
    out_path: Path,
    n: int = 100,
    topics: list[str] | None = None,
    seed: int = 1337,
) -> int:
    """Append up to n new candidate rows. Resume-safe: chunk_ids already in
    the file (any status) are never re-drafted."""
    seen = _existing_chunk_ids(out_path)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for payload in sample_chunks(store, n=n + len(seen), topics=topics, seed=seed):
            if written >= n:
                break
            cid = payload["chunk_id"]
            if cid in seen:
                continue
            try:
                question = draft_question(generator, payload["text"])
            except Exception as e:
                log.warning("golden: draft failed for %s: %s — skipping", cid, e)
                continue
            if question is None:
                log.warning("golden: empty draft for %s — skipping", cid)
                continue
            row = {
                "question": question,
                "relevant_chunk_ids": [cid],
                "relevant_refs": [{"source_path": payload["source_path"],
                                   "section": payload["section"]}],
                "topic": payload.get("topic", "default"),
                "preview": payload["text"][:PREVIEW_CHARS],
                "status": "candidate",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(cid)
            written += 1
    log.info("golden: wrote %d candidates to %s", written, out_path)
    return written
```

Append to `.gitignore`:

```
# Golden sets over the real corpus — personal data, never committed
eval/golden_candidates.jsonl
eval/golden_real.jsonl
```

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `feat(eval): golden candidate generation over the live index`

---

### Task 5: Curation — golden_review module + `rag golden` CLI group

**Files:**
- Create: `eval/golden_review.py`
- Modify: `cli.py` (new `golden` group with `generate`, `review`, `stats`)
- Test: `tests/test_golden_review.py` (create)

**Interfaces:**
- Consumes: candidate row schema from Task 4.
- Produces: `apply_decision(candidate: dict, decision: str, edited: str | None = None) -> tuple[dict, dict | None]` — returns (updated candidate, golden row or None); `load_rows(path) -> list[dict]`; `save_rows(path, rows)`; `pending(rows) -> list[dict]`; `stats(candidates_path, golden_path) -> dict`. CLI: `rag golden generate --n/--topic/--seed`, `rag golden review`, `rag golden stats`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_golden_review.py
"""Curation state machine: y/n/e transitions, resume, stats."""
from __future__ import annotations

import json

from eval import golden_review


CAND = {"question": "Draft?", "relevant_chunk_ids": ["c1"],
        "relevant_refs": [{"source_path": "/a.md", "section": "S"}],
        "topic": "notes", "preview": "…", "status": "candidate"}


def test_accept_produces_golden_row_without_status_or_preview():
    updated, golden = golden_review.apply_decision(dict(CAND), "y")
    assert updated["status"] == "accepted"
    assert golden["question"] == "Draft?"
    assert golden["relevant_chunk_ids"] == ["c1"]
    assert golden["relevant_refs"][0]["section"] == "S"
    assert "status" not in golden and "preview" not in golden


def test_reject_produces_no_golden_row():
    updated, golden = golden_review.apply_decision(dict(CAND), "n")
    assert updated["status"] == "rejected"
    assert golden is None


def test_edit_replaces_question_then_accepts():
    updated, golden = golden_review.apply_decision(dict(CAND), "e", edited="Better?")
    assert updated["status"] == "accepted"
    assert golden["question"] == "Better?"


def test_pending_filters_only_candidates():
    rows = [dict(CAND), {**CAND, "status": "accepted"}, {**CAND, "status": "rejected"}]
    assert len(golden_review.pending(rows)) == 1


def test_rows_roundtrip(tmp_path):
    p = tmp_path / "c.jsonl"
    golden_review.save_rows(p, [dict(CAND)])
    assert golden_review.load_rows(p) == [CAND]


def test_stats_counts(tmp_path):
    cands = tmp_path / "c.jsonl"
    golden = tmp_path / "g.jsonl"
    golden_review.save_rows(cands, [dict(CAND), {**CAND, "status": "rejected"}])
    golden.write_text(json.dumps({"question": "q", "relevant_chunk_ids": ["c1"],
                                  "topic": "notes"}) + "\n", encoding="utf-8")
    s = golden_review.stats(cands, golden)
    assert s["candidates_pending"] == 1
    assert s["rejected"] == 1
    assert s["accepted_total"] == 1
    assert s["accepted_by_topic"] == {"notes": 1}
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `eval/golden_review.py`**

```python
"""Curation for golden candidates: pure state transitions + JSONL IO.

The CLI drives the interaction; everything testable lives here."""
from __future__ import annotations

import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(path)


def pending(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "candidate"]


def apply_decision(candidate: dict, decision: str, edited: str | None = None):
    """y = accept, n = reject, e = accept with edited question.

    Returns (updated_candidate, golden_row_or_None). The golden row drops
    curation-only fields (status, preview)."""
    if decision == "n":
        candidate["status"] = "rejected"
        return candidate, None
    if decision == "e":
        if not edited or not edited.strip():
            raise ValueError("edit decision requires a non-empty question")
        candidate["question"] = edited.strip()
    elif decision != "y":
        raise ValueError(f"unknown decision {decision!r}")
    candidate["status"] = "accepted"
    golden = {k: v for k, v in candidate.items() if k not in ("status", "preview")}
    return candidate, golden


def stats(candidates_path: Path, golden_path: Path) -> dict:
    cands = load_rows(candidates_path)
    accepted = load_rows(golden_path)
    by_topic: dict[str, int] = {}
    for row in accepted:
        t = row.get("topic", "default")
        by_topic[t] = by_topic.get(t, 0) + 1
    return {
        "candidates_pending": len(pending(cands)),
        "rejected": sum(1 for r in cands if r.get("status") == "rejected"),
        "accepted_total": len(accepted),
        "accepted_by_topic": by_topic,
    }
```

In `cli.py`, add a `golden` group (place after the `eval` command; lazy-import the eval modules inside each command, matching `eval`'s existing pattern). Module-level constants next to the other path constants:

```python
GOLDEN_CANDIDATES = Path("eval/golden_candidates.jsonl")
GOLDEN_REAL = Path("eval/golden_real.jsonl")


@cli.group()
def golden():
    """Build and curate the golden question set over the LIVE index."""


@golden.command("generate")
@click.option("--n", default=100, type=int, help="Candidates to draft.")
@click.option("--topic", "topics", multiple=True, help="Restrict to topic(s). Repeatable.")
@click.option("--seed", default=1337, type=int, help="Sampling seed (determinism).")
@click.pass_context
def golden_generate(ctx, n, topics, seed):
    """Sample chunks from the live index and draft candidate questions (needs Ollama)."""
    from eval import golden_gen

    cfg = ctx.obj["config"]
    store = make_store(cfg)
    try:
        generator = make_generator(cfg)
    except Exception as e:
        click.echo(f"error: generator unavailable ({e}) — is Ollama running?", err=True)
        sys.exit(1)
    out = _resolve_repo_path(str(GOLDEN_CANDIDATES))
    written = golden_gen.generate_candidates(
        store, generator, out, n=n, topics=list(topics) or None, seed=seed)
    click.echo(f"wrote {written} candidates to {out}. Next: rag golden review")


@golden.command("review")
@click.pass_context
def golden_review_cmd(ctx):
    """Interactive y/n/e/q pass over pending candidates."""
    from eval import golden_review

    cands_path = _resolve_repo_path(str(GOLDEN_CANDIDATES))
    golden_path = cands_path.parent / GOLDEN_REAL.name
    rows = golden_review.load_rows(cands_path)
    todo = golden_review.pending(rows)
    if not todo:
        click.echo("no pending candidates. Run: rag golden generate")
        return
    accepted: list[dict] = golden_review.load_rows(golden_path)
    done = 0
    for cand in todo:
        click.echo(f"\nQ: {cand['question']}")
        click.echo(f"   [{cand.get('topic', 'default')}] {cand.get('preview', '')}")
        choice = click.prompt("accept? [y]es / [n]o / [e]dit / [q]uit",
                              type=click.Choice(["y", "n", "e", "q"]))
        if choice == "q":
            break
        edited = click.prompt("edited question") if choice == "e" else None
        _, golden_row = golden_review.apply_decision(cand, choice, edited=edited)
        if golden_row:
            accepted.append(golden_row)
        done += 1
    golden_review.save_rows(cands_path, rows)          # statuses updated in place
    golden_review.save_rows(golden_path, accepted)
    click.echo(f"reviewed {done}; accepted total now {len(accepted)}")


@golden.command("stats")
@click.pass_context
def golden_stats(ctx):
    """Counts: pending / rejected / accepted (per topic)."""
    from eval import golden_review

    cands_path = _resolve_repo_path(str(GOLDEN_CANDIDATES))
    s = golden_review.stats(cands_path, cands_path.parent / GOLDEN_REAL.name)
    click.echo(json.dumps(s, indent=2))
```

- [ ] **Step 4: Full suite green.** Optionally add a CLI smoke to `tests/test_golden_review.py`: `CliRunner().invoke(cli, ["golden", "--help"])` asserting `generate` / `review` / `stats` appear.

- [ ] **Step 5: Commit** — `feat(cli): rag golden generate/review/stats curation flow`

---

### Task 6: Sweep engine (eval/sweep.py)

**Files:**
- Create: `eval/sweep.py`
- Test: `tests/test_sweep.py` (create)

**Interfaces:**
- Consumes: Task 3's `run_ragas.run(...)` signature; `core.pipeline.PassthroughReranker`, `core.pipeline.make_reranker`.
- Produces: `DEFAULT_SWEEP` dict; `build_grid(cfg) -> list[dict]` (cartesian product of `cfg["eval"]["sweep"]` or defaults, stable order); `run_sweep(golden_path, cfg, embedder, store, metadata=None, generator=None, nli=None) -> list[dict]` where each result is `{"params": combo, "status": "ok"|"failed: <err>", "recall_at_5": float|None, "mrr": float|None}` sorted ok-first, best-first; `format_table(results) -> str`. Task 7 consumes all three.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep.py
"""Sweep grid construction, per-combo isolation, reranker reuse."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from eval import sweep


def test_build_grid_defaults():
    grid = sweep.build_grid({})
    # 3 dense_weights x 2 top_k_dense x 1 top_k_final x 2 rerankers = 12
    assert len(grid) == 12
    assert {"dense_weight", "top_k_dense", "top_k_final", "reranker"} == set(grid[0])


def test_build_grid_config_override():
    cfg = {"eval": {"sweep": {"dense_weight": [0.5], "top_k_dense": [10],
                              "top_k_final": [5], "reranker": ["passthrough"]}}}
    grid = sweep.build_grid(cfg)
    assert grid == [{"dense_weight": 0.5, "top_k_dense": 10,
                     "top_k_final": 5, "reranker": "passthrough"}]


def test_run_sweep_isolates_combo_failures(tmp_path):
    cfg = {"eval": {"sweep": {"dense_weight": [0.3, 0.7], "top_k_dense": [20],
                              "top_k_final": [5], "reranker": ["passthrough"]}}}
    calls = []

    def fake_run(golden_path, **kw):
        calls.append(kw)
        if kw["dense_weight"] == 0.3:
            raise RuntimeError("boom")
        return {"recall_at_5": 0.9, "mrr": 0.8}

    with patch.object(sweep.run_ragas, "run", side_effect=fake_run):
        results = sweep.run_sweep(Path("g.jsonl"), cfg,
                                  embedder=MagicMock(), store=MagicMock())
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"].startswith("failed")]
    assert len(ok) == 1 and len(failed) == 1
    assert ok[0]["recall_at_5"] == 0.9
    assert results[0]["status"] == "ok"  # ok rows sort first
    assert all(kw["hybrid"] is True for kw in calls)


def test_run_sweep_builds_each_reranker_once():
    cfg = {"eval": {"sweep": {"dense_weight": [0.3, 0.7], "top_k_dense": [20],
                              "top_k_final": [5], "reranker": ["bge"]}}}
    with patch.object(sweep.run_ragas, "run",
                      return_value={"recall_at_5": 1.0, "mrr": 1.0}), \
         patch.object(sweep, "_make_reranker_for") as mk:
        mk.return_value = MagicMock()
        sweep.run_sweep(Path("g.jsonl"), cfg, embedder=MagicMock(), store=MagicMock())
    assert mk.call_count == 1  # 2 combos, same reranker instance reused


def test_format_table_contains_params_and_metrics():
    results = [{"params": {"dense_weight": 0.5, "top_k_dense": 20,
                           "top_k_final": 5, "reranker": "bge"},
                "status": "ok", "recall_at_5": 0.91, "mrr": 0.85}]
    table = sweep.format_table(results)
    assert "0.91" in table and "bge" in table
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `eval/sweep.py`**

```python
"""Query-time eval sweep: grid over retrieval knobs, per-combo isolation.

No re-ingestion here — chunking sweeps live in their own heavy mode."""
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

from core.pipeline import PassthroughReranker, make_reranker
from eval import run_ragas

log = logging.getLogger(__name__)

DEFAULT_SWEEP: dict[str, list] = {
    "dense_weight": [0.3, 0.5, 0.7],
    "top_k_dense": [20, 40],
    "top_k_final": [5],
    "reranker": ["passthrough", "bge"],
}


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    axes = {**DEFAULT_SWEEP, **(cfg.get("eval", {}).get("sweep", {}) or {})}
    keys = ["dense_weight", "top_k_dense", "top_k_final", "reranker"]
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(axes[k] for k in keys))]


def _make_reranker_for(name: str, cfg: dict[str, Any]):
    if name == "passthrough":
        return PassthroughReranker()
    if name == "bge":
        return make_reranker(cfg)
    raise ValueError(f"unknown reranker axis value: {name!r}")


def run_sweep(
    golden_path: Path,
    cfg: dict[str, Any],
    embedder,
    store,
    metadata=None,
    generator=None,
    nli=None,
) -> list[dict[str, Any]]:
    rerankers: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    grid = build_grid(cfg)
    for i, combo in enumerate(grid, 1):
        log.info("sweep [%d/%d]: %s", i, len(grid), combo)
        try:
            if combo["reranker"] not in rerankers:
                rerankers[combo["reranker"]] = _make_reranker_for(combo["reranker"], cfg)
            metrics = run_ragas.run(
                golden_path,
                embedder=embedder,
                store=store,
                reranker=rerankers[combo["reranker"]],
                generator=generator,
                top_k_dense=combo["top_k_dense"],
                top_k_final=combo["top_k_final"],
                metadata=metadata,
                nli_faithfulness=nli,
                hybrid=True,
                dense_weight=combo["dense_weight"],
                extra_params=dict(combo),
            )
            results.append({"params": combo, "status": "ok",
                            "recall_at_5": metrics["recall_at_5"],
                            "mrr": metrics["mrr"]})
        except Exception as e:
            log.warning("sweep: combo %s failed: %s", combo, e)
            results.append({"params": combo, "status": f"failed: {e}",
                            "recall_at_5": None, "mrr": None})
    results.sort(key=lambda r: (r["status"] != "ok",
                                -(r["recall_at_5"] or 0.0),
                                -(r["mrr"] or 0.0)))
    return results


def format_table(results: list[dict[str, Any]]) -> str:
    header = f'{"dense_w":>7}  {"k_dense":>7}  {"k_final":>7}  {"reranker":<11}  {"recall@5":>8}  {"mrr":>6}  status'
    lines = [header, "-" * len(header)]
    for r in results:
        p = r["params"]
        r5 = "-" if r["recall_at_5"] is None else f'{r["recall_at_5"]:.2f}'
        mrr = "-" if r["mrr"] is None else f'{r["mrr"]:.2f}'
        lines.append(f'{p["dense_weight"]:>7}  {p["top_k_dense"]:>7}  '
                     f'{p["top_k_final"]:>7}  {p["reranker"]:<11}  '
                     f'{r5:>8}  {mrr:>6}  {r["status"]}')
    return "\n".join(lines)
```

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `feat(eval): query-time sweep engine with per-combo isolation`

---

### Task 7: Wire `rag eval --sweep` + NLI config default

**Files:**
- Modify: `cli.py` (`eval` command), `config.yaml` (add `eval:` block)
- Test: `tests/test_sweep.py` (append CLI test)

**Interfaces:**
- Consumes: Task 6's `run_sweep`/`format_table`; the existing `--nli` flag and scorer-loading block in `cli.py`'s `eval` command.
- Produces: `rag eval --sweep [--golden …]` printing the table; `--nli/--no-nli` tri-state defaulting to `cfg["eval"]["nli"]` (true). config.yaml gains the `eval:` block below.

- [ ] **Step 1: Write the failing test** (append to `tests/test_sweep.py`)

```python
def test_cli_eval_sweep_prints_table(monkeypatch):
    from click.testing import CliRunner
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "make_embedder", lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_store", lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_metadata", lambda cfg: None)

    fake_results = [{"params": {"dense_weight": 0.5, "top_k_dense": 20,
                                "top_k_final": 5, "reranker": "bge"},
                     "status": "ok", "recall_at_5": 0.91, "mrr": 0.85}]
    with patch("eval.sweep.run_sweep", return_value=fake_results):
        res = CliRunner().invoke(cli_mod.cli, ["eval", "--sweep"])
    assert res.exit_code == 0
    assert "recall@5" in res.output and "0.91" in res.output
```

- [ ] **Step 2: Run to verify failure** — `--sweep` unknown option.

- [ ] **Step 3: Implement**

In `cli.py` `eval` command: change `@click.option("--nli", is_flag=True, ...)` to `@click.option("--nli/--no-nli", default=None, help="NLI faithfulness (default: config eval.nli, true)")`, add `@click.option("--sweep", is_flag=True, help="Grid over retrieval knobs; prints a results table.")`. In the body, before the scorer-loading block:

```python
    if nli is None:
        nli = cfg.get("eval", {}).get("nli", True)
```

(keep the existing `if nli:` scorer-loading block unchanged), and after providers are built:

```python
    if sweep:
        from eval import sweep as sweep_mod
        results = sweep_mod.run_sweep(
            golden_path, cfg, embedder=embedder, store=store,
            metadata=metadata, generator=generator, nli=nli_scorer)
        if metadata is not None:
            metadata.close()
        click.echo(sweep_mod.format_table(results))
        return
```

In `config.yaml`, append:

```yaml
# ---- Eval ---------------------------------------------------------------------
eval:
  nli: true                       # NLI faithfulness by default (--no-nli to skip)
  sweep:                          # rag eval --sweep grid (query-time knobs only)
    dense_weight: [0.3, 0.5, 0.7]
    top_k_dense: [20, 40]
    top_k_final: [5]
    reranker: [passthrough, bge]
```

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `feat(cli): rag eval --sweep + NLI-by-default via config`

---

### Task 8: Chunking sweep heavy mode (scratch collections)

**Files:**
- Modify: `core/interfaces.py` (add `drop()` to `VectorStore`), `store/qdrant_store.py` (implement `drop()`), `eval/sweep.py` (add `run_chunking_sweep` + `format_chunking_table`), `cli.py` (`--sweep-chunking` + `--markdown` options on eval)
- Test: `tests/test_sweep.py` (append)

**Interfaces:**
- Consumes: `MarkdownDirIngester`, `core.pipeline.ingest`, `chunking_params(cfg)`, Task 3's `match_mode="section"`.
- Produces: `VectorStore.drop() -> None` (deletes the collection; raising default in the ABC, implemented by QdrantStore); `run_chunking_sweep(golden_path, cfg, markdown_root: str, embedder, target_tokens_list: list[int] | None = None, metadata=None) -> list[dict]` (rows shaped like `run_sweep`'s, params `{"target_tokens": int}`); `format_chunking_table(results) -> str`. CLI: `rag eval --sweep-chunking --markdown <root>`. Markdown-only in P3.1 (the personal-notes corpus); other source types deferred.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sweep.py`)

```python
def test_chunking_sweep_uses_scratch_collections_and_drops_them(tmp_path):
    cfg = {"store": {"class": "store.qdrant_store.QdrantStore",
                     "url": "http://localhost:6333", "collection": "kb",
                     "dense_dim": 4}}
    made = []

    def fake_make_store(c):
        s = MagicMock()
        s.collection = c["store"]["collection"]
        made.append(s)
        return s

    with patch.object(sweep, "make_store", side_effect=fake_make_store), \
         patch.object(sweep, "ingest_pipeline", return_value=3), \
         patch.object(sweep.run_ragas, "run",
                      return_value={"recall_at_5": 0.8, "mrr": 0.7}) as run_mock:
        results = sweep.run_chunking_sweep(
            Path("g.jsonl"), cfg, markdown_root=str(tmp_path),
            embedder=MagicMock(), target_tokens_list=[512, 1024])

    assert [s.collection for s in made] == ["kb_tune_512", "kb_tune_1024"]
    for s in made:
        s.drop.assert_called_once()          # cleaned up even on success
    assert all(c.kwargs["match_mode"] == "section"
               for c in run_mock.call_args_list)
    assert {r["params"]["target_tokens"] for r in results} == {512, 1024}


def test_chunking_sweep_drops_scratch_on_failure(tmp_path):
    cfg = {"store": {"class": "x", "url": "u", "collection": "kb", "dense_dim": 4}}
    scratch = MagicMock()
    scratch.collection = "kb_tune_512"
    with patch.object(sweep, "make_store", return_value=scratch), \
         patch.object(sweep, "ingest_pipeline", side_effect=RuntimeError("embed died")):
        results = sweep.run_chunking_sweep(
            Path("g.jsonl"), cfg, markdown_root=str(tmp_path),
            embedder=MagicMock(), target_tokens_list=[512])
    scratch.drop.assert_called_once()
    assert results[0]["status"].startswith("failed")
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`core/interfaces.py`, inside `VectorStore` (concrete raising default, mirroring the `enable_hybrid` pattern already used there):

```python
    def drop(self) -> None:
        """Delete this store's collection entirely. Scratch-collection
        lifecycle (eval chunking sweeps) depends on this."""
        raise NotImplementedError(f"{type(self).__name__} does not implement drop()")
```

`store/qdrant_store.py`:

```python
    def drop(self) -> None:
        """Delete the collection. Used by eval scratch collections."""
        self.client.delete_collection(self.collection)
```

`eval/sweep.py` — extend the core.pipeline import to `from core.pipeline import (PassthroughReranker, make_reranker, make_store, chunking_params, ingest as ingest_pipeline)`, add `import copy` and `from ingest.markdown_dir import MarkdownDirIngester`, then:

```python
DEFAULT_TARGET_TOKENS = [512, 768, 1024]


def run_chunking_sweep(
    golden_path: Path,
    cfg: dict[str, Any],
    markdown_root: str,
    embedder,
    target_tokens_list: list[int] | None = None,
    metadata=None,
) -> list[dict[str, Any]]:
    """Grid over target_tokens using scratch collections (kb_tune_<n>).

    The live collection is never touched. Metrics use section-level
    matching because chunk_ids are chunking-dependent (UUID5 over
    path/section/chunk_index). Markdown-only in P3.1."""
    results: list[dict[str, Any]] = []
    for tt in (target_tokens_list or DEFAULT_TARGET_TOKENS):
        combo = {"target_tokens": tt}
        scratch_cfg = copy.deepcopy(cfg)
        scratch_cfg["store"]["collection"] = f"kb_tune_{tt}"
        scratch_cfg.setdefault("chunking", {})["target_tokens"] = tt
        scratch = make_store(scratch_cfg)
        try:
            cp = chunking_params(scratch_cfg)
            ingester = MarkdownDirIngester(
                root=markdown_root,
                target_tokens=cp["target_tokens"],
                overlap_pct=cp["overlap_pct"],
                min_chunk_tokens=cp["min_chunk_tokens"],
                default_topic=cp["default_topic"],
                max_chunks_per_doc=cp["max_chunks_per_doc"],
            )
            ingest_pipeline(ingester, embedder, scratch, recreate=True)
            metrics = run_ragas.run(
                golden_path, embedder=embedder, store=scratch,
                reranker=PassthroughReranker(), generator=None,
                metadata=metadata, match_mode="section",
                extra_params=dict(combo, sweep="chunking"),
            )
            results.append({"params": combo, "status": "ok",
                            "recall_at_5": metrics["recall_at_5"],
                            "mrr": metrics["mrr"]})
        except Exception as e:
            log.warning("chunking sweep: target_tokens=%d failed: %s", tt, e)
            results.append({"params": combo, "status": f"failed: {e}",
                            "recall_at_5": None, "mrr": None})
        finally:
            try:
                scratch.drop()
            except Exception as e:
                log.warning("chunking sweep: could not drop %s: %s",
                            scratch.collection, e)
    results.sort(key=lambda r: (r["status"] != "ok",
                                -(r["recall_at_5"] or 0.0)))
    return results


def format_chunking_table(results: list[dict[str, Any]]) -> str:
    header = f'{"target_tokens":>13}  {"recall@5":>8}  {"mrr":>6}  status'
    lines = [header, "-" * len(header)]
    for r in results:
        r5 = "-" if r["recall_at_5"] is None else f'{r["recall_at_5"]:.2f}'
        mrr = "-" if r["mrr"] is None else f'{r["mrr"]:.2f}'
        lines.append(f'{r["params"]["target_tokens"]:>13}  {r5:>8}  {mrr:>6}  {r["status"]}')
    return "\n".join(lines)
```

Note: `recreate=True` in `ingest_pipeline` is safe here — it recreates the *scratch* collection, and the CLI confirmation gate lives in the `rag ingest` command, not in `pipeline.ingest()`.

`cli.py` `eval` command: add `@click.option("--sweep-chunking", is_flag=True, help="Heavy: re-ingest --markdown root into scratch collections per chunk size.")` and `@click.option("--markdown", "markdown_root", default=None, help="Source root for --sweep-chunking.")`; in the body before the `--sweep` branch:

```python
    if sweep_chunking:
        if not markdown_root:
            raise click.UsageError("--sweep-chunking requires --markdown <root>")
        from eval import sweep as sweep_mod
        results = sweep_mod.run_chunking_sweep(
            golden_path, cfg, markdown_root=markdown_root,
            embedder=embedder, metadata=metadata)
        if metadata is not None:
            metadata.close()
        click.echo(sweep_mod.format_chunking_table(results))
        return
```

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `feat(eval): chunking sweep over scratch collections, VectorStore.drop()`

---

### Task 9: rerank_bge capability check (audit P7)

**Files:**
- Modify: `providers/rerank_bge.py` (`__init__` + `rerank`)
- Test: `tests/test_rerank_bge.py` (append)

**Interfaces:**
- Consumes: existing `BgeReranker` with `self._model = CrossEncoder(...)`; this test module's autouse `sentence_transformers` stub fixture (post-Wave-C).
- Produces: `self._has_rank: bool` decided once in `__init__`; `rerank()` branches on it with NO try/except around inference — a CUDA OOM now propagates once, with the real traceback.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rerank_bge.py` — READ the file first and reuse its existing construction helpers/stub conventions; if none fit, build the instance via `BgeReranker.__new__(BgeReranker)` and assign `_model = MagicMock()`, `batch_size = 32`)

```python
def test_missing_rank_uses_predict_without_swallowing():
    """Older CrossEncoders lack rank(): decided ONCE at init, not per call."""
    rr = _make_reranker()                      # file's helper, or __new__ + mocks
    rr._has_rank = False                        # what __init__ computes for old API
    rr._model.predict.return_value = [0.9, 0.1]
    out = rr.rerank("q", [_chunk("A"), _chunk("B")], top_k=1)
    assert [c.chunk_id for c in out] == ["A"]
    rr._model.predict.assert_called_once()
    rr._model.rank.assert_not_called()


def test_inference_errors_propagate():
    """A CUDA OOM in rank() must surface, not trigger a second forward pass."""
    import pytest
    rr = _make_reranker()
    rr._has_rank = True
    rr._model.rank.side_effect = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        rr.rerank("q", [_chunk("A")], top_k=1)
    rr._model.predict.assert_not_called()
```

- [ ] **Step 2: Run to verify failure** — currently `predict` IS called after the swallowed error (second test fails).

- [ ] **Step 3: Implement**

In `__init__`, after `self._model = CrossEncoder(...)`:

```python
        self._has_rank = hasattr(self._model, "rank")
        if not self._has_rank:
            log.info("reranker: CrossEncoder has no rank() (older "
                     "sentence-transformers) — using predict() fallback")
```

In `rerank()`, replace the whole try/except block (keep the `pairs` construction, moving it into the else-branch since only `predict()` consumes it):

```python
        if self._has_rank:
            ranked = self._model.rank(
                query=query,
                documents=[c.text for c in chunks],
                top_k=top_k,
                batch_size=self.batch_size,
                show_progress_bar=False,
                return_documents=False,
            )
        else:
            pairs = [(query, c.text) for c in chunks]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            indexed = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
            ranked = [(idx, float(score)) for idx, score in indexed[:top_k]]
```

(Delete the old module-level `pairs = [(query, c.text) for c in chunks]` line above the block if it becomes unused.)

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit** — `fix(rerank): one-time rank() capability check; inference errors propagate`

---

### Task 10: PLAN.md refresh + operational runbook

**Files:**
- Modify: `PLAN.md` (status banner)
- Create: `docs/superpowers/specs/2026-08-10-p3-runbook.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Edit PLAN.md** — replace the `> Status:` block at the top with:

```markdown
> Status: P0–P2 shipped (P2: EPUB ingester, BM25 hybrid, NLI faithfulness,
> conversation memory — commit `3f976c9`), then hardened by the verified
> 2026-08 audit fix waves (branch `fix/p2-audit`). The plan below is the
> HISTORICAL P0 design — kept for reference.
>
> Current work: **P3** — see
> `docs/superpowers/specs/2026-08-10-p3-quality-eval-design.md` for the
> five-step roadmap (Quality & Eval → Living index → Business memory →
> Agent layer → GUI QoL) and the active sub-project.
> Owner: Jani.
```

- [ ] **Step 2: Write the runbook** — `docs/superpowers/specs/2026-08-10-p3-runbook.md`:

```markdown
# P3.1 runbook — the human part

Code done ≠ P3.1 done. The success criteria need these operator steps:

1. `rag golden generate --n 100` (Ollama must be up; a few minutes).
2. `rag golden review` — accept/edit until ≥50 accepted
   (`rag golden stats` to check; spread across topics).
3. Baseline: `rag eval --golden eval/golden_real.jsonl` (records to eval_runs).
4. `rag eval --sweep --golden eval/golden_real.jsonl` — pick the winner.
5. Optional heavy: `rag eval --sweep-chunking --markdown <notes root>`.
6. Commit the winning knobs to config.yaml with the measured delta in the
   commit message (or record "defaults already optimal").
```

- [ ] **Step 3: Run the full suite one last time (docs shouldn't break it; verify anyway).**

- [ ] **Step 4: Commit** — `docs: PLAN.md P3 status refresh + P3.1 runbook`

---

## Self-Review (done at authoring time)

- **Spec coverage:** golden generation/curation (T4, T5), sweep + recorded params (T1, T2, T3, T6, T7), chunking heavy mode + scratch lifecycle (T8), NLI default (T7), rerank P7 fix (T9), PLAN refresh (T10), gitignore (T4). Success criteria requiring the operator (≥50 curated questions, tuned-config commit) live in T10's runbook — a plan cannot curate for Jani.
- **Spec deviation, intentional:** spec said "flip reranker default from passthrough" — config.yaml already defaults to BGE; the sweep validates instead (see Global Constraints REALITY NOTE).
- **Spec addition, forced by reality:** `relevant_refs` + `match_mode="section"` (T3/T4/T8), because chunk_ids are UUID5 over (path, section, chunk_index) and cannot survive re-chunking; without this the chunking sweep would always score 0.
- **Type consistency:** `record_eval_run(params=dict|None, faithfulness_method=str|None)` identical in T1/T3; `run(...)` signature in T3 matches every call in T6/T8; candidate schema in T4 matches T5's `apply_decision` and T3's `relevant_refs` validation; `drop()` consistent across T8's ABC, impl, and tests.
- **Placeholder scan:** clean — no TBDs; every code step contains runnable content.
