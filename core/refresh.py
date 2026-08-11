"""`rag refresh` — re-ingest only what actually changed.

Two phases, deliberately separated:

- `plan_refresh` reads the filesystem and the metadata DB and decides what
  WOULD happen. It mutates nothing, so `--dry-run` is the same code path as a
  real run with the last step removed.
- `run_refresh` applies that plan.

The expensive thing in this system is the embedder: loading it costs seconds
and a few GB of VRAM. So `run_refresh` builds it lazily, once, and only if the
plan actually has work. A refresh over an unchanged corpus is a directory walk
plus one SHA-256 per file — milliseconds, no model.

THE SAFETY PROPERTY
-------------------
An absent source root means "unknown", never "empty".

This runs as a scheduled task. If a drive is unplugged or a share is unmounted
when it fires, a naive implementation enumerates zero files, concludes the
whole corpus was deleted, and `--prune` wipes the index.

"Unknown" is broader than "the configured root is gone", and the three cases
below are all treated the same way — plan them, report them, prune nothing:

1. The root itself is absent (or a docset has lost its `.dsidx`).
2. An unreachable subtree BELOW a present root — a junction to a NAS, a
   removable volume mounted into a folder. The parent enumerates perfectly
   well, so its own guard never fires. Ownership (below) is what saves it.
3. A present root that enumerates ZERO files while the index holds rows for
   it. A genuinely emptied folder therefore never prunes automatically, which
   is the right trade: `rag forget` exists for deliberate removal, and silent
   mass deletion does not.

OWNERSHIP
---------
Every recorded row belongs to exactly ONE configured source: the one whose
root is the longest matching prefix. Without that, a parent source claims rows
that belong to a nested one and deletes files it can only see half of. On top
of ownership, a source may only ever report a row as vanished if the row's
suffix is one its own enumeration could have produced — which independently
keeps a Zeal docset's synthetic per-page paths out of a markdown source's
prune list.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.metadata import MetadataStore, hash_file
from core.pipeline import (
    chunking_params,
    configured_sources,
    ingest as ingest_pipeline,
    invalidate_hybrid_cache,
)
from core.walk import iter_source_files


log = logging.getLogger(__name__)


# Which suffixes each source type contributes. Must stay in step with
# `pipeline.SOURCE_TYPES`; `_plan_one` raises rather than silently planning an
# empty source if a new type is added here without a suffix set.
#
# This doubles as the prune whitelist: a source may only report a row vanished
# if its suffix is in here. Zeal is deliberately absent — it is not walked, and
# it must never prune anything.
SUFFIXES: dict[str, set[str]] = {
    "markdown": {".md", ".markdown", ".py"},
    "pdf": {".pdf"},
    "epub": {".epub"},
}

DEFAULT_ZEAL_INDEX = "docSet.dsidx"
DEFAULT_ZEAL_PAGES = "Contents/Resources/Documents"

# doc_type of the marker row a refreshed docset leaves behind. See
# `record_zeal_marker` for why the marker has to exist at all.
ZEAL_MARKER_DOC_TYPE = "zeal-docset"


# -----------------------------------------------------------------------------
# Plan
# -----------------------------------------------------------------------------

@dataclass
class SourcePlan:
    """What one configured source needs, decided but not yet done."""

    type: str
    path: Path
    new: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    unchanged: int = 0
    # Recorded source_paths, as stored, that this source OWNS and that are gone
    # from disk. Always empty when `prune_blocked` — see the module docstring.
    vanished: list[str] = field(default_factory=list)
    # Enumerated, but hash_file could not read them. Left exactly as they are:
    # not new, not changed, not vanished.
    unreadable: list[Path] = field(default_factory=list)
    root_missing: bool = False
    # Why this source may not prune this run, in words, or None if it may.
    prune_blocked: str | None = None
    # Set if this source's ingest raised. The other sources still run.
    error: str | None = None

    @property
    def has_work(self) -> bool:
        return bool(self.new or self.changed)


@dataclass
class RefreshPlan:
    sources: list[SourcePlan] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return any(s.has_work for s in self.sources)

    @property
    def has_missing_roots(self) -> bool:
        return any(s.root_missing for s in self.sources)

    @property
    def has_blocked_prunes(self) -> bool:
        return any(s.prune_blocked for s in self.sources)

    @property
    def has_errors(self) -> bool:
        return any(s.error for s in self.sources)


# -----------------------------------------------------------------------------
# Path comparison
# -----------------------------------------------------------------------------

def _norm(p: str | Path) -> str:
    """A comparable form of a path: absolute, separator-normalised, and
    case-folded on the platforms where the filesystem is.

    NOT `Path.resolve()`, on purpose. resolve() hits the filesystem for every
    component; measured on this machine it costs ~480 us per existing path and
    ~800 us per missing one, against ~2.6 us here — roughly 200x.
    `source_hashes()` returns one row per indexed FILE, and a single Zeal
    docset can contribute ~50,000 of them, so resolving the recorded set would
    add ~25 s to every refresh — including the runs where nothing changed and
    the whole point is that it costs nothing.

    The semantic gap is closed upstream rather than here: `configured_sources`
    already resolves every root, and `iter_source_files` resolves the root it
    walks, so both sides of every comparison descend from the same canonical
    path.
    """
    return os.path.normcase(os.path.abspath(str(p)))


def _index_recorded(recorded: dict[str, str | None]) -> dict[str, tuple[str, str | None]]:
    """Recorded rows keyed by comparable path → (path as stored, file_hash).

    The stored form is kept because that is the key `store.delete_by_source`
    and `metadata.delete_source` need back.
    """
    return {_norm(key): (key, value) for key, value in recorded.items()}


def _is_under(candidate: str, root: str) -> bool:
    """Both already normalised by `_norm`."""
    return candidate == root or candidate.startswith(root + os.sep)


def _assign_owners(
    roots: list[str], indexed: dict[str, tuple[str, str | None]]
) -> list[dict[str, tuple[str, str | None]]]:
    """Split the recorded rows across the configured roots, one owner each.

    The owner is the LONGEST matching root, so a row under `…/docs/nas/`
    belongs to the `nas` source and not to `…/docs/` above it. That is the
    whole fix for a nested source whose own root is unreachable: the parent can
    no longer claim its files and delete them while they sit intact on the
    other side of an unmounted link.

    Rows under no configured root at all are owned by nobody and can never be
    pruned — dropping a source from config.yaml does not silently delete it.
    """
    owned: list[dict[str, tuple[str, str | None]]] = [{} for _ in roots]
    # Longest first, so the first match is the most specific one.
    order = sorted(range(len(roots)), key=lambda i: len(roots[i]), reverse=True)
    for norm, record in indexed.items():
        for i in order:
            if _is_under(norm, roots[i]):
                owned[i][norm] = record
                break
    return owned


# -----------------------------------------------------------------------------
# Zeal
# -----------------------------------------------------------------------------

def _zeal_index(docset_root: Path, sqlite_filename: str = DEFAULT_ZEAL_INDEX) -> Path:
    """The docset's SQLite index — the one file a docset is hashed by.

    A docset is all-or-nothing: there is no cheap way to tell which of its
    pages changed, and the pages are generated from this index anyway.
    """
    return Path(docset_root) / "Contents" / "Resources" / sqlite_filename


def _zeal_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """The `ingest.zeal` block, or an empty one. Every level uses `or`, not
    `.get(k, {})`: a section written as `zeal:` with nothing under it is a
    PRESENT key holding None, and `.get` hands that straight back."""
    return (cfg.get("ingest") or {}).get("zeal") or {}


def zeal_index_name(cfg: dict[str, Any]) -> str:
    """`ingest.zeal.sqlite_filename`, so the plan hashes the same file the
    ingester will open.

    Public, and the ONLY place this key is resolved — `cli.py` calls it too.
    A second spelling of the same fallback is not a duplicate constant, it is a
    second opinion: `.get(key, DEFAULT)` returns None for `sqlite_filename:`
    written with nothing after it, where this returns the default. The two
    disagreed on which file a docset is identified by, so the ingest crashed on
    `Path / None` while the planner beside it went on hashing docSet.dsidx.
    """
    return _zeal_cfg(cfg).get("sqlite_filename") or DEFAULT_ZEAL_INDEX


def zeal_pages_dirname(cfg: dict[str, Any]) -> str:
    """`ingest.zeal.pages_dirname` — where a docset keeps its HTML.

    Same null-safety as `zeal_index_name`, for the same reason: this value is
    joined onto the docset root, and None makes that a TypeError rather than a
    fallback.
    """
    return _zeal_cfg(cfg).get("pages_dirname") or DEFAULT_ZEAL_PAGES


def _docset_topic(docset_root: Path) -> str:
    """Mirrors `ZealIngester._docset_topic` so the marker row is filed under
    the same topic as the pages it stands for."""
    name = docset_root.name
    if name.lower().endswith(".docset"):
        name = name[:-7]
    return name.strip().lower().replace(" ", "-")


def record_zeal_marker(
    metadata: MetadataStore,
    docset_root: str | Path,
    sqlite_filename: str = DEFAULT_ZEAL_INDEX,
) -> None:
    """Record a docset's `.dsidx` as a source row in its own right.

    Ingest writes one `sources` row per PAGE, so nothing is ever keyed by the
    `.dsidx` that `plan_refresh` hashes. Without this marker the plan finds it
    absent, calls the docset "new", and re-ingests a docset that can hold
    ~50,000 pages on every single refresh, forever. The per-page rows stay
    (they are what `rag sources` lists); this is the row refresh compares
    against.

    Public because a docset can also arrive via a manual `rag ingest --zeal`,
    which must leave the same marker or the next refresh rebuilds it from
    scratch. Call it after the ingest succeeds, never before.

    Args:
        metadata: the store to write the marker into.
        docset_root: the `.docset` directory.
        sqlite_filename: `ingest.zeal.sqlite_filename` from config, if set.
    """
    docset_root = Path(docset_root)
    index_file = _zeal_index(docset_root, sqlite_filename)
    metadata.record_source(
        source_path=str(index_file),
        doc_type=ZEAL_MARKER_DOC_TYPE,
        topic=_docset_topic(docset_root),
        chunk_count=0,
        content_hash="",
        file_hash=hash_file(index_file),
    )


# -----------------------------------------------------------------------------
# Planning
# -----------------------------------------------------------------------------

def _prunable(
    owned: dict[str, tuple[str, str | None]], suffixes: set[str]
) -> dict[str, tuple[str, str | None]]:
    """The owned rows this source is even allowed to consider deleting.

    Defence in depth on top of ownership. A source may only prune a row its own
    enumeration could have produced, so a markdown source can never delete the
    `.html` rows of a Zeal docset that happens to live inside its tree — rows
    which name synthetic paths that never exist on disk and would otherwise
    look, to a plain existence check, exactly like 50,000 deleted files.
    """
    return {
        norm: record
        for norm, record in owned.items()
        if os.path.splitext(norm)[1].lower() in suffixes
    }


def _plan_one(
    entry: dict[str, Any],
    indexed: dict[str, tuple[str, str | None]],
    owned: dict[str, tuple[str, str | None]],
    index_name: str = DEFAULT_ZEAL_INDEX,
) -> SourcePlan:
    """Classify one configured source. Reads only; never writes.

    `indexed` is every recorded row — the hash lookup has to see all of them,
    because a file under a nested source is enumerated by both roots and must
    not look "new" to the outer one. `owned` is only this source's rows, and is
    the sole input to `vanished`.
    """
    stype = entry["type"]
    root = Path(entry["path"])
    plan = SourcePlan(type=stype, path=root)

    if stype == "zeal":
        index_file = _zeal_index(root, index_name)
        # A docset without its index is as unusable as one that is not there:
        # treat both as "unknown", so neither ingests nor prunes.
        if not root.is_dir() or not index_file.is_file():
            plan.root_missing = True
            plan.prune_blocked = "the docset or its .dsidx index is not present"
            log.warning("refresh: zeal docset unavailable — skipping %s", root)
            return plan
        files = [index_file]
        # A docset's rows are per page and its pages are not the unit refresh
        # tracks, so it never prunes. Reporting them would shred an intact
        # docset. It either changed or it did not.
        prunable: dict[str, tuple[str, str | None]] = {}
        plan.prune_blocked = "a docset is all-or-nothing; its pages are not tracked"
    else:
        suffixes = SUFFIXES.get(stype)
        if suffixes is None:
            raise ValueError(f"refresh: no suffix map for source type {stype!r}")
        if not root.exists():
            plan.root_missing = True
            plan.prune_blocked = "the source root is not present"
            log.warning(
                "refresh: source root is not present — skipping %s "
                "(nothing under it will be pruned)", root,
            )
            return plan
        files = iter_source_files(root, suffixes)
        prunable = _prunable(owned, suffixes)
        if not files and prunable:
            # Present but yielding nothing, while the index says it held files.
            # Unknown, not emptied.
            plan.prune_blocked = (
                f"the root is present but enumerated no files, while the index "
                f"holds {len(prunable)} row(s) for it"
            )
            log.warning(
                "refresh: %s enumerated 0 files but the index holds %d row(s) "
                "for it — refusing to prune. If you emptied it on purpose, use "
                "`rag forget`.", root, len(prunable),
            )
            return plan

    for f in files:
        current = hash_file(f)
        if not current:
            # hash_file's documented failure return. Not a hash — treating it
            # as one would re-ingest a permanently locked file every run.
            plan.unreadable.append(f)
            log.warning("refresh: cannot read %s — leaving it as it is", f)
            continue
        record = indexed.get(_norm(f))
        if record is None:
            plan.new.append(f)
        elif not record[1]:
            # NULL/empty file_hash: a row written before schema v2, or by a
            # caller that did not hash. Unknown, so re-ingest once and record.
            plan.changed.append(f)
        elif record[1] == current:
            plan.unchanged += 1
        else:
            plan.changed.append(f)

    if plan.prune_blocked is None:
        plan.vanished = sorted(
            stored for norm, (stored, _hash) in prunable.items()
            if not os.path.exists(norm)
        )

    return plan


def _index_is_deserted(store: Any, recorded_rows: int) -> bool:
    """True when the vector index holds nothing while the catalog holds rows.

    The index-side counterpart of `_plan_one`'s disk-side guard: a root that
    enumerates zero files while the DB holds rows is unknown, not empty. Point
    the same reasoning at the other side of the comparison and an index holding
    zero points while the DB holds rows is unknown, not up to date.

    Reached by `rag ingest --recreate` (which drops the whole collection), by a
    wiped Qdrant volume, or by a recreate that died before it wrote anything
    back. In every case the recorded hashes describe chunks that are not there,
    and believing them means refusing to rebuild.

    Fails OPEN. A store that cannot answer returns False, because the cost of
    guessing wrong here is a full re-embed of the entire corpus on a run that
    should have cost a directory walk.
    """
    if store is None or not recorded_rows:
        return False
    count = getattr(store, "count", None)
    if not callable(count):
        return False
    try:
        n = count()
        return n is not None and int(n) == 0
    except Exception as e:
        log.warning(
            "refresh: could not read the index size (%s) — assuming it is "
            "populated", e,
        )
        return False


def plan_refresh(
    cfg: dict[str, Any], metadata: MetadataStore, store: Any = None
) -> RefreshPlan:
    """Decide what a refresh would do. Filesystem and database reads only.

    `store` is optional and read-only here: it is asked for a point count so an
    emptied index cannot be mistaken for an up-to-date one. Omitting it just
    skips that check.
    """
    if metadata is None:
        raise ValueError(
            "refresh needs a metadata store — it is the only record of what "
            "was ingested, and without it every file looks new"
        )
    entries = configured_sources(cfg)
    indexed = _index_recorded(metadata.source_hashes())
    if _index_is_deserted(store, len(indexed)):
        log.warning(
            "refresh: the index holds 0 points while the catalog holds %d "
            "row(s) — treating every recorded hash as unknown and re-ingesting. "
            "This is what `rag ingest --recreate` over part of the corpus, or a "
            "wiped Qdrant volume, looks like from here.", len(indexed),
        )
        indexed = {norm: (stored, None) for norm, (stored, _h) in indexed.items()}
    owned = _assign_owners([_norm(e["path"]) for e in entries], indexed)
    index_name = zeal_index_name(cfg)
    return RefreshPlan(
        sources=[
            _plan_one(entry, indexed, own, index_name)
            for entry, own in zip(entries, owned)
        ]
    )


# -----------------------------------------------------------------------------
# Applying
# -----------------------------------------------------------------------------

def _make_ingester(plan: SourcePlan, cfg: dict[str, Any], only_paths: set[Path]):
    """Build the ingester for one source, restricted to the files that changed.

    Imported lazily: `plan_refresh` is the hot path and must not pay for
    bs4 / ebooklib / pymupdf on a run that turns out to have no work.
    """
    cp = chunking_params(cfg)
    ing_cfg = cfg.get("ingest") or {}

    if plan.type == "markdown":
        from ingest.markdown_dir import MarkdownDirIngester
        return MarkdownDirIngester(
            root=plan.path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            frontmatter_topic_key=(ing_cfg.get("markdown") or {}).get(
                "frontmatter_topic_key", "topic"
            ),
            max_chunks_per_doc=cp["max_chunks_per_doc"],
            only_paths=only_paths,
        )

    if plan.type == "pdf":
        from ingest.pdf_dir import PdfDirIngester
        return PdfDirIngester(
            path=plan.path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            max_chunks_per_doc=cp["max_chunks_per_doc"],
            only_paths=only_paths,
        )

    if plan.type == "epub":
        from ingest.epub_dir import EpubDirIngester
        return EpubDirIngester(
            path=plan.path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            max_chunks_per_doc=cp["max_chunks_per_doc"],
            only_paths=only_paths,
        )

    if plan.type == "zeal":
        from ingest.zeal_docsets import ZealIngester
        # Through the helpers, like `plan_refresh`. This line used to spell the
        # fallback itself with `.get(key, DEFAULT)`, so the engine's own
        # re-ingest path raised TypeError on a present-but-null value that the
        # planner ten lines away resolved fine.
        # No only_paths: the docset is one unit.
        return ZealIngester(
            docset_path=plan.path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            sqlite_filename=zeal_index_name(cfg),
            pages_dirname=zeal_pages_dirname(cfg),
        )

    raise ValueError(f"refresh: no ingester for source type {plan.type!r}")


def _refresh_one(
    source: SourcePlan,
    cfg: dict[str, Any],
    metadata: MetadataStore,
    store: Any,
    embedder: Any,
    index_name: str,
) -> None:
    """Re-ingest one source's new and changed files. Raises on failure."""
    # A changed file's OLD chunks have to go first. chunk_id is deterministic
    # on position, so re-ingesting overwrites same-position chunks but orphans
    # every chunk whose heading was renamed or whose index no longer exists
    # once the file shrank — and nothing ever prunes those, so deleted text
    # stays queryable and keeps coming back in citations. Only `changed` needs
    # this; a `new` path has nothing to delete.
    if source.type != "zeal":
        for path in source.changed:
            store.delete_by_source(str(path))

    ingester = _make_ingester(source, cfg, set(source.new) | set(source.changed))
    written = ingest_pipeline(ingester, embedder, store, metadata=metadata)
    log.info(
        "refresh: %s %s — %d new, %d changed, %d unchanged, %s chunks written",
        source.type, source.path, len(source.new), len(source.changed),
        source.unchanged, written,
    )
    if source.type == "zeal":
        record_zeal_marker(metadata, source.path, index_name)


def _mark_for_reingest(metadata: MetadataStore, source: SourcePlan) -> None:
    """After a failed ingest, forget the hashes of every file it touched.

    `pipeline.ingest` appends to its `sources` accumulator only once a batch
    has been upserted, and its abort path records those sources before
    re-raising — with `hash_file` of the WHOLE file. So a file large enough to
    span more than one embed batch ends up with a row claiming the new hash
    while only the first batch is in the index. Every later refresh then reads
    it as unchanged and never revisits it.

    That non-convergence predates delete-before-reingest; what that ordering
    changed is the consequence. The file's old chunks are already gone by then,
    so the un-indexed tail is not stale, it is missing. The run does say so at
    the time via `SourcePlan.error`, but nothing would ever go back and fix it.

    Clearing the hash makes the next refresh treat these files as changed.
    Both `changed` and `new` need it: a new file has no row until the ingest
    writes one, and a partial ingest writes exactly the same bad row.

    Never raises — it runs inside an except handler and must not mask the
    original failure.
    """
    paths = [str(p) for p in source.new + source.changed]
    if not paths:
        return
    try:
        cleared = metadata.clear_file_hashes(paths)
    except Exception as e:  # pragma: no cover — defensive
        log.error(
            "refresh: could not clear file hashes under %s (%s) — those files "
            "may read as unchanged next run", source.path, e,
        )
        return
    if cleared:
        log.warning(
            "refresh: cleared %d file hash(es) under %s so the next run "
            "re-ingests them from scratch", cleared, source.path,
        )


def run_refresh(
    cfg: dict[str, Any],
    metadata: MetadataStore,
    store: Any,
    embedder_factory: Callable[[], Any],
    *,
    prune: bool = False,
    dry_run: bool = False,
) -> RefreshPlan:
    """Apply the plan and return it.

    `embedder_factory` is called at most once, and only when there is
    something to ingest.

    One source failing does not stop the others. This is an unattended
    scheduled task: a single corrupt PDF must not mean nothing else refreshed,
    the prune never ran, and the hybrid cache was left stale over chunks that
    had already been written.
    """
    plan = plan_refresh(cfg, metadata, store)

    if dry_run:
        log.info(
            "refresh: dry run — %d source(s) would be re-ingested, "
            "%d recorded file(s) are gone",
            sum(1 for s in plan.sources if s.has_work),
            sum(len(s.vanished) for s in plan.sources),
        )
        return plan

    index_name = zeal_index_name(cfg)
    embedder = None
    touched = False

    try:
        for source in plan.sources:
            if not source.has_work:
                continue
            if embedder is None:
                embedder = embedder_factory()
            # Set before the attempt, not after: `pipeline.ingest` records the
            # chunks it managed to write before aborting, so a failure still
            # means the corpus moved.
            touched = True
            try:
                _refresh_one(source, cfg, metadata, store, embedder, index_name)
            except Exception as e:
                source.error = f"{type(e).__name__}: {e}"
                log.error(
                    "refresh: %s %s failed — %s (continuing with the "
                    "remaining sources)", source.type, source.path, source.error,
                )
                # The ingest may have written a row claiming the new hash for
                # content that is only partly indexed — and whose old chunks
                # this run already deleted. Forget the hash so the next run
                # rebuilds these files instead of calling them unchanged.
                _mark_for_reingest(metadata, source)

        if prune:
            for source in plan.sources:
                # `vanished` is already empty whenever prune_blocked is set;
                # this is the second lock on the same door, because the cost of
                # being wrong here is a deleted corpus.
                if source.prune_blocked:
                    continue
                if source.error:
                    # Something is wrong with this source. Deleting under it is
                    # the one irreversible thing here; skipping a run is free.
                    log.warning(
                        "refresh: not pruning %s — its ingest failed this run",
                        source.path,
                    )
                    continue
                for path in source.vanished:
                    store.delete_by_source(path)
                    metadata.delete_source(path)
                    touched = True
                    log.info("refresh: pruned %s", path)
    finally:
        if touched:
            # The corpus moved; a cached IDF map built against the old one
            # would silently skew the next hybrid query. In a finally so it
            # still runs if the prune loop itself blows up.
            invalidate_hybrid_cache()

    return plan
