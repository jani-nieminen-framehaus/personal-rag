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
whole corpus was deleted, and `--prune` wipes the index. So a missing root sets
`root_missing` and reports NOTHING as vanished — there is no flag combination
that lets an unreachable root delete anything.
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
SUFFIXES: dict[str, set[str]] = {
    "markdown": {".md", ".markdown", ".py"},
    "pdf": {".pdf"},
    "epub": {".epub"},
}

DEFAULT_ZEAL_INDEX = "docSet.dsidx"

# doc_type of the marker row a refreshed docset leaves behind. See
# `_record_zeal_marker` for why the marker has to exist at all.
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
    # Recorded source_paths, as stored, that live under this root and are gone
    # from disk. Always empty when `root_missing` — see the module docstring.
    vanished: list[str] = field(default_factory=list)
    root_missing: bool = False

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

    What resolve() would buy is symlink canonicalisation. That only matters if
    the same file is recorded under two different names, which cannot happen
    here: a recorded path and the walk that re-finds it both come from the same
    configured root.
    """
    return os.path.normcase(os.path.abspath(str(p)))


def _index_recorded(recorded: dict[str, str | None]) -> dict[str, tuple[str, str | None]]:
    """Recorded rows keyed by comparable path → (path as stored, file_hash).

    The stored form is kept because that is the key `store.delete_by_source`
    and `metadata.delete_source` need back.
    """
    return {_norm(key): (key, value) for key, value in recorded.items()}


# -----------------------------------------------------------------------------
# Zeal
# -----------------------------------------------------------------------------

def _zeal_index(docset_root: Path, sqlite_filename: str = DEFAULT_ZEAL_INDEX) -> Path:
    """The docset's SQLite index — the one file a docset is hashed by.

    A docset is all-or-nothing: there is no cheap way to tell which of its
    pages changed, and the pages are generated from this index anyway.
    """
    return Path(docset_root) / "Contents" / "Resources" / sqlite_filename


def _zeal_index_name(cfg: dict[str, Any]) -> str:
    """`ingest.zeal.sqlite_filename`, so the plan hashes the same file the
    ingester will open."""
    zeal_cfg = ((cfg.get("ingest") or {}).get("zeal") or {})
    return zeal_cfg.get("sqlite_filename") or DEFAULT_ZEAL_INDEX


def _docset_topic(docset_root: Path) -> str:
    """Mirrors `ZealIngester._docset_topic` so the marker row is filed under
    the same topic as the pages it stands for."""
    name = docset_root.name
    if name.lower().endswith(".docset"):
        name = name[:-7]
    return name.strip().lower().replace(" ", "-")


def _record_zeal_marker(
    metadata: MetadataStore, docset_root: Path, sqlite_filename: str
) -> None:
    """Record the docset's `.dsidx` as a source row in its own right.

    Ingest writes one `sources` row per PAGE, so nothing is ever keyed by the
    `.dsidx` that `plan_refresh` hashes. Without this marker the plan finds it
    absent, calls the docset "new", and re-ingests ~50,000 pages on every
    single refresh, forever. The per-page rows stay (they are what `rag
    sources` lists); this is the row refresh compares against.
    """
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

def _vanished_under(root: Path, indexed: dict[str, tuple[str, str | None]]) -> list[str]:
    """Recorded paths under `root` that are no longer on disk.

    Only called once the root is known to be present — a missing root can never
    reach here.
    """
    root_norm = _norm(root)
    prefix = root_norm + os.sep
    gone = [
        stored
        for norm, (stored, _hash) in indexed.items()
        if (norm == root_norm or norm.startswith(prefix)) and not os.path.exists(norm)
    ]
    return sorted(gone)


def _plan_one(
    entry: dict[str, Any],
    indexed: dict[str, tuple[str, str | None]],
    zeal_index_name: str = DEFAULT_ZEAL_INDEX,
) -> SourcePlan:
    """Classify one configured source. Reads only; never writes."""
    stype = entry["type"]
    root = Path(entry["path"])
    plan = SourcePlan(type=stype, path=root)

    if stype == "zeal":
        index_file = _zeal_index(root, zeal_index_name)
        # A docset without its index is as unusable as one that is not there:
        # treat both as "unknown", so neither ingests nor prunes.
        if not root.is_dir() or not index_file.is_file():
            plan.root_missing = True
            log.warning("refresh: zeal docset unavailable — skipping %s", root)
            return plan
        files = [index_file]
    else:
        suffixes = SUFFIXES.get(stype)
        if suffixes is None:
            raise ValueError(f"refresh: no suffix map for source type {stype!r}")
        if not root.exists():
            plan.root_missing = True
            log.warning(
                "refresh: source root is not present — skipping %s "
                "(nothing under it will be pruned)", root,
            )
            return plan
        files = iter_source_files(root, suffixes)

    for f in files:
        record = indexed.get(_norm(f))
        if record is None:
            plan.new.append(f)
            continue
        recorded_hash = record[1]
        if not recorded_hash:
            # NULL file_hash: a row written before schema v2, or by a caller
            # that did not hash. Unknown, so re-ingest once and record one.
            plan.changed.append(f)
        elif recorded_hash == hash_file(f):
            plan.unchanged += 1
        else:
            plan.changed.append(f)

    # A docset's recorded rows are per page, and pages are not the unit refresh
    # tracks — reporting them as vanished would let --prune shred a docset that
    # is perfectly intact. The docset either changed or it did not.
    if stype != "zeal":
        plan.vanished = _vanished_under(root, indexed)

    return plan


def plan_refresh(cfg: dict[str, Any], metadata: MetadataStore) -> RefreshPlan:
    """Decide what a refresh would do. Filesystem and database reads only."""
    if metadata is None:
        raise ValueError(
            "refresh needs a metadata store — it is the only record of what "
            "was ingested, and without it every file looks new"
        )
    entries = configured_sources(cfg)
    indexed = _index_recorded(metadata.source_hashes())
    zeal_index_name = _zeal_index_name(cfg)
    return RefreshPlan(
        sources=[_plan_one(entry, indexed, zeal_index_name) for entry in entries]
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
        zeal_cfg = ing_cfg.get("zeal") or {}
        # No only_paths: the docset is one unit.
        return ZealIngester(
            docset_path=plan.path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            sqlite_filename=zeal_cfg.get("sqlite_filename", DEFAULT_ZEAL_INDEX),
            pages_dirname=zeal_cfg.get("pages_dirname", "Contents/Resources/Documents"),
        )

    raise ValueError(f"refresh: no ingester for source type {plan.type!r}")


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
    """
    plan = plan_refresh(cfg, metadata)

    if dry_run:
        log.info(
            "refresh: dry run — %d source(s) would be re-ingested, "
            "%d recorded file(s) are gone",
            sum(1 for s in plan.sources if s.has_work),
            sum(len(s.vanished) for s in plan.sources),
        )
        return plan

    zeal_index_name = _zeal_index_name(cfg)
    embedder = None
    touched = False

    for source in plan.sources:
        if not source.has_work:
            continue
        if embedder is None:
            embedder = embedder_factory()
        ingester = _make_ingester(source, cfg, set(source.new) | set(source.changed))
        written = ingest_pipeline(ingester, embedder, store, metadata=metadata)
        touched = True
        log.info(
            "refresh: %s %s — %d new, %d changed, %d unchanged, %s chunks written",
            source.type, source.path, len(source.new), len(source.changed),
            source.unchanged, written,
        )
        if source.type == "zeal":
            _record_zeal_marker(metadata, source.path, zeal_index_name)

    if prune:
        for source in plan.sources:
            # `vanished` is already empty for a missing root; this is the
            # second lock on the same door, because the cost of being wrong
            # here is a wiped index.
            if source.root_missing:
                continue
            for path in source.vanished:
                store.delete_by_source(path)
                metadata.delete_source(path)
                touched = True
                log.info("refresh: pruned %s", path)

    if touched:
        # The corpus moved; a cached IDF map built against the old one would
        # silently skew the next hybrid query.
        invalidate_hybrid_cache()

    return plan
