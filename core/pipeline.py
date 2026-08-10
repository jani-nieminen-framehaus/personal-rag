"""Pipeline orchestration.

Pulls together: config loading, factory for every ABC, ingest driver, and
the ask driver (embed query → retrieve top-k dense → rerank → build prompt
→ generate).

All factories take a section of the parsed config dict and return a concrete
instance. The CLI only ever talks to these factories — it never instantiates
a provider directly. That is what makes the whole system swappable from
config.yaml.
"""
from __future__ import annotations

import importlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from core.interfaces import Chunk, Embedder, Reranker, Generator, Ingester, VectorStore
from core.metadata import MetadataStore, hash_text


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Path is overridable via RAG_CONFIG env var."""
    p = Path(path or os.environ.get("RAG_CONFIG") or DEFAULT_CONFIG_PATH)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping; got {type(cfg).__name__}")
    return cfg


def chunking_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Chunking + ingest defaults, resolved in exactly one place.

    Every ingester construction site (cli, serve, eval/build_golden) reads
    these through here. The defaults used to be copy-pasted at 8+ call
    sites — one copy drifting was the main way the eval could silently
    index a different corpus than the live one.
    """
    ch = cfg.get("chunking", {}) or {}
    ing = cfg.get("ingest", {}) or {}
    return {
        "target_tokens": ch.get("target_tokens", 768),
        "overlap_pct": ch.get("overlap_pct", 12),
        "min_chunk_tokens": ch.get("min_chunk_tokens", 32),
        "default_topic": ing.get("default_topic", "default"),
        "max_chunks_per_doc": ch.get("max_chunks_per_doc", 2000),
    }


def _import_class(dotted: str):
    """Import a class from a dotted path like 'providers.embed_qwen3.Qwen3Embedder'."""
    module_name, _, class_name = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"invalid class path: {dotted!r}")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"{dotted!r} not found")
    return cls


# -----------------------------------------------------------------------------
# Factories
# -----------------------------------------------------------------------------

def make_embedder(cfg: dict[str, Any]) -> Embedder:
    section = cfg["embedder"]
    cls = _import_class(section["class"])
    # Filter kwargs to the ones the constructor accepts so config can have
    # extra comments / future keys without breaking instantiation.
    import inspect
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in section.items() if k != "class" and k in sig.parameters}
    log.info("embedder: %s(%s)", section["class"], ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    return cls(**kwargs)


def make_reranker(cfg: dict[str, Any]) -> Reranker:
    section = cfg["reranker"]
    cls = _import_class(section["class"])
    import inspect
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in section.items() if k != "class" and k in sig.parameters}
    log.info("reranker: %s(%s)", section["class"], ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    return cls(**kwargs)


def make_generator(cfg: dict[str, Any]) -> Generator:
    section = cfg["generator"]
    cls = _import_class(section["class"])
    import inspect
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in section.items() if k != "class" and k in sig.parameters}
    log.info("generator: %s(%s)", section["class"], ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    return cls(**kwargs)


def make_store(cfg: dict[str, Any]) -> VectorStore:
    """P0 only has one store impl, so this is direct — but the factory keeps
    the same shape as the others so a new store only needs to be added to
    store/ and pointed at via config."""
    section = cfg["store"]
    cls = _import_class(section["class"])
    import inspect
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in section.items() if k != "class" and k in sig.parameters}
    log.info("store: %s(%s)", section["class"], ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    return cls(**kwargs)


def make_metadata(cfg: dict[str, Any]) -> MetadataStore | None:
    """Build a MetadataStore from the `metadata:` config block, or None if disabled.

    Config shape:
        metadata:
          enabled: true            # default true
          path: ./metadata.sqlite3 # default ./metadata.sqlite3
    """
    section = cfg.get("metadata", {}) or {}
    if not section.get("enabled", True):
        return None
    path = section.get("path", "./metadata.sqlite3")
    return MetadataStore(path)


# -----------------------------------------------------------------------------
# Reranker default
# -----------------------------------------------------------------------------

class PassthroughReranker(Reranker):
    """No-op reranker. P0 default. Just returns the input as-is, sliced to top_k."""

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        return chunks[:top_k]


# -----------------------------------------------------------------------------
# Ingest driver
# -----------------------------------------------------------------------------

def ingest(
    ingester: Ingester,
    embedder: Embedder,
    store: VectorStore,
    *,
    recreate: bool = False,
    batch_size: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    metadata: MetadataStore | None = None,
) -> int:
    """Walk the ingester, embed, upsert. Returns number of chunks written.

    Args:
        ingester: yields chunks.
        embedder: turns chunk text into vectors.
        store: writes to Qdrant.
        recreate: if True, drop the collection first (dev wipe).
        batch_size: override embedder batch size (None = use embedder's default).
        progress: optional callback(done, total). `total` is unknown up front
                  for streaming ingesters, so we pass done and -1.
        metadata: optional MetadataStore. If set, we record one row per
            unique source_path in the `sources` table after the embed
            loop finishes. The content_hash is computed from the joined
            chunk text (deterministic given a deterministic chunker).
    """
    # Pass the embedder's dim so ensure_collection can validate the
    # existing collection (or use the dim when creating a new one).
    # Bug #7 fix: previously the store's config-only dim was used and
    # the embedder's actual dim was never checked.
    store.ensure_collection(recreate=recreate, expected_dense_dim=embedder.dim())

    # Buffer chunks for batched embedding. We pick the embedder's batch size
    # unless the caller overrides.
    bs = batch_size or getattr(embedder, "batch_size", 32) or 32
    buffer: list[Chunk] = []
    total = 0
    # Per-source accumulator for the metadata store. Keyed by
    # source_path; value is (doc_type, topic, joined_text).
    sources: dict[str, tuple[str, str, list[str]]] = {}

    def _flush() -> None:
        nonlocal total
        if not buffer:
            return
        texts = [c.text for c in buffer]
        vecs = embedder.embed(texts)
        store.upsert_chunks(buffer, vecs)
        for c in buffer:
            entry = sources.get(c.source_path)
            if entry is None:
                sources[c.source_path] = (c.doc_type, c.topic, [c.text])
            else:
                entry[2].append(c.text)
        total += len(buffer)
        if progress:
            progress(total, -1)
        buffer.clear()

    def _record_sources() -> None:
        # P1 metadata: one row per unique source file. Cheap (one
        # row per file, not per chunk) and idempotent — re-ingest
        # just refreshes chunk_count + content_hash.
        if metadata is None or not sources:
            return
        for source_path, (doc_type, topic, texts) in sources.items():
            metadata.record_source(
                source_path=source_path,
                doc_type=doc_type,
                topic=topic,
                chunk_count=len(texts),
                content_hash=hash_text("\n\n".join(texts)),
            )
        log.info("metadata: recorded %d source rows", len(sources))

    try:
        for chunk in ingester.iter_chunks():
            buffer.append(chunk)
            if len(buffer) >= bs:
                _flush()
        _flush()
    except Exception:
        # A source failing mid-stream leaves the collection partially
        # updated. Record the sources that DID land so metadata matches
        # what is actually in the collection, then propagate.
        log.error(
            "ingest: aborted after %d chunks — collection %s is partially updated",
            total, store.collection,
        )
        _record_sources()
        raise
    finally:
        # The corpus changed (or may have) — a stale IDF map would silently
        # skew hybrid scores on the next query.
        invalidate_hybrid_cache()

    log.info("ingest: wrote %d chunks into %s", total, store.collection)
    _record_sources()

    return total


# -----------------------------------------------------------------------------
# Hybrid (BM25) retrieval helpers
# -----------------------------------------------------------------------------


def _tokenize_for_sparse(text: str) -> list[str]:
    """Simple tokenizer matching the one used in store.enable_hybrid()."""
    import re
    return re.findall(r"\w{2,}", text.lower())


def _build_query_sparse_vector(query: str, vocab: dict[str, int] | None, idf: dict[str, float] | None) -> "SparseVector | None":
    """Build a sparse vector for the query.

    Must use the same IDF weights as enable_hybrid() so scores are comparable.
    If vocab/idf are not provided, returns None (sparse not populated yet).
    """
    from collections import Counter
    from qdrant_client.http.models import SparseVector

    if vocab is None or idf is None:
        return None
    tokens = _tokenize_for_sparse(query)
    tf = Counter(tokens)
    if not tf:
        return None
    max_tf = max(tf.values()) or 1
    indices: list[int] = []
    values: list[float] = []
    for term, count in tf.items():
        if term in vocab:
            score = (count / max_tf) * idf.get(term, 0.0)
            if score > 0:
                indices.append(vocab[term])
                values.append(score)
    if not indices:
        return None
    return SparseVector(indices=indices, values=values)


# Cached corpus stats for hybrid search — built lazily on first hybrid query,
# guarded by _hybrid_lock so two FastAPI threadpool workers can't both scroll
# the corpus on a cold start. Invalidated after every ingest.
_hybrid_cache: dict = {"vocab": None, "idf": None, "built": False}
_hybrid_lock = threading.Lock()


def invalidate_hybrid_cache() -> None:
    """Drop the cached corpus IDF map so the next hybrid query rebuilds it.

    Must be called after anything that changes the corpus (ingest, delete):
    a stale IDF map silently shifts hybrid scores against the new corpus."""
    with _hybrid_lock:
        _hybrid_cache["vocab"] = None
        _hybrid_cache["idf"] = None
        _hybrid_cache["built"] = False


def _retrieve_hybrid(
    query: str,
    qvec: list[float],
    store: VectorStore,
    top_k: int,
    topic: str | None,
    dense_weight: float = 0.5,
) -> list[tuple]:
    """Dense + sparse hybrid search with RRF fusion.

    On first call, lazily builds the corpus IDF map by scrolling all chunks.
    Subsequent calls reuse the cached IDF (refreshes only if needed).
    """
    import math
    import re
    from collections import Counter

    from qdrant_client.http.models import SparseVector

    global _hybrid_cache

    # Lazy IDF build. The lock makes the check-then-build atomic: a second
    # worker arriving mid-build blocks here instead of scrolling the corpus
    # a second time.
    with _hybrid_lock:
        if not _hybrid_cache["built"]:
            log.info("hybrid: building corpus IDF map...")
            _TOKEN_RE = re.compile(r"\w{2,}")

            def tokenize(t: str) -> list[str]:
                return _TOKEN_RE.findall(t.lower())

            all_texts: list[tuple[str, str]] = list(store.iter_texts())

            N = len(all_texts)
            doc_freq: Counter = Counter()
            for _, text in all_texts:
                for tok in set(tokenize(text)):
                    doc_freq[tok] += 1

            _hybrid_cache["idf"] = {
                t: math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                for t, df in doc_freq.items()
            }
            _hybrid_cache["vocab"] = {t: i for i, t in enumerate(sorted(_hybrid_cache["idf"]))}
            _hybrid_cache["built"] = True
            log.info("hybrid: IDF map ready — vocab=%d terms", len(_hybrid_cache["vocab"]))

        # Snapshot under the lock. Invalidation replaces these dicts rather
        # than mutating them, so using the snapshots outside the lock is safe.
        vocab = _hybrid_cache["vocab"]
        idf = _hybrid_cache["idf"]
    query_sparse = _build_query_sparse_vector(query, vocab, idf)

    if query_sparse is None:
        log.warning("hybrid: no query terms found in vocabulary — falling back to dense")
        return store.search_dense(qvec, top_k=top_k)

    # Call the store's hybrid search.
    try:
        hits = store.search_hybrid(
            query_vector=qvec,
            query_sparse=query_sparse,
            top_k=top_k,
            dense_weight=dense_weight,
        )
    except Exception as e:
        log.warning("hybrid search failed (%s) — falling back to dense", e)
        return store.search_dense(qvec, top_k=top_k)

    # Filter by topic if requested (simple post-filter).
    if topic:
        hits = [(c, ds, hs) for c, ds, hs in hits if c.topic == topic]

    # Return as (chunk, score) tuples for compatibility with the rest of ask().
    return [(c, hs) for c, ds, hs in hits]


# -----------------------------------------------------------------------------
# Ask driver
# -----------------------------------------------------------------------------

@dataclass
class AskResult:
    answer: str
    citations: list[dict[str, Any]]   # post-rerank: [{n, source_path, section, chunk_id, topic, doc_type, score, text}]
    dense_hits: list[dict[str, Any]] = field(default_factory=list)
    # pre-rerank top-k from the store, so the eval can log dense-stage
    # recall@20 even when a reranker (P1) changes the post-rerank order.
    # Audit #10.


def ask(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    reranker: Reranker,
    generator: Generator,
    *,
    top_k_dense: int = 20,
    top_k_final: int = 5,
    topic: str | None = None,
    metadata: MetadataStore | None = None,
    hybrid: bool = False,
    dense_weight: float = 0.5,
) -> AskResult:
    """Full retrieve → rerank → build → generate pipeline.

    Args:
        hybrid: if True, use hybrid (dense + sparse BM25) search with RRF fusion.
            Requires sparse vectors to be populated first via `store.enable_hybrid()`
            or `rag ingest --populate-sparse`.
        dense_weight: dense-vs-sparse balance for hybrid search, passed through
            to `store.search_hybrid()`. Ignored unless `hybrid=True`.
    """
    # 1. Embed the query. Audit #9: use embed_query (with the model's
    #    retrieval instruction prefix) instead of plain embed. Documents
    #    were embedded without the prefix at ingest time, so the
    #    asymmetric space is what the model was trained for.
    qvec = embedder.embed_query(query)

    # 2. Retrieve top_k_dense from the store (or hybrid search).
    if hybrid:
        hits = _retrieve_hybrid(query, qvec, store, top_k=top_k_dense, topic=topic, dense_weight=dense_weight)
    else:
        hits = store.search_with_filter(qvec, top_k=top_k_dense, topic=topic) if topic \
            else store.search_dense(qvec, top_k=top_k_dense)
    if not hits:
        return AskResult(answer="(no results found in the index)", citations=[])

    # 3. Rerank (passthrough in P0).
    reranked = reranker.rerank(query, [c for c, _ in hits], top_k=top_k_final)
    # Keep the score alongside the chunk for transparency in the citation footer.
    score_by_id = {c.chunk_id: s for c, s in hits}
    chunks = reranked

    # 4. Build the prompt with [n]-style citations.
    prompt = build_prompt(query, chunks)

    # 5. Generate.
    answer = generator.generate(prompt)

    citations = [
        {
            "n": i + 1,
            "source_path": c.source_path,
            "section": c.section,
            "chunk_id": c.chunk_id,
            "topic": c.topic,
            "doc_type": c.doc_type,
            "score": round(score_by_id.get(c.chunk_id, 0.0), 4),
            # Include the chunk text so the faithfulness proxy can measure
            # answer↔source overlap. Previously this was missing and the
            # proxy silently fell back to source_path (audit #5).
            "text": c.text,
        }
        for i, c in enumerate(chunks)
    ]
    # Audit #10: surface pre-rerank dense hits so the eval can compare
    # dense-stage recall against post-rerank recall when a real reranker
    # lands in P1.
    dense_hits = [
        {
            "chunk_id": c.chunk_id,
            "source_path": c.source_path,
            "section": c.section,
            "score": round(float(s), 4),
        }
        for c, s in hits
    ]
    # P1 metadata: log this ask's citations. One row per [n] in the
    # answer, with the query, chunk_id, and source_path. Failures are
    # logged but never break the answer (a slow / locked DB should
    # not surface as a user-visible RAG error).
    if metadata is not None and citations:
        try:
            metadata.record_citations(query, citations)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("metadata: record_citations failed: %s", e)
    return AskResult(answer=answer, citations=citations, dense_hits=dense_hits)


# -----------------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below.
If a source supports a claim, cite it inline with [n] where n matches the
source number. If no source supports the claim, say so explicitly. Do not
invent citations.

Sources:
{sources}

Question: {query}

Answer:"""


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Format chunks into a numbered, source-anchored prompt block.

    Each source keeps its `source_path` and `section` so the model can ground
    its [n] markers. The chunk text is included verbatim.
    """
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.source_path} :: {c.section}"
        blocks.append(f"{header}\n{c.text}")
    return PROMPT_TEMPLATE.format(sources="\n\n".join(blocks), query=query)


def format_citation_footer(citations: list[dict[str, Any]]) -> str:
    """Human-readable citation list for the CLI footer."""
    if not citations:
        return "(no citations)"
    lines = ["--- citations ---"]
    for c in citations:
        lines.append(
            f"{c['n']}. {c['source_path']} :: {c['section']}  "
            f"(topic={c['topic']}, score={c['score']}, chunk={c['chunk_id'][:8]}…)"
        )
    return "\n".join(lines)
