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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from core.interfaces import Chunk, Embedder, Reranker, Generator, Ingester
from store.qdrant_store import QdrantStore


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


def make_store(cfg: dict[str, Any]) -> QdrantStore:
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
    store: QdrantStore,
    *,
    recreate: bool = False,
    batch_size: int | None = None,
    progress: Callable[[int, int], None] | None = None,
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

    def _flush() -> None:
        nonlocal total
        if not buffer:
            return
        texts = [c.text for c in buffer]
        vecs = embedder.embed(texts)
        store.upsert_chunks(buffer, vecs)
        total += len(buffer)
        if progress:
            progress(total, -1)
        buffer.clear()

    for chunk in ingester.iter_chunks():
        buffer.append(chunk)
        if len(buffer) >= bs:
            _flush()
    _flush()
    log.info("ingest: wrote %d chunks into %s", total, store.collection)
    return total


# -----------------------------------------------------------------------------
# Ask driver
# -----------------------------------------------------------------------------

@dataclass
class AskResult:
    answer: str
    citations: list[dict[str, Any]]   # [{n, source_path, section, chunk_id, topic, doc_type}]


def ask(
    query: str,
    embedder: Embedder,
    store: QdrantStore,
    reranker: Reranker,
    generator: Generator,
    *,
    top_k_dense: int = 20,
    top_k_final: int = 5,
    topic: str | None = None,
) -> AskResult:
    """Full retrieve → rerank → build → generate pipeline."""
    # 1. Embed the query.
    qvec = embedder.embed([query])[0]

    # 2. Retrieve top_k_dense from the store.
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
    return AskResult(answer=answer, citations=citations)


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
