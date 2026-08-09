"""Core ABCs for the RAG pipeline.

All swap points live here. Concrete implementations are in `providers/` and
`ingest/`. The pipeline and CLI depend ONLY on these abstractions, so swapping
an embedding model or generator is a `config.yaml` change.

Design contract:
- `Chunk` is the unit that flows everywhere: ingester → embedder → store → pipeline → generator.
- `Chunk.chunk_id` is deterministic (UUID5 over source/section/index) so re-ingest is idempotent.
- `Chunk.parent_id` is the doc-level id (one source_path → one parent_id).
- Payload fields mirror the Qdrant payload exactly so the store is a thin adapter.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator


# Stable namespace so chunk_ids are reproducible across machines.
CHUNK_NS = uuid.UUID("00000000-0000-0000-0000-000000000001")


def make_chunk_id(source_path: str, section: str, index: int) -> str:
    """Deterministic chunk id. Same inputs → same id, forever.

    The source_path is normalized to forward slashes internally so the
    same logical file hashes to the same id regardless of:
      - where the repo is cloned (absolute vs relative)
      - Windows vs Unix path separators
    The caller should still pass the ROOT-RELATIVE path (so two repos
    with different roots don't collide) — this normalization is just
    defensive against slash-style differences.
    """
    norm = source_path.replace("\\", "/")
    key = f"{norm}::{section}::{index}"
    return str(uuid.uuid5(CHUNK_NS, key))


def make_parent_id(source_path: str) -> str:
    """Deterministic doc-level id, one per source file.

    Same normalization rules as make_chunk_id."""
    norm = source_path.replace("\\", "/")
    return str(uuid.uuid5(CHUNK_NS, f"parent::{norm}"))


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrievable unit. The payload schema is mirrored in Qdrant."""
    chunk_id: str
    parent_id: str
    text: str
    source_path: str           # absolute or repo-relative path to the source file
    topic: str                 # e.g. "photography", "ml", "python" — from frontmatter or dir
    doc_type: str              # "markdown" | "code_python" | "zeal" | "pdf" | ...
    section: str               # heading (markdown) or function/class name (code) or page title
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Payload shape written to Qdrant. Generator reads this back."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "source_path": self.source_path,
            "topic": self.topic,
            "doc_type": self.doc_type,
            "section": self.section,
            **self.extra,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        """Reverse of to_payload — used by the pipeline after a Qdrant hit."""
        known = {"chunk_id", "parent_id", "text", "source_path", "topic", "doc_type", "section"}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(
            chunk_id=payload["chunk_id"],
            parent_id=payload["parent_id"],
            text=payload["text"],
            source_path=payload["source_path"],
            topic=payload["topic"],
            doc_type=payload["doc_type"],
            section=payload["section"],
            extra=extra,
        )


# -----------------------------------------------------------------------------
# ABCs
# -----------------------------------------------------------------------------

class Embedder(ABC):
    """Turns a list of strings into a list of fixed-dim dense vectors.

    Implementations must:
    - Return one vector per input string, in the same order.
    - Return lists of plain Python floats (so they're JSON-serializable for debugging).
    - Be deterministic for the same input + same model state.
    """

    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output vectors. Used to size the Qdrant collection."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed. Implementations should chunk internally if needed."""


class Reranker(ABC):
    """Re-orders a candidate list given a query.

    P0: PassthroughReranker (in core/pipeline.py) — returns the input unchanged.
    P1: BGE-reranker-v2 or similar cross-encoder.
    """

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return up to `top_k` chunks, best first."""


class Generator(ABC):
    """Generates a final answer from a prompt. Stateless w.r.t. conversation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model text. Prompt format is the caller's responsibility."""


class Ingester(ABC):
    """A source of chunks. Implementations know how to read one kind of corpus.

    P0: MarkdownDirIngester, ZealIngester.
    P1: PdfIngester (papers), GithubIngester, etc.
    """

    @abstractmethod
    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield chunks from the source. May be a generator — be lazy on big dirs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging, e.g. 'markdown_dir:/path/to/notes'."""
