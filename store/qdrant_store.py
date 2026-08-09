"""Qdrant adapter.

P0 uses a single named dense vector (`dense`). The collection also declares a
`SparseVectorParams` slot named `sparse` at creation time so P1 can populate it
with BM25 sparse vectors without re-creating the collection or re-ingesting
the existing dense embeddings. P1 hybrid query path is documented in
`enable_hybrid()` below — it's a no-op in P0.

All upserts use deterministic UUID5 ids (see core.interfaces.make_chunk_id),
so re-running an ingest over the same source paths updates points in place
rather than duplicating them.
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PointStruct,
    Filter,
)

from core.interfaces import Chunk


log = logging.getLogger(__name__)


class QdrantStore:
    """Thin adapter over the Qdrant HTTP/gRPC client.

    The class is intentionally simple — collection management, upsert, and
    dense-only query. The store does not know about embeddings or the LLM.
    """

    def __init__(self, url: str, collection: str, dense_dim: int, distance: str = "COSINE"):
        self.url = url
        self.collection = collection
        self.dense_dim = dense_dim
        self.distance = Distance[distance]
        self.client = QdrantClient(url=url, timeout=60)

    # -- collection management ------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if missing. Idempotent.

        P0 schema:
            vectors_config["dense"]   : the embedding model output (P0 only)
            sparse_vectors_config["sparse"] : placeholder, populated in P1

        Args:
            recreate: if True, drop and recreate (handy for dev wipes).
        """
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            log.warning("recreate=True — dropping collection %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            log.info("creating collection %s (dense_dim=%d)", self.collection, self.dense_dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": VectorParams(size=self.dense_dim, distance=self.distance),
                },
                # Placeholder so P1 can populate without schema change.
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
        else:
            log.info("collection %s already exists", self.collection)

    def count(self) -> int:
        """Number of points currently in the collection."""
        info = self.client.get_collection(self.collection)
        # `points_count` lives on the info in 1.10+
        return int(getattr(info, "points_count", 0) or 0)

    # -- writes ---------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert (chunk, dense_vector) pairs. Returns count written.

        Idempotent: deterministic ids mean re-runs overwrite cleanly.
        Qdrant batches in chunks of 64 by default — we keep it explicit.
        """
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")

        points = [
            PointStruct(
                id=c.chunk_id,
                vector={"dense": v},
                payload=c.to_payload(),
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )
        return len(points)

    # -- reads ----------------------------------------------------------------

    def search_dense(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Return (chunk, score) pairs, best first."""
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points
        out: list[tuple[Chunk, float]] = []
        for h in hits:
            payload = h.payload or {}
            # Required fields; if missing, skip (data corruption / older schema).
            if not all(k in payload for k in ("chunk_id", "text", "source_path", "topic", "doc_type", "section")):
                log.warning("skipping hit with missing payload keys: %s", list(payload.keys()))
                continue
            out.append((Chunk.from_payload(payload), float(h.score)))
        return out

    def search_with_filter(
        self, vector: list[float], top_k: int, topic: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Same as search_dense, optionally constrained to one topic."""
        qfilter = None
        if topic:
            qfilter = Filter(must=[models.FieldCondition(key="topic", match=models.MatchValue(value=topic))])
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using="dense",
            limit=top_k,
            with_payload=True,
            query_filter=qfilter,
        ).points
        out: list[tuple[Chunk, float]] = []
        for h in hits:
            payload = h.payload or {}
            if not all(k in payload for k in ("chunk_id", "text", "source_path", "topic", "doc_type", "section")):
                continue
            out.append((Chunk.from_payload(payload), float(h.score)))
        return out

    # -- P1 migration path ----------------------------------------------------

    def enable_hybrid(self) -> None:
        """No-op in P0.

        P1 will:
          1. iterate all points, compute BM25 sparse vectors for each chunk's
             text, and call update_vectors(using='sparse', ...).
          2. switch retrieve_dense to a hybrid query with RRF fusion
             (see core.pipeline.retrieve_hybrid).
        The collection schema already supports it — no recreate needed.
        """
        log.info("enable_hybrid() is a no-op in P0 — wired for P1")

    # -- ops ------------------------------------------------------------------

    def delete_by_source(self, source_path: str) -> int:
        """Remove every chunk that came from a given source file. Returns count.

        Used by `rag ingest` to support `ingest --replace <path>` in the future.
        For P0 the CLI doesn't expose it; it's here so the surface is complete.
        """
        result = self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=Filter(must=[models.FieldCondition(key="source_path", match=models.MatchValue(value=source_path))])
            ),
        )
        return int(result.status or 0)  # type: ignore[attr-defined]
