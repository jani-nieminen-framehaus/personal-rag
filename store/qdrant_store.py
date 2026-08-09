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

    def ensure_collection(
        self,
        recreate: bool = False,
        expected_dense_dim: int | None = None,
    ) -> None:
        """Create the collection if missing. Idempotent.

        P0 schema:
            vectors_config["dense"]   : the embedding model output (P0 only)
            sparse_vectors_config["sparse"] : placeholder, populated in P1

        Args:
            recreate: if True, drop and recreate (handy for dev wipes).
            expected_dense_dim: when the collection already exists, this is
                compared against the stored dense vector size. If they
                differ (e.g. you swapped the embedding model in config.yaml
                without `--recreate`), a clear ValueError is raised so you
                don't get a confusing error from the first upsert.

                Pass the embedder's `dim()` here. The pipeline does this
                for you via `ingest()` — call this directly only if you're
                building your own.
        """
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            log.warning("recreate=True — dropping collection %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            dim = expected_dense_dim if expected_dense_dim is not None else self.dense_dim
            log.info("creating collection %s (dense_dim=%d)", self.collection, dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": VectorParams(size=dim, distance=self.distance),
                },
                # Placeholder so P1 can populate without schema change.
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            return

        log.info("collection %s already exists", self.collection)
        # If the caller passed an expected dim, validate against the stored one.
        # Bug #7 fix: previously we never checked, so swapping embedders
        # without --recreate would fail at upsert time with a confusing
        # Qdrant-side error.
        if expected_dense_dim is not None:
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors.get("dense")  # type: ignore[union-attr]
            if existing is not None and existing.size != expected_dense_dim:
                raise ValueError(
                    f"collection {self.collection!r} has dense_dim={existing.size} "
                    f"but the configured embedder produces dim={expected_dense_dim}. "
                    f"Either change embedder.model in config.yaml to one that "
                    f"matches {existing.size}, or run `rag ingest --recreate` "
                    f"to drop and re-create the collection."
                )

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

        Bug #6 fix: qdrant-client 1.x returns `status` as a string enum
        ("completed"/"acknowledged"), so `int(result.status)` was raising
        ValueError. We pre-count the points and return that count instead.
        """
        flt = Filter(must=[models.FieldCondition(key="source_path", match=models.MatchValue(value=source_path))])
        # Count the points that will be deleted (the delete response
        # itself doesn't include a count).
        count_result = self.client.count(
            collection_name=self.collection,
            count_filter=flt,
        )
        n = int(getattr(count_result, "count", 0) or 0)
        if n > 0:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=flt),
            )
        return n
