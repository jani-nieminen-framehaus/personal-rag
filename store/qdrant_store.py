"""Qdrant adapter.

P0 uses a single named dense vector (`dense`). The collection also declares a
`SparseVectorParams` slot named `sparse` at creation time so P1 can populate it
with BM25 sparse vectors without re-creating the collection or re-ingesting
the existing dense embeddings. The P1 hybrid path is implemented:
`enable_hybrid()` populates TF-IDF sparse vectors, `search_hybrid()` fuses
dense + sparse results with weighted reciprocal-rank fusion.

All upserts use deterministic UUID5 ids (see core.interfaces.make_chunk_id),
so re-running an ingest over the same source paths updates points in place
rather than duplicating them.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
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
    SparseVector,
)

from core.interfaces import Chunk, VectorStore


log = logging.getLogger(__name__)


class QdrantStore(VectorStore):
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

    def iter_payloads(self):
        """Yield (point_id, payload) for every point, scrolling in batches."""
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                yield p.id, (p.payload or {})
            if offset is None:
                break

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
        """Populate the sparse slot with TF-IDF (≈BM25) vectors.

        Algorithm:
          1. Scroll all points, extract chunk text from payload.
          2. Build corpus: all chunk texts → vocabulary + document frequencies.
          3. Compute IDF for each vocabulary term (standard Lucene formula).
          4. For each chunk: tokenize, compute TF-IDF per term, store as a
             sparse vector in Qdrant via update_vectors.

        The collection schema already has the sparse slot declared — no
        recreate needed. Safe to re-run; update_vectors is idempotent.

        After calling this, search_hybrid() becomes available.
        """
        log.info("enable_hybrid: scrolling all points to build corpus vocabulary...")
        # Step 1: collect all chunk texts.
        chunk_texts: list[tuple[str, str]] = list(self.iter_texts())

        if not chunk_texts:
            log.warning("enable_hybrid: no chunks found in collection")
            return

        log.info("enable_hybrid: building vocabulary from %d chunks", len(chunk_texts))

        # Step 2: build vocabulary and document frequencies.
        # Simple whitespace+punctuation tokenizer.
        _TOKEN_RE = re.compile(r"\w{2,}")  # ≥2-char words, strips noise

        def tokenize(text: str) -> list[str]:
            return _TOKEN_RE.findall(text.lower())

        doc_freq: Counter[str] = Counter()
        for _, text in chunk_texts:
            tokens = set(tokenize(text))
            for t in tokens:
                doc_freq[t] += 1

        N = len(chunk_texts)
        # IDF using the Lucene formula (smoothed).
        idf: dict[str, float] = {
            term: math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }
        # Build a sorted vocab so indices are deterministic.
        vocab = sorted(idf.keys())
        term_to_idx: dict[str, int] = {t: i for i, t in enumerate(vocab)}
        log.info(
            "enable_hybrid: vocabulary size=%d, N=%d, avg_df=%.1f",
            len(vocab), N, sum(doc_freq.values()) / N,
        )

        # Step 3: compute sparse vectors for each chunk.
        log.info("enable_hybrid: computing sparse vectors for %d chunks...", len(chunk_texts))
        from qdrant_client.http.models import PointVectors

        batch: list[PointVectors] = []
        for chunk_id, text in chunk_texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            # Build sparse vector: indices + BM25-like TF-IDF values.
            # Use TF * IDF with k1=0 (binary-like, just IDF weighting).
            # This gives a pure IDF-weighted term vector.
            # max_tf is loop-invariant per chunk — normalising by it gives
            # BM25-like behaviour without a k1 param.
            max_tf = max(tf.values()) if tf else 1
            indices: list[int] = []
            values: list[float] = []
            for term, count in tf.items():
                if term in term_to_idx:
                    score = (count / max_tf) * idf.get(term, 0.0)
                    if score > 0:
                        indices.append(term_to_idx[term])
                        values.append(score)

            if not indices:
                # Empty/skip — give it a zero vector to avoid Qdrant errors.
                indices = [0]
                values = [0.0]

            batch.append(
                PointVectors(
                    id=chunk_id,
                    vector={"sparse": SparseVector(indices=indices, values=values)},
                )
            )

        # Step 4: upsert all sparse vectors.
        self.client.update_vectors(
            collection_name=self.collection,
            points=batch,
            wait=True,
        )
        log.info(
            "enable_hybrid: done — %d sparse vectors written to %s",
            len(batch), self.collection,
        )

    def search_hybrid(
        self,
        query_vector: list[float],
        query_sparse: SparseVector,
        top_k: int = 20,
        dense_weight: float = 0.5,
    ) -> list[tuple[Chunk, float, float]]:
        """Hybrid search: weighted reciprocal-rank fusion of dense and sparse.

        Rank-based fusion (not score-based), so the dense cosine scores and
        the sparse TF-IDF scores — which live on completely different scales —
        can never dominate each other through raw magnitude.

        Args:
            query_vector: the dense query embedding.
            query_sparse: a pre-built SparseVector for the query text.
            top_k: number of candidates to retrieve from each branch.
            dense_weight: balance between dense and sparse (0.0-1.0).
                0.5 = equal weight. Higher = more dense. Lower = more sparse.

        Returns (chunk, dense_score, hybrid_score) tuples best-first. The
        hybrid_score is the fused RRF value (max ~1/(k+1), i.e. ~0.016).
        """
        # Retrieve from both branches independently.
        dense_hits = self.search_dense(query_vector, top_k=top_k)
        sparse_hits = self._search_sparse(query_sparse, top_k=top_k)

        dense_scores: dict[str, float] = {c.chunk_id: s for c, s in dense_hits}

        # Weighted RRF: score(c) = w/(k+rank_dense) + (1-w)/(k+rank_sparse),
        # with a branch contributing 0 when the chunk is absent from it.
        k = 60  # standard RRF constant
        dense_rank = {c.chunk_id: r for r, (c, _) in enumerate(dense_hits, start=1)}
        sparse_rank = {c.chunk_id: r for r, (c, _) in enumerate(sparse_hits, start=1)}

        rrf_scores: dict[str, float] = {}
        for cid in set(dense_rank) | set(sparse_rank):
            d = dense_rank.get(cid)
            s = sparse_rank.get(cid)
            rrf_scores[cid] = (
                dense_weight * (1.0 / (k + d) if d else 0.0)
                + (1 - dense_weight) * (1.0 / (k + s) if s else 0.0)
            )

        # Return best-first by fused score.
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        chunk_by_id = {c.chunk_id: c for c, _ in dense_hits}
        chunk_by_id.update({c.chunk_id: c for c, _ in sparse_hits})

        out: list[tuple[Chunk, float, float]] = []
        for cid in sorted_ids[:top_k]:
            c = chunk_by_id.get(cid)
            if c:
                out.append((c, dense_scores.get(cid, 0.0), rrf_scores[cid]))
        return out

    def _search_sparse(self, query: SparseVector, top_k: int) -> list[tuple[Chunk, float]]:
        """Search using the sparse vector slot. Returns (chunk, score) pairs."""
        try:
            hits = self.client.query_points(
                collection_name=self.collection,
                query=query,
                using="sparse",
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            # Sparse not populated yet.
            return []
        out: list[tuple[Chunk, float]] = []
        for h in hits:
            payload = h.payload or {}
            if not all(k in payload for k in ("chunk_id", "text", "source_path", "topic", "doc_type", "section")):
                continue
            out.append((Chunk.from_payload(payload), float(h.score)))
        return out

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
