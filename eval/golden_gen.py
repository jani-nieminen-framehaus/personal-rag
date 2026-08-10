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
