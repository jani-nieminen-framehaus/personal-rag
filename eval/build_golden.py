"""Regenerate eval/golden_set.jsonl from the chunker + a questions manifest.

Why this exists
---------------
The chunker used to embed absolute paths into chunk_id, which meant the
golden set (which references chunks by id) had to be regenerated every
time the repo was cloned. We fixed the chunker so it uses root-relative,
forward-slash paths (bug #1 in the audit). This script is the matching
fix on the eval side: it derives the current chunk_ids from the live
chunking output, so the golden set can be regenerated at any time.

Inputs
------
- samples/notes/  : the source corpus (what we evaluate against)
- eval/questions.jsonl : hand-written questions with (path, section)
                         references. The human writes the manifest by
                         looking at the sample notes and saying "this
                         question should be answered by this section."

Output
------
- eval/golden_set.jsonl : the same schema as before — one JSON per line
                          with `question`, `relevant_chunk_ids`, and
                          optional `topic`. The chunk_ids are the
                          current ones, computed from the chunker.

When to run
-----------
- After any change to the sample notes
- After any change to the chunker logic
- After any change to eval/questions.jsonl
- After any change to chunking config (target/overlap/min_size)

It is safe to re-run at any time; the script is deterministic.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

# Make the project root importable. We deliberately do NOT import
# `core.pipeline` here because that pulls in `store.qdrant_store` which
# requires `qdrant_client`. This script is pure-Python: chunk + index.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.markdown_dir import MarkdownDirIngester


SAMPLES = ROOT / "samples" / "notes"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
CONFIG = ROOT / "config.yaml"


log = logging.getLogger("build_golden")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_chunking_config() -> dict:
    """Chunking params from config.yaml via the SAME resolver the live
    ingest paths use (core.pipeline.chunking_params) — the golden set must
    chunk exactly like the live corpus or recall silently drifts."""
    from core.pipeline import chunking_params
    with CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return chunking_params(cfg)


def build_index():
    """Walk samples/ and build a (rel_path, section) -> list[chunk_id] map.

    A single section may produce multiple chunks (when the body exceeds
    the target token count and gets token-split). The map is a list so
    the caller can pick the first chunk_id, or all of them for a more
    thorough eval.
    """
    cp = _load_chunking_config()

    mi = MarkdownDirIngester(
        root=SAMPLES,
        target_tokens=cp["target_tokens"],
        overlap_pct=cp["overlap_pct"],
        min_chunk_tokens=cp["min_chunk_tokens"],
        default_topic=cp["default_topic"],
    )

    index: dict[tuple[str, str], list[str]] = {}
    for chunk in mi.iter_chunks():
        rel_path = Path(chunk.source_path).relative_to(SAMPLES).as_posix()
        key = (rel_path, chunk.section)
        index.setdefault(key, []).append(chunk.chunk_id)
    return index


def main() -> int:
    if not SAMPLES.is_dir():
        log.error("samples dir not found: %s", SAMPLES)
        return 1
    if not QUESTIONS.is_file():
        log.error("questions manifest not found: %s", QUESTIONS)
        return 1

    log.info("indexing samples from %s", SAMPLES)
    index = build_index()
    log.info("  %d unique (path, section) keys", len(index))

    out_lines: list[str] = []
    missing: list[tuple[str, tuple[str, str]]] = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q = json.loads(line)
        rel_ids: list[str] = []
        for rel in q.get("relevant", []):
            key = (rel["path"], rel["section"])
            if key not in index:
                missing.append((q["question"], key))
                continue
            # Take all chunk_ids for this (path, section). When the section
            # was token-split into multiple pieces, we want any of them to
            # count as a hit — recall@5 is more forgiving, MRR still works.
            rel_ids.extend(index[key])

        entry: dict = {
            "question": q["question"],
            "relevant_chunk_ids": rel_ids,
        }
        if "topic" in q:
            entry["topic"] = q["topic"]
        out_lines.append(json.dumps(entry, ensure_ascii=False))

    if missing:
        log.warning("missing chunk_ids for %d references:", len(missing))
        for question, key in missing:
            log.warning("  %r  ->  %s", question, key)
        log.warning("(check eval/questions.jsonl — paths/sections may have drifted)")

    GOLDEN.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log.info("wrote %s  (%d questions)", GOLDEN, len(out_lines))
    if missing:
        # The file is still written (useful for inspecting the drift), but a
        # golden set with dangling references must fail the build — otherwise
        # CI quietly runs evals against a shrunken question set.
        log.error("golden set has %d drifted references — exiting 1", len(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
