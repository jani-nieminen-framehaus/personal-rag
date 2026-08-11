"""The one file-enumeration function.

Extracted from three near-identical copies in ingest/{markdown,pdf,epub}_dir.py
(the original audit's cross-cutting finding #9). `rag refresh` needs to
enumerate a source's files WITHOUT instantiating its ingester, which is what
finally forced the extraction.
"""
from __future__ import annotations

from pathlib import Path


def iter_source_files(
    root: str | Path,
    suffixes: set[str],
    skip_hidden: bool = True,
    only_paths: set[Path] | None = None,
) -> list[Path]:
    """Every file under `root` whose suffix is in `suffixes`, sorted.

    Args:
        root: a directory to walk, or a single file (returned as-is if it
            matches the suffix filter).
        suffixes: lower-case suffixes including the dot, e.g. {".md", ".py"}.
        skip_hidden: skip anything under a dot-prefixed directory, and
            dot-prefixed files themselves.
        only_paths: when set, restrict the result to these paths. Entries that
            the walk would not have produced anyway are ignored, so a caller
            cannot smuggle in a file from outside `root` or of the wrong type.

    A missing root yields an empty list rather than raising: callers that care
    about the difference between "no files" and "root absent" must check the
    root themselves. `core/refresh.py` depends on making exactly that
    distinction before it prunes anything.
    """
    root = Path(root).resolve()
    wanted = {s.lower() for s in suffixes}

    if root.is_file():
        candidates = [root] if root.suffix.lower() in wanted else []
    elif root.is_dir():
        candidates = [
            p for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in wanted
            and not (
                skip_hidden
                and any(part.startswith(".") for part in p.relative_to(root).parts)
            )
        ]
    else:
        candidates = []

    if only_paths is not None:
        wanted_paths = {Path(p).resolve() for p in only_paths}
        candidates = [p for p in candidates if p.resolve() in wanted_paths]

    return sorted(candidates)
