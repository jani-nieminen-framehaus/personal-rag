"""config.yaml's sources: list — the record of what the index should contain."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.pipeline import configured_sources


def test_absent_or_empty_yields_nothing():
    assert configured_sources({}) == []
    assert configured_sources({"sources": None}) == []
    assert configured_sources({"sources": []}) == []


def test_entries_are_normalised(tmp_path):
    cfg = {"sources": [{"type": "markdown", "path": str(tmp_path)}]}
    got = configured_sources(cfg)
    assert got == [{"type": "markdown", "path": Path(tmp_path).resolve()}]


def test_unknown_type_is_rejected_by_name():
    with pytest.raises(ValueError, match="pdfs"):
        configured_sources({"sources": [{"type": "pdfs", "path": "/x"}]})


def test_missing_key_is_rejected():
    with pytest.raises(ValueError, match="path"):
        configured_sources({"sources": [{"type": "markdown"}]})


def test_non_mapping_entry_is_rejected():
    with pytest.raises(ValueError, match="mapping"):
        configured_sources({"sources": ["D:/notes"]})
