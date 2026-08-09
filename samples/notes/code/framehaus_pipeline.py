"""Pipeline helpers used across the Framehaus apps.

This is a short, intentionally trivial Python file so the chunker's AST path
has something real to chew on when you run the demo.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


log = logging.getLogger(__name__)


@dataclass
class Client:
    id: int
    name: str
    email: str


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database with row-factory enabled.

    The caller owns the connection — close it with `.close()` or use as a
    context manager (`.close()` is called automatically on exit).
    """
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_clients(conn: sqlite3.Connection, *, limit: int = 100) -> list[Client]:
    """Return up to `limit` clients ordered by id."""
    cur = conn.execute("SELECT id, name, email FROM clients ORDER BY id LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [Client(id=r["id"], name=r["name"], email=r["email"]) for r in rows]


def upsert_client(conn: sqlite3.Connection, client: Client) -> int:
    """Insert or update a client. Returns the affected row id."""
    cur = conn.execute(
        """
        INSERT INTO clients (name, email) VALUES (?, ?)
        ON CONFLICT(email) DO UPDATE SET name=excluded.name
        """,
        (client.name, client.email),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def iter_clients_jsonl(conn: sqlite3.Connection) -> Iterable[str]:
    """Stream clients as JSONL — used by the Mac bridge to export prospect data."""
    import json
    for row in conn.execute("SELECT id, name, email FROM clients"):
        yield json.dumps({"id": row["id"], "name": row["name"], "email": row["email"]})
