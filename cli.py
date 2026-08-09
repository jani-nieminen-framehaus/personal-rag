"""CLI: `rag ask` / `rag ingest` / `rag eval`.

Run from the repo root:
    python cli.py ask "What is exposure compensation?"
    python cli.py ingest --markdown ./samples
    python cli.py eval

Click is the only third-party dep. No TUI, no GUI — terminal-first per spec.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import click

from core import pipeline
from core.pipeline import (
    load_config,
    make_embedder,
    make_reranker,
    make_generator,
    make_store,
    make_metadata,
    format_citation_footer,
    ask as ask_pipeline,
    ingest as ingest_pipeline,
    DEFAULT_CONFIG_PATH,
)
from core.metadata import MetadataStore
from ingest.markdown_dir import MarkdownDirIngester
from ingest.zeal_docsets import ZealIngester
from ingest.pdf_dir import PdfDirIngester


# Persistent service state for the GUI. Lives in the user's home so
# `rag status` works from any CWD. The server writes this on startup,
# clears it on clean shutdown. PIDs guard against stale state during
# restarts — we only clear on exit if the PID in the file is ours.
SERVICE_STATE_DIR = Path.home() / ".rag"
SERVICE_STATE_FILE = SERVICE_STATE_DIR / "state.json"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------------------------------------------------------
# root group
# -----------------------------------------------------------------------------

@click.group()
@click.option("-c", "--config", default=None, help="Path to config.yaml (default: ./config.yaml or $RAG_CONFIG).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
@click.pass_context
def cli(ctx, config, verbose):
    """Personal, multi-topic RAG. Terminal-first."""
    _setup_logging(verbose)
    cfg = load_config(config)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


# -----------------------------------------------------------------------------
# rag ask
# -----------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--top-k-dense", default=None, type=int, help="Override pipeline.top_k_dense.")
@click.option("--top-k-final", default=None, type=int, help="Override pipeline.top_k_final.")
@click.option("--topic", default=None, help="Restrict retrieval to a single topic.")
@click.option("--no-citations", is_flag=True, help="Print the answer only.")
@click.option("--json", "as_json", is_flag=True, help="Print a machine-readable AskResult.")
@click.pass_context
def ask(ctx, query, top_k_dense, top_k_final, topic, no_citations, as_json):
    """Ask a question over the knowledge base."""
    cfg = ctx.obj["config"]
    pip = cfg.get("pipeline", {})
    top_k_dense = top_k_dense or pip.get("top_k_dense", 20)
    top_k_final = top_k_final or pip.get("top_k_final", 5)

    embedder = make_embedder(cfg)
    store = make_store(cfg)
    reranker = make_reranker(cfg)
    generator = make_generator(cfg)
    metadata = make_metadata(cfg)

    result = ask_pipeline(
        query,
        embedder=embedder,
        store=store,
        reranker=reranker,
        generator=generator,
        top_k_dense=top_k_dense,
        top_k_final=top_k_final,
        topic=topic,
        metadata=metadata,
    )
    if metadata is not None:
        metadata.close()

    if as_json:
        click.echo(json.dumps({"answer": result.answer, "citations": result.citations}, indent=2))
        return

    click.echo(result.answer)
    if not no_citations:
        click.echo("")
        click.echo(format_citation_footer(result.citations))


# -----------------------------------------------------------------------------
# rag ingest
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--markdown", "markdown_path", default=None, help="Directory of .md files to ingest.")
@click.option("--zeal", "zeal_path", default=None, help="Path to a .docset directory.")
@click.option("--pdf", "pdf_path", default=None, help="Path to a single .pdf file OR a directory of PDFs.")
@click.option("--recreate", is_flag=True, help="Drop and recreate the collection before ingest.")
@click.option("--batch-size", default=None, type=int, help="Override embedder batch size.")
@click.pass_context
def ingest(ctx, markdown_path, zeal_path, pdf_path, recreate, batch_size):
    """Ingest Markdown notes, Zeal docsets, and/or PDFs into the index."""
    if not (markdown_path or zeal_path or pdf_path):
        raise click.UsageError("pass at least one of --markdown, --zeal, or --pdf")

    cfg = ctx.obj["config"]
    ch = cfg.get("chunking", {})
    ing = cfg.get("ingest", {})

    embedder = make_embedder(cfg)
    store = make_store(cfg)
    metadata = make_metadata(cfg)
    total = 0

    if markdown_path:
        mi = MarkdownDirIngester(
            root=markdown_path,
            target_tokens=ch.get("target_tokens", 768),
            overlap_pct=ch.get("overlap_pct", 12),
            min_chunk_tokens=ch.get("min_chunk_tokens", 32),
            default_topic=ing.get("default_topic", "default"),
            frontmatter_topic_key=ing.get("markdown", {}).get("frontmatter_topic_key", "topic"),
            max_chunks_per_doc=ch.get("max_chunks_per_doc", 2000),
        )
        click.echo(f"ingesting markdown from {markdown_path} …")
        total += ingest_pipeline(
            mi, embedder, store,
            recreate=recreate, batch_size=batch_size, metadata=metadata,
        )

    if zeal_path:
        zi = ZealIngester(
            docset_path=zeal_path,
            target_tokens=ch.get("target_tokens", 768),
            overlap_pct=ch.get("overlap_pct", 12),
            min_chunk_tokens=ch.get("min_chunk_tokens", 32),
            default_topic=ing.get("default_topic", "default"),
            sqlite_filename=ing.get("zeal", {}).get("sqlite_filename", "docSet.dsidx"),
            pages_dirname=ing.get("zeal", {}).get("pages_dirname", "Contents/Resources/Documents"),
        )
        click.echo(f"ingesting zeal docset at {zeal_path} …")
        # For Zeal we don't recreate on the second source — only the first call
        # should be allowed to recreate. Idempotent because of UUID5 ids.
        total += ingest_pipeline(
            zi, embedder, store,
            recreate=False, batch_size=batch_size, metadata=metadata,
        )

    if pdf_path:
        pi = PdfDirIngester(
            path=pdf_path,
            target_tokens=ch.get("target_tokens", 768),
            overlap_pct=ch.get("overlap_pct", 12),
            min_chunk_tokens=ch.get("min_chunk_tokens", 32),
            default_topic=ing.get("default_topic", "default"),
            max_chunks_per_doc=ch.get("max_chunks_per_doc", 2000),
        )
        click.echo(f"ingesting PDF from {pdf_path} …")
        # If this is the only source, --recreate is honored. If the
        # user chained with --markdown / --zeal, the collection
        # already exists and recreating would wipe their work.
        rec = recreate if not (markdown_path or zeal_path) else False
        total += ingest_pipeline(
            pi, embedder, store,
            recreate=rec, batch_size=batch_size, metadata=metadata,
        )

    if metadata is not None:
        metadata.close()
    click.echo(f"done. wrote {total} chunks into {store.collection}.")


# -----------------------------------------------------------------------------
# rag eval
# -----------------------------------------------------------------------------

def _resolve_repo_path(path_str: str) -> Path:
    """Resolve a path against multiple candidate roots.

    The user can be sitting in any directory when they invoke
    `rag eval` (especially when launched via rag.bat / rag.ps1 from
    somewhere else). The golden set lives at `eval/golden_set.jsonl`
    inside the repo, so we try in order:
      1. The literal path (absolute or relative-to-CWD)
      2. CWD + path_str
      3. The repo root (the directory containing config.yaml) + path_str

    Returns the resolved absolute path of the first match. Falls back
    to the literal if nothing is found, so the user sees the real
    "file not found" from the eval runner.

    This was the bug the audit called "rag eval golden path is CWD-
    relative while config is repo-anchored".
    """
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
    for root in (Path.cwd(), DEFAULT_CONFIG_PATH.parent):
        candidate = (root / path_str).resolve()
        if candidate.is_file():
            return candidate
    # Not found anywhere — return the literal so the user sees the real
    # "file not found" from the eval runner.
    return p


@cli.command()
@click.option("--golden", default="eval/golden_set.jsonl",
              help="Path to golden_set.jsonl. Resolved against CWD, then the repo root.")
@click.option("--json", "as_json", is_flag=True, help="Print metrics as JSON.")
@click.option("--with-faithfulness", is_flag=True,
              help="Also run the (slow) LLM generation step to compute the faithfulness proxy. "
                   "Requires Ollama to be running with the configured generator model.")
@click.pass_context
def eval(ctx, golden, as_json, with_faithfulness):
    """Run the eval harness: recall@5, MRR, naive faithfulness."""
    # Lazy import so the eval dependencies don't load on every command.
    from eval import run_ragas

    golden_path = _resolve_repo_path(golden)
    cfg = ctx.obj["config"]
    embedder = make_embedder(cfg)
    store = make_store(cfg)
    reranker = make_reranker(cfg)
    metadata = make_metadata(cfg)
    # Generator is only loaded when --with-faithfulness is set. By default
    # the eval is retrieval-only (no LLM call) so it's fast and works even
    # when Ollama isn't running.
    generator = make_generator(cfg) if with_faithfulness else None

    metrics = run_ragas.run(
        golden_path=golden_path,
        embedder=embedder,
        store=store,
        reranker=reranker,
        generator=generator,
        top_k_dense=cfg.get("pipeline", {}).get("top_k_dense", 20),
        top_k_final=cfg.get("pipeline", {}).get("top_k_final", 5),
        metadata=metadata,
    )
    if metadata is not None:
        metadata.close()
    if as_json:
        click.echo(json.dumps(metrics, indent=2))
    else:
        run_ragas.print_report(metrics)


# -----------------------------------------------------------------------------
# rag serve / rag status / rag url
# -----------------------------------------------------------------------------
#
# P1 GUI: the server runs as a long-lived process on a persistent port
# (default 8420). It writes its PID + URL to ~/.rag/state.json on
# startup so `rag status` and `rag url` can find it from any shell.
# The Task Scheduler task (created by scripts/install-service.ps1)
# starts the server at user logon.

def _read_service_state() -> dict | None:
    """Read the service state file. Returns None if absent or invalid."""
    if not SERVICE_STATE_FILE.is_file():
        return None
    try:
        return json.loads(SERVICE_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int) -> bool:
    """Best-effort check: is the given PID still running on this host?

    On Windows, `os.kill(pid, 0)` raises ValueError (the signal-0 trick
    is a Unix idiom). We fall back to the Win32 OpenProcess API.
    On Unix, signal 0 is the standard check.
    """
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        # Win32 OpenProcess. Returns 0 (NULL handle) if the process
        # doesn't exist or we don't have access. PROCESS_QUERY_LIMITED_
        # INFORMATION is enough to check existence without elevated rights.
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid))
            if handle == 0:
                return False
            try:
                # OpenProcess alone succeeds for a just-exited process whose
                # kernel object hasn't been reaped — ask for the exit code
                # to distinguish "running" from "zombie".
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    # Unix: signal 0 is the standard "is the process alive" check.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, we just don't own it
    except OSError:
        return False


@cli.command()
@click.option("--port", default=None, type=int, help="Port to bind (default 8420, or $RAG_PORT).")
@click.option("--host", default=None, help="Host to bind (default 127.0.0.1, or $RAG_HOST). Use 0.0.0.0 for LAN access.")
def serve(port, host):
    """Start the GUI server (FastAPI). Long-running; usually auto-launched."""
    if port is not None:
        os.environ["RAG_PORT"] = str(port)
    if host is not None:
        os.environ["RAG_HOST"] = host
    # The serve module reads RAG_PORT / RAG_HOST env at run().
    import serve as serve_mod
    serve_mod.run()


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of a human-readable line.")
def status(as_json):
    """Show the running rag GUI service: URL, PID, started_at, health."""
    state = _read_service_state()
    if not state:
        if as_json:
            click.echo(json.dumps({"running": False}, indent=2))
        else:
            click.echo("rag GUI: not running (no state file at ~/.rag/state.json)")
            click.echo("hint:  rag serve          # start it now")
            click.echo("       scripts\\install-service.ps1  # auto-launch on logon")
        return

    pid = state.get("pid")
    alive = _pid_alive(pid) if pid else False
    if as_json:
        click.echo(json.dumps({**state, "alive": alive}, indent=2))
    else:
        if alive:
            click.echo(f"rag GUI: running")
            click.echo(f"  url        : {state.get('url')}")
            if state.get("lan_url"):
                click.echo(f"  lan        : {state.get('lan_url')}")
            click.echo(f"  pid        : {pid}")
            click.echo(f"  started_at : {state.get('started_at')}")
        else:
            click.echo(f"rag GUI: state file says pid {pid} but no process is running")
            click.echo(f"  started_at : {state.get('started_at')}")
            click.echo("hint: rm ~/.rag/state.json  # clean up the stale state, then `rag serve`")


@cli.command()
def url():
    """Print the GUI URL to stdout (for piping into the browser). Exits 1 if not running."""
    state = _read_service_state()
    if not state or not _pid_alive(state.get("pid", 0)):
        click.echo("rag GUI not running", err=True)
        sys.exit(1)
    click.echo(state["url"])


@cli.command()
def tray():
    """Run the Windows system tray icon. Left-click opens the GUI."""
    import tray
    tray.run()


# -----------------------------------------------------------------------------
# rag start / rag open - boot orchestrator + browser launcher (P2)
# -----------------------------------------------------------------------------
#
# `rag start` is the "double-click to bring up the whole stack" entry point.
# On Windows it dispatches to scripts/start_all.ps1 which checks Ollama,
# Qdrant, and the rag server, starts the ones that aren't running, polls
# until /api/health says ready, then opens the browser. Idempotent: if
# everything is already up, it just opens the browser.
#
# `rag open` is a quieter version that only opens the browser to the
# currently-running GUI (errors if the server isn't up).

@cli.command("start")
@click.option("--no-browser", is_flag=True,
              help="Don't open the browser at the end (for headless / smoke).")
def start(no_browser):
    """Bring up the full rag stack (Ollama + Qdrant + rag server) and open the GUI.

    Idempotent. If everything is already running, just opens the browser.
    This is the one-click entry point pinned to your desktop.
    """
    repo = Path(__file__).resolve().parent
    if sys.platform == "win32":
        ps1 = repo / "scripts" / "start_all.ps1"
        if not ps1.is_file():
            click.echo(f"start: script missing: {ps1}", err=True)
            sys.exit(1)
        args = ["-ExecutionPolicy", "Bypass", "-File", str(ps1)]
        if no_browser:
            args += ["-NoBrowser"]
        rc = subprocess.call(["powershell.exe"] + args)
        sys.exit(rc)
    # Non-Windows fallback: just start the rag server (user manages
    # Ollama + Qdrant themselves).
    click.echo("start: non-Windows detected; only starting the rag server.")
    click.echo("start: you'll need to run Ollama and Qdrant separately.")
    if no_browser:
        os.environ["RAG_NO_BROWSER"] = "1"
    import serve as serve_mod
    serve_mod.run()


@cli.command("open")
def open_cmd():
    """Open the running rag GUI in your default browser. Exits 1 if not running."""
    state = _read_service_state()
    if not state or not _pid_alive(state.get("pid", 0)):
        click.echo("rag GUI not running. Try: rag start", err=True)
        sys.exit(1)
    url = state.get("url") or "http://localhost:8420"
    webbrowser.open(url)
    click.echo(f"opened {url}")


# -----------------------------------------------------------------------------
# rag citations / rag sources / rag stats / rag eval-runs
# -----------------------------------------------------------------------------
#
# All four read from the SQLite metadata store. They exit 1 with a friendly
# message if metadata is disabled in config.yaml.

def _open_metadata(cfg: dict) -> MetadataStore:
    """Open the metadata store, or raise click.UsageError if disabled."""
    meta_cfg = cfg.get("metadata", {}) or {}
    if not meta_cfg.get("enabled", True):
        raise click.UsageError(
            "metadata is disabled in config.yaml. Set `metadata.enabled: true` to use this command."
        )
    return MetadataStore(meta_cfg.get("path", "./metadata.sqlite3"))


@cli.command("citations")
@click.option("--limit", default=20, type=int, help="Max rows to show (default 20).")
@click.option("--source", "source_path", default=None, help="Filter to one source_path.")
@click.option("--json", "as_json", is_flag=True, help="Print as JSON.")
@click.pass_context
def citations(ctx, limit, source_path, as_json):
    """Show recent citations logged by `rag ask`."""
    cfg = ctx.obj["config"]
    md = _open_metadata(cfg)
    try:
        rows = md.get_citations(limit=limit, source_path=source_path)
    finally:
        md.close()
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("(no citations yet — run `rag ask` a few times first)")
        return
    click.echo(f"{len(rows)} recent citation(s):")
    for r in rows:
        q = (r["query"] or "").strip()
        if len(q) > 60:
            q = q[:57] + "…"
        click.echo(
            f"  [{r['rank']}] {r['source_path']}  ::  {q!r}  ({r['asked_at']})"
        )


@cli.command("sources")
@click.option("--limit", default=50, type=int, help="Max rows to show (default 50).")
@click.option("--json", "as_json", is_flag=True, help="Print as JSON.")
@click.pass_context
def sources(ctx, limit, as_json):
    """Show the catalog of files indexed by `rag ingest`."""
    cfg = ctx.obj["config"]
    md = _open_metadata(cfg)
    try:
        rows = md.get_sources(limit=limit)
    finally:
        md.close()
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("(no sources yet — run `rag ingest --markdown …` first)")
        return
    click.echo(f"{len(rows)} indexed source(s):")
    for r in rows:
        click.echo(
            f"  {r['ingested_at']}  topic={r['topic']:<14}  "
            f"chunks={r['chunk_count']:<5}  {r['source_path']}"
        )


@cli.command("stats")
@click.option("--json", "as_json", is_flag=True, help="Print as JSON.")
@click.pass_context
def stats(ctx, as_json):
    """Show overall counts + the most-cited source."""
    cfg = ctx.obj["config"]
    md = _open_metadata(cfg)
    try:
        s = md.get_stats()
    finally:
        md.close()
    if as_json:
        click.echo(json.dumps(s, indent=2))
        return
    click.echo("rag metadata stats:")
    click.echo(f"  total sources    : {s['total_sources']}")
    click.echo(f"  total citations  : {s['total_citations']}")
    click.echo(f"  total eval runs  : {s['total_eval_runs']}")
    if s["top_cited_source"]:
        click.echo(
            f"  most-cited source: {s['top_cited_source']}  ({s['top_cited_count']}×)"
        )
    else:
        click.echo("  most-cited source: (none yet)")


@cli.command("eval-runs")
@click.option("--limit", default=10, type=int, help="Max rows to show (default 10).")
@click.option("--json", "as_json", is_flag=True, help="Print as JSON.")
@click.pass_context
def eval_runs(ctx, limit, as_json):
    """Show recent eval runs (recall@5, MRR over time)."""
    cfg = ctx.obj["config"]
    md = _open_metadata(cfg)
    try:
        rows = md.get_eval_runs(limit=limit)
    finally:
        md.close()
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("(no eval runs logged — run `rag eval` first)")
        return
    click.echo(f"{len(rows)} recent eval run(s):")
    for r in rows:
        rd = r.get("recall_at_dense")
        rd_str = f"  recall@dense={rd:.2f}" if rd is not None else ""
        fp = r.get("faithfulness_proxy")
        fp_str = f"  faith={fp:.2f}" if fp is not None else ""
        click.echo(
            f"  {r['ran_at']}  n={r['n_questions']:<3}  "
            f"recall@5={r['recall_at_5']:.2f}  mrr={r['mrr']:.2f}{rd_str}{fp_str}"
        )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli(obj={})
