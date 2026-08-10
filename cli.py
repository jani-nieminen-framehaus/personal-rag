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

import service_state
from core import pipeline
from core.pipeline import (
    load_config,
    make_embedder,
    make_reranker,
    make_generator,
    make_store,
    make_metadata,
    chunking_params,
    format_citation_footer,
    ask as ask_pipeline,
    ingest as ingest_pipeline,
    DEFAULT_CONFIG_PATH,
)
from core.metadata import MetadataStore
from ingest.markdown_dir import MarkdownDirIngester
from ingest.zeal_docsets import ZealIngester
from ingest.pdf_dir import PdfDirIngester
from ingest.epub_dir import EpubDirIngester


# Persistent service state for the GUI. Path + helpers live in
# service_state.py (single source of truth, mirrored for PowerShell in
# scripts/_config.ps1). Module-level aliases kept for monkeypatchability.
SERVICE_STATE_DIR = service_state.STATE_DIR
SERVICE_STATE_FILE = service_state.STATE_FILE


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
@click.option("--hybrid", is_flag=True, help="Use hybrid (dense + sparse BM25) search.")
@click.pass_context
def ask(ctx, query, top_k_dense, top_k_final, topic, no_citations, as_json, hybrid):
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
        hybrid=hybrid,
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
@click.option("--epub", "epub_path", default=None, help="Path to a single .epub file OR a directory of EPUBs.")
@click.option("--recreate", is_flag=True, help="Drop and recreate the collection before ingest.")
@click.option("--batch-size", default=None, type=int, help="Override embedder batch size.")
@click.option("--populate-sparse", is_flag=True,
              help="After ingest, compute BM25 sparse vectors for hybrid search.")
@click.option("--yes", "-y", is_flag=True,
              help="Skip the confirmation prompt when using --recreate.")
@click.pass_context
def ingest(ctx, markdown_path, zeal_path, pdf_path, epub_path, recreate, batch_size, populate_sparse, yes):
    """Ingest Markdown notes, Zeal docsets, PDFs, and/or EPUBs into the index."""
    if not (markdown_path or zeal_path or pdf_path or epub_path):
        raise click.UsageError("pass at least one of --markdown, --zeal, --pdf, or --epub")

    cfg = ctx.obj["config"]
    cp = chunking_params(cfg)
    ing = cfg.get("ingest", {})

    embedder = make_embedder(cfg)
    store = make_store(cfg)
    metadata = make_metadata(cfg)
    total = 0

    if recreate and not yes:
        click.confirm(
            f"--recreate will DROP collection '{store.collection}' before "
            "re-ingesting; a failed ingest then leaves a partial index. Continue?",
            abort=True,
        )

    if markdown_path:
        mi = MarkdownDirIngester(
            root=markdown_path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            frontmatter_topic_key=ing.get("markdown", {}).get("frontmatter_topic_key", "topic"),
            max_chunks_per_doc=cp["max_chunks_per_doc"],
        )
        click.echo(f"ingesting markdown from {markdown_path} …")
        total += ingest_pipeline(
            mi, embedder, store,
            recreate=recreate, batch_size=batch_size, metadata=metadata,
        )

    if zeal_path:
        zi = ZealIngester(
            docset_path=zeal_path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
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
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            max_chunks_per_doc=cp["max_chunks_per_doc"],
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

    if epub_path:
        ei = EpubDirIngester(
            path=epub_path,
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            max_chunks_per_doc=cp["max_chunks_per_doc"],
        )
        click.echo(f"ingesting EPUB from {epub_path} …")
        # If this is the only source, --recreate is honored.
        rec = recreate if not (markdown_path or zeal_path or pdf_path) else False
        total += ingest_pipeline(
            ei, embedder, store,
            recreate=rec, batch_size=batch_size, metadata=metadata,
        )

    if metadata is not None:
        metadata.close()

    if populate_sparse:
        click.echo("populating sparse (BM25) vectors …")
        try:
            store.enable_hybrid()
            click.echo("sparse vectors populated.")
        except Exception as e:
            click.echo(f"error: sparse population failed: {e}", err=True)
            click.echo(
                f"wrote {total} chunks into {store.collection}, but the index is "
                "dense-only — hybrid queries will fall back to dense search. "
                "Re-run with --populate-sparse after fixing the error.",
                err=True,
            )
            sys.exit(1)

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
              help="Also run the (slow) LLM generation step to compute faithfulness. "
                   "Requires Ollama to be running with the configured generator model.")
@click.option("--nli/--no-nli", default=None,
              help="Use NLI cross-encoder (cross-encoder/nli-deberta-v3-xsmall) "
                   "for the faithfulness score. Falls back to token-overlap if unavailable. "
                   "Default: config eval.nli (true).")
@click.option("--sweep", is_flag=True,
              help="Grid over retrieval knobs; prints a results table.")
@click.pass_context
def eval(ctx, golden, as_json, with_faithfulness, nli, sweep):
    """Run the eval harness: recall@5, MRR, NLI faithfulness."""
    # Lazy import so the eval dependencies don't load on every command.
    from eval import run_ragas

    golden_path = _resolve_repo_path(golden)
    cfg = ctx.obj["config"]
    embedder = make_embedder(cfg)
    store = make_store(cfg)
    metadata = make_metadata(cfg)
    # Generator is only loaded when --with-faithfulness is set. By default
    # the eval is retrieval-only (no LLM call) so it's fast and works even
    # when Ollama isn't running.
    generator = make_generator(cfg) if with_faithfulness else None

    if nli is None:
        nli = (cfg.get("eval") or {}).get("nli", True)

    # NLI faithfulness scorer — loaded lazily; None if unavailable. Only
    # worth loading when there's a generator to score (faithfulness isn't
    # computed on retrieval-only runs), so a plain `rag eval` stays fast.
    nli_scorer = None
    if nli and generator is not None:
        try:
            from eval.nli_faithfulness import make_nli_faithfulness
            nli_scorer = make_nli_faithfulness(cfg)
            if nli_scorer is None:
                click.echo("nli: model unavailable — falling back to token-overlap proxy")
        except Exception as e:
            click.echo(f"nli: could not load scorer: {e} — skipping")

    if sweep:
        from eval import sweep as sweep_mod
        results = sweep_mod.run_sweep(
            golden_path, cfg, embedder=embedder, store=store,
            metadata=metadata, generator=generator, nli=nli_scorer)
        if metadata is not None:
            metadata.close()
        click.echo(sweep_mod.format_table(results))
        return

    reranker = make_reranker(cfg)

    metrics = run_ragas.run(
        golden_path=golden_path,
        embedder=embedder,
        store=store,
        reranker=reranker,
        generator=generator,
        top_k_dense=cfg.get("pipeline", {}).get("top_k_dense", 20),
        top_k_final=cfg.get("pipeline", {}).get("top_k_final", 5),
        metadata=metadata,
        nli_faithfulness=nli_scorer,
    )
    if metadata is not None:
        metadata.close()
    if as_json:
        click.echo(json.dumps(metrics, indent=2))
    else:
        run_ragas.print_report(metrics)


# -----------------------------------------------------------------------------
# rag golden generate / review / stats
# -----------------------------------------------------------------------------
#
# Curation flow for the golden eval set. `generate` drafts candidate rows
# over the live index (needs Ollama); `review` is an interactive y/n/e/q
# pass that promotes accepted candidates into golden_real.jsonl; `stats`
# reports counts. Both files are gitignored (personal corpus).
#
# NOTE: these two constants are deliberately NOT run through
# `_resolve_repo_path` — that helper falls back to a bare CWD-relative
# path when the target file doesn't exist yet, which is correct for an
# INPUT like --golden (read-only, must already exist) but wrong for an
# OUTPUT path like these: `rag golden generate` run from outside the repo
# root would then write to (or fail creating) the wrong `eval/` directory.
# Anchoring to DEFAULT_CONFIG_PATH.parent (the repo root) makes the
# output location independent of CWD.
GOLDEN_CANDIDATES = DEFAULT_CONFIG_PATH.parent / "eval" / "golden_candidates.jsonl"
GOLDEN_REAL = DEFAULT_CONFIG_PATH.parent / "eval" / "golden_real.jsonl"


@cli.group()
def golden():
    """Build and curate the golden question set over the LIVE index."""


@golden.command("generate")
@click.option("--n", default=100, type=int, help="Candidates to draft.")
@click.option("--topic", "topics", multiple=True, help="Restrict to topic(s). Repeatable.")
@click.option("--seed", default=1337, type=int, help="Sampling seed (determinism).")
@click.pass_context
def golden_generate(ctx, n, topics, seed):
    """Sample chunks from the live index and draft candidate questions (needs Ollama)."""
    from eval import golden_gen

    cfg = ctx.obj["config"]
    store = make_store(cfg)
    try:
        generator = make_generator(cfg)
    except Exception as e:
        click.echo(f"error: generator unavailable ({e}) — is Ollama running?", err=True)
        sys.exit(1)
    GOLDEN_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    written = golden_gen.generate_candidates(
        store, generator, GOLDEN_CANDIDATES, n=n, topics=list(topics) or None, seed=seed)
    click.echo(f"wrote {written} candidates to {GOLDEN_CANDIDATES}. Next: rag golden review")


@golden.command("review")
@click.pass_context
def golden_review_cmd(ctx):
    """Interactive y/n/e/q pass over pending candidates."""
    from eval import golden_review

    rows = golden_review.load_rows(GOLDEN_CANDIDATES)
    todo = golden_review.pending(rows)
    if not todo:
        click.echo("no pending candidates. Run: rag golden generate")
        return
    accepted: list[dict] = golden_review.load_rows(GOLDEN_REAL)
    done = 0
    # The atomic write in save_rows() only protects a single write; it does
    # NOT protect the review SESSION. Without this try/finally, a Ctrl-C
    # (KeyboardInterrupt — Click converts it to Abort -> "Aborted!" + exit 1)
    # or any other exception mid-loop would skip straight past both
    # save_rows() calls below and silently discard every y/n/e decision made
    # earlier in the pass, not just the in-flight one. Both saves now run on
    # every exit path (normal completion, "q", or an exception), and the
    # exception/interrupt is left to propagate afterward — never swallowed.
    try:
        for cand in todo:
            click.echo(f"\nQ: {cand['question']}")
            click.echo(f"   [{cand.get('topic', 'default')}] {cand.get('preview', '')}")
            choice = click.prompt("accept? [y]es / [n]o / [e]dit / [q]uit",
                                  type=click.Choice(["y", "n", "e", "q"]))
            if choice == "q":
                break
            edited = None
            if choice == "e":
                # Re-prompt until non-blank so apply_decision's ValueError
                # ("edit decision requires a non-empty question") can never
                # actually be raised from here. Ctrl-C is the "back out" —
                # it's still persisted by the try/finally above.
                while not edited or not edited.strip():
                    edited = click.prompt("edited question")
            _, golden_row = golden_review.apply_decision(cand, choice, edited=edited)
            if golden_row:
                accepted.append(golden_row)
            done += 1
    finally:
        golden_review.save_rows(GOLDEN_CANDIDATES, rows)          # statuses updated in place
        golden_review.save_rows(GOLDEN_REAL, accepted)
    click.echo(f"reviewed {done}; accepted total now {len(accepted)}")


@golden.command("stats")
@click.pass_context
def golden_stats(ctx):
    """Counts: pending / rejected / accepted (per topic)."""
    from eval import golden_review

    s = golden_review.stats(GOLDEN_CANDIDATES, GOLDEN_REAL)
    click.echo(json.dumps(s, indent=2))


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
    return service_state.read_state(path=SERVICE_STATE_FILE)


def _pid_alive(pid: int) -> bool:
    """Best-effort check: is the given PID still running? (service_state)"""
    return service_state.pid_alive(pid)


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
    # --no-browser needs no plumbing here: the non-Windows path never
    # opens a browser in the first place.
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


@cli.command("sessions")
@click.option("--limit", default=20, type=int, help="Max sessions to show (default 20).")
@click.option("--session", "session_id", default=None,
              help="Show turns for a specific session ID instead of listing sessions.")
@click.option("--json", "as_json", is_flag=True, help="Print as JSON.")
@click.pass_context
def sessions(ctx, limit, session_id, as_json):
    """List recent chat sessions, or show turns for a specific session.

    Use --session <id> to see the full conversation for one session.
    Session IDs are printed alongside each session in the list output.
    """
    cfg = ctx.obj["config"]
    md = _open_metadata(cfg)
    try:
        if session_id:
            rows = md.get_turns(session_id, limit=limit)
        else:
            rows = md.get_sessions(limit=limit)
    finally:
        md.close()

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        if session_id:
            click.echo(f"(no turns found for session {session_id})")
        else:
            click.echo("(no sessions yet — start chatting in the GUI first)")
        return

    if session_id:
        click.echo(f"session {session_id}  ({len(rows)} turn(s)):")
        for t in rows:
            q = t["query"]
            a = t["answer"]
            ts = (t["asked_at"] or "").replace("T", " ")[:19]
            click.echo(f"  [{ts}]")
            click.echo(f"    Q: {q}")
            # One-line preview of the answer
            preview = a.split("\n")[0]
            if len(preview) > 80:
                preview = preview[:77] + "…"
            click.echo(f"    A: {preview}")
            click.echo("")
    else:
        click.echo(f"{len(rows)} recent session(s):")
        for s in rows:
            ts = (s["updated_at"] or "").replace("T", " ")[:19]
            title = s["title"] or "(untitled)"
            n = s["turn_count"]
            click.echo(f"  {ts}  [{s['id']}]  {n} turn(s)  {title}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli(obj={})
