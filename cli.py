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
import sys
from pathlib import Path

import click

from core import pipeline
from core.pipeline import (
    load_config,
    make_embedder,
    make_reranker,
    make_generator,
    make_store,
    format_citation_footer,
    ask as ask_pipeline,
    ingest as ingest_pipeline,
)
from ingest.markdown_dir import MarkdownDirIngester
from ingest.zeal_docsets import ZealIngester


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

    result = ask_pipeline(
        query,
        embedder=embedder,
        store=store,
        reranker=reranker,
        generator=generator,
        top_k_dense=top_k_dense,
        top_k_final=top_k_final,
        topic=topic,
    )

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
@click.option("--recreate", is_flag=True, help="Drop and recreate the collection before ingest.")
@click.option("--batch-size", default=None, type=int, help="Override embedder batch size.")
@click.pass_context
def ingest(ctx, markdown_path, zeal_path, recreate, batch_size):
    """Ingest Markdown notes and/or Zeal docsets into the index."""
    if not markdown_path and not zeal_path:
        raise click.UsageError("pass at least one of --markdown or --zeal")

    cfg = ctx.obj["config"]
    ch = cfg.get("chunking", {})
    ing = cfg.get("ingest", {})

    embedder = make_embedder(cfg)
    store = make_store(cfg)
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
        total += ingest_pipeline(mi, embedder, store, recreate=recreate, batch_size=batch_size)

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
        total += ingest_pipeline(zi, embedder, store, recreate=False, batch_size=batch_size)

    click.echo(f"done. wrote {total} chunks into {store.collection}.")


# -----------------------------------------------------------------------------
# rag eval
# -----------------------------------------------------------------------------

@cli.command()
@click.option("--golden", default="eval/golden_set.jsonl", help="Path to golden_set.jsonl.")
@click.option("--json", "as_json", is_flag=True, help="Print metrics as JSON.")
@click.option("--with-faithfulness", is_flag=True,
              help="Also run the (slow) LLM generation step to compute the faithfulness proxy. "
                   "Requires Ollama to be running with the configured generator model.")
@click.pass_context
def eval(ctx, golden, as_json, with_faithfulness):
    """Run the eval harness: recall@5, MRR, naive faithfulness."""
    # Lazy import so the eval dependencies don't load on every command.
    from eval import run_ragas

    cfg = ctx.obj["config"]
    embedder = make_embedder(cfg)
    store = make_store(cfg)
    reranker = make_reranker(cfg)
    # Generator is only loaded when --with-faithfulness is set. By default
    # the eval is retrieval-only (no LLM call) so it's fast and works even
    # when Ollama isn't running.
    generator = make_generator(cfg) if with_faithfulness else None

    metrics = run_ragas.run(
        golden_path=Path(golden),
        embedder=embedder,
        store=store,
        reranker=reranker,
        generator=generator,
        top_k_dense=cfg.get("pipeline", {}).get("top_k_dense", 20),
        top_k_final=cfg.get("pipeline", {}).get("top_k_final", 5),
    )
    if as_json:
        click.echo(json.dumps(metrics, indent=2))
    else:
        run_ragas.print_report(metrics)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli(obj={})
