# RAG Walking Skeleton — P0 Plan

> Status: P0 implemented and pushed. Three waves of bug fixes landed; see
> git log on `main` (commits `5880de9`, `f2259e9`, `d5b34fc`).
> Owner: Jani. Target: runnable P0 with end-to-end `rag ask` over personal notes + Zeal docsets.

## 1. Goal & non-goals

**P0 must demonstrate** (walking skeleton = thin slice through every layer):

- Multi-topic retrieval: personal notes (photography, ML, code) + Zeal docsets.
- Citations: every claim in the answer maps to a source chunk (file + section + chunk_id).
- Swappable models via `config.yaml` (Embedder / Reranker / Generator behind ABCs).
- Eval harness: `golden_set.jsonl` + `run_ragas.py` (recall@5, MRR) wired to the live index.
- Future-proof store: dense-only in P0, but Qdrant collection schema is shaped so BM25/sparse + RRF land in P1 with **no re-ingestion**.

**Explicitly out of P0** (will appear as stubs/placeholders only):

- Real reranker — `Reranker` ABC + `PassthroughReranker` only; `BGE-reranker-v2` is a stub file.
- Sparse / hybrid vectors — collection reserves the `sparse` named vector; not populated.
- PDF / paper ingesters — `Ingester` ABC designed for them; only Markdown + Zeal implemented.
- Multi-query, HyDE, sentence-window retrieval, re-ranking with cross-encoders, agent loops.
- Web UI, auth, multi-user, cloud vector DBs.

## 2. Architecture (one-screen view)

```
                         ┌──────────────────────┐
                         │       config.yaml    │
                         │  embedder/reranker/  │
                         │  generator/store/    │
                         │  chunking/paths      │
                         └──────────┬───────────┘
                                    │
   ┌────────────┐    ┌──────────────▼──────────────┐    ┌──────────────┐
   │ Ingesters  │───▶│            core/             │◀───│   CLI (cli)  │
   │ md_dir     │    │ interfaces / chunker /       │    │  rag ask     │
   │ zeal       │    │ pipeline                     │    │  rag ingest  │
   └─────┬──────┘    └──────────────┬───────────────┘    │  rag eval    │
         │                          │                    └──────┬───────┘
         ▼                          ▼                           │
   ┌────────────┐           ┌──────────────┐                    │
   │  Chunk     │           │  Embedder    │                    │
   │  (datacls) │──────────▶│  Qwen3-8B Q4 │                    │
   │  + meta    │           └──────┬───────┘                    │
   └────────────┘                  │                            │
                                   ▼                            │
                            ┌──────────────┐                    │
                            │   Qdrant     │◀───────────────────┘
                            │  kb_p0       │   query
                            │  dense vec   │   pipeline
                            │  + payload   │
                            └──────┬───────┘
                                   │ top-20 dense
                                   ▼
                            ┌──────────────┐
                            │  Reranker    │   ABC → Passthrough in P0
                            │  (P1 stub)   │
                            └──────┬───────┘
                                   │ top-5
                                   ▼
                            ┌──────────────┐
                            │  Generator   │   qwen3:30b-a3b via
                            │  prompt+[n]  │   Ollama OpenAI endpoint
                            └──────┬───────┘
                                   │ answer + [n] → footer citation list
                                   ▼
                                stdout
```

## 3. File-by-file responsibilities

| Path | Lines (est) | Responsibility |
| --- | --- | --- |
| `config.yaml` | ~60 | Single source of truth for models, paths, chunking params, Qdrant URL. No model names elsewhere. |
| `core/interfaces.py` | ~80 | `Embedder`, `Reranker`, `Generator`, `Ingester` ABCs + `Chunk` dataclass + `Citation` namedtuple. |
| `core/chunker.py` | ~180 | Markdown splitter (heading-aware, frontmatter-aware), Python AST splitter, fallback line-based splitter. Token counting via `tiktoken` cl100k. 12% overlap. Deterministic `chunk_id`. |
| `core/pipeline.py` | ~150 | `embed_chunks()`, `retrieve()`, `build_prompt()`, `generate_with_citations()`. Pure orchestration; no IO of its own except via injected deps. |
| `providers/embed_qwen3.py` | ~80 | `Qwen3Embedder` — sentence-transformers, Q4 via `BitsAndBytesConfig`, batched, normalized embeddings. |
| `providers/embed_bgem3.py` | ~30 | **Stub** — raises `NotImplementedError`; documents the dense+sparse plan for P1. |
| `providers/rerank_bge.py` | ~30 | **Stub** — same pattern. `PassthroughReranker` lives in `core/pipeline.py`. |
| `providers/llm_local.py` | ~70 | `LocalGenerator` — `openai.OpenAI` client pointed at Ollama (`/v1`), chat completions, system+user prompt. |
| `ingest/markdown_dir.py` | ~120 | `MarkdownDirIngester` — walks a root, reads frontmatter, splits into chunks, tags topic. |
| `ingest/zeal_docsets.py` | ~120 | `ZealIngester` — opens docset SQLite, reads `pages`, joins HTML files, strips to text, chunks. |
| `store/qdrant_store.py` | ~150 | `QdrantStore` — create collection (`dense` + reserved `sparse`), upsert, query. Cosine distance, payload includes chunk text. |
| `eval/golden_set.jsonl` | starter | 8–12 hand-written Q→chunk_id pairs against sample notes (so the harness runs end-to-end on day 1). |
| `eval/run_ragas.py` | ~120 | Custom recall@5 + MRR + a naive faithfulness heuristic (token overlap between answer and retrieved chunks). No Ragas framework. |
| `cli.py` | ~150 | `rag ask`, `rag ingest`, `rag eval` subcommands. `--verbose`, `--top-k-dense`, `--top-k-final`, `--topic`, `--no-citations`, `--json` flags. |
| `requirements.txt` | ~15 | Pinned: `qdrant-client`, `sentence-transformers`, `transformers`, `bitsandbytes`, `accelerate`, `torch`, `tiktoken`, `openai`, `pyyaml`, `python-frontmatter`, `beautifulsoup4`, `click`, `pytest` (dev). |
| `README.md` | big | Windows-native setup (Ollama + Qdrant binary), install, first ingest, first query, eval, troubleshooting. |

Total target: ~1.5k lines, mostly comments/docstrings.

## 4. Locked design decisions (defaults, overridable via config)

| Decision | Choice | Why |
| --- | --- | --- |
| Chunk IDs | UUID5 over `(source_path, section, chunk_index)` | Re-ingest is idempotent — no duplicate chunks, no orphan deletes needed for P0. |
| Qdrant collection | `kb_p0` | Single collection; named vectors `dense` (P0) and `sparse` (P1). |
| Distance | `COSINE` | Qwen3-Embedding is cosine-normalized; matches. |
| Payload fields | `chunk_id`, `parent_id`, `text`, `source_path`, `topic`, `doc_type`, `section`, `extra` (JSON) | Everything the generator needs to cite is in the payload — no second fetch. |
| Token counting | `tiktoken` cl100k | Decoupled from embedding model; ~5x faster at chunk-boundary checks. ±10% drift vs. Qwen tokenizer is fine. |
| Markdown topic resolution | YAML frontmatter `topic:` → parent dir → `default` | Notes already have frontmatter in your ecosystem; this is the lowest-friction path. |
| Zeal docset topic | `docset` name (e.g. `Python`, `QML`) | Zeal docsets are already topic-scoped. |
| Code chunking (P0) | **Python via `ast`**; other languages → whole-file as one chunk with a warning | Matches your actual stack (Python + TS/React). Keeps P0 small. P1 can add tree-sitter if needed. |
| Generator protocol | OpenAI Python client → `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint on Windows. No new SDK to learn. |
| Citation format | `[n]` in answer; footer `n. <source_path> :: <section> :: <chunk_id>` | Standard, copy-pasteable, easy to parse. |
| Eval | Custom (no Ragas framework despite filename) | You asked for explicit-over-magic; filename kept from the taxonomy. |
| Logging | stdlib `logging` + `--verbose` flag | No extra deps. |

## 5. Qdrant collection schema (P0 → P1 migration story)

```python
# store/qdrant_store.py — actual API used in the code
from qdrant_client.http.models import (
    VectorParams, SparseVectorParams, Distance, Modifier,
)

client.create_collection(
    collection_name="kb_p0",
    vectors_config={
        # Qwen3-Embedding-8B output dim; cosine distance
        "dense": VectorParams(size=4096, distance=Distance.COSINE),
    },
    # Placeholder slot; P1 populates with BM25 vectors via update_vectors
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
    },
)
```

In P0 we only populate `dense`. P1 adds BM25 `sparse` per point + a hybrid query (`<` vector name `=` `dense` `>` + `<` vector name `=` `sparse` `>`) with RRF — no schema change, no re-ingest of existing `dense` vectors needed because IDs are stable.

## 6. CLI shape

```
rag ask "what was said about exposure in photography"          # main path
rag ask "..." --top-k-dense 20 --top-k-final 5 --no-citations  # knobs
rag ingest --markdown ./notes --topic notes
rag ingest --zeal ~/.local/share/Zeal/Zeal/docsets/Python.docset
rag eval                                                    # recall@5 + MRR
rag eval --json                                             # machine-readable
```

`rag ask` output:

```
[1] Exposure compensation is useful when the meter is fooled by snow
[2] In landscape work, -1/3 EV is a common starting point for golden hour
---
1. notes/photography/exposure.md :: Golden hour :: c1b2...
2. notes/photography/exposure.md :: Metering modes :: c1b2...
```

## 7. Windows-native setup (preview — full version in README)

Three services, all on the Windows host. No WSL2, no Docker Desktop.

```powershell
# 1. Ollama (Windows installer from ollama.com/download)
ollama pull qwen3:30b-a3b

# 2. Qdrant — Windows binary, no Docker
Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip" -OutFile "$env:TEMP\qdrant.zip"
Expand-Archive "$env:TEMP\qdrant.zip" -DestinationPath "C:\Tools\qdrant"
& "C:\Tools\qdrant\qdrant.exe" --storage-snapshots-dir C:\qdrant\storage

# 3. The CLI
cd D:\Tinkering sideprojects\rag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Verification checklist (preview — full version at end of build)

1. `ollama list` shows `qwen3:30b-a3b`.
2. `curl http://localhost:6333/collections` returns `{"result":{"collections":[]}}`.
3. `python cli.py ingest --markdown ./samples` creates chunks, no errors.
4. `python cli.py ask "What is exposure compensation?"` returns answer with `[n]` markers and a footer.
5. `python eval/run_ragas.py` prints recall@5 + MRR.
6. Swap `embedder.model` in `config.yaml` to a different HF model → re-ingest still works without code changes.

## 9. Out of scope (P0)

Anything not in §1 "must demonstrate" — including but not limited to: streaming responses, conversation memory, agent/tool loops, structured output enforcement, multi-language code chunking beyond Python, image/multimodal content, any cloud deps.

---

## Open questions for you (before I start coding)

1. **Code chunking scope.** I want P0 to do Python via `ast` and treat other languages as whole-file (with a one-line warning). Acceptable, or do you want a basic regex-based split for TS/JS/Go too?
2. **Sample data.** I plan to create `samples/notes/` with ~5 short markdown files (photography, ML, code) so you can verify end-to-end on day 1 without your real notes. OK?
3. **First-run token budget.** Qwen3-Embedding-8B Q4 downloads ~5 GB. Default it, or check first?

Once you confirm those, I'll write the files in this order: `config.yaml` → `core/interfaces.py` → `core/chunker.py` → `store/qdrant_store.py` → `providers/embed_qwen3.py` → `providers/llm_local.py` → `ingest/markdown_dir.py` → `ingest/zeal_docsets.py` → `core/pipeline.py` → `cli.py` → `eval/run_ragas.py` → `eval/golden_set.jsonl` → `samples/` → `README.md` → `requirements.txt`.
