"""Desktop window for rag.

Three views — Ask, Ingest, Library — over the same pipeline the CLI and
the FastAPI server use. Models load once, on a worker thread. Ollama and
Qdrant stay separate processes.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.pipeline import (
    ask as ask_pipeline,
    chunking_params,
    configured_sources,
    ingest as ingest_pipeline,
    load_config,
    make_embedder,
    make_generator,
    make_metadata,
    make_reranker,
    make_store,
)
from ingest.epub_dir import EpubDirIngester
from ingest.markdown_dir import MarkdownDirIngester
from ingest.pdf_dir import PdfDirIngester


log = logging.getLogger("rag.gui")

_EXT_KIND = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "markdown",
}


def _topics(store) -> list[str]:
    seen: set[str] = set()
    for _pid, payload in store.iter_payloads():
        topic = (payload or {}).get("topic")
        if topic:
            seen.add(str(topic))
    return sorted(seen)


def _ingest_root(path: Path, kind: str, sources: list[dict]) -> Path:
    """Directory (or single file) the ingester should walk.

    A file under a configured source uses that source root, so chunk ids
    match `rag refresh`. A one-off markdown file uses its parent directory.
    A one-off PDF or EPUB is ingested as that file.
    """
    matches: list[Path] = []
    for entry in sources:
        if entry["type"] != kind:
            continue
        root = entry["path"]
        if root.is_file():
            if root == path:
                return root
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        matches.append(root)
    if matches:
        return max(matches, key=lambda p: len(p.parts))
    if kind == "markdown":
        return path.parent
    return path


def _ingest_files(paths: list[Path], stack: dict) -> dict:
    """Ingest the picked files in place.

    Same extension rules as serve.py POST /api/ingest. The original path
    is what gets stored. Returns chunk count and the names that were skipped.
    """
    cfg = stack["config"]
    sources = configured_sources(cfg)
    saved: list[str] = []
    skipped: list[str] = []
    groups: dict[tuple[str, Path], set[Path]] = {}
    for src in paths:
        kind = _EXT_KIND.get(src.suffix.lower())
        if kind is None:
            skipped.append(src.name)
            continue
        resolved = src.resolve()
        if not resolved.is_file():
            skipped.append(src.name)
            continue
        root = _ingest_root(resolved, kind, sources)
        groups.setdefault((kind, root), set()).add(resolved)
        saved.append(src.name)

    cp = chunking_params(cfg)
    ing = cfg.get("ingest", {}) or {}
    topic_key = (ing.get("markdown") or {}).get("frontmatter_topic_key", "topic")
    total = 0
    for (kind, root), only in groups.items():
        common = dict(
            target_tokens=cp["target_tokens"],
            overlap_pct=cp["overlap_pct"],
            min_chunk_tokens=cp["min_chunk_tokens"],
            default_topic=cp["default_topic"],
            max_chunks_per_doc=cp["max_chunks_per_doc"],
            only_paths=only,
        )
        if kind == "pdf":
            ingester = PdfDirIngester(path=root, **common)
        elif kind == "epub":
            ingester = EpubDirIngester(path=root, **common)
        elif kind == "markdown":
            ingester = MarkdownDirIngester(
                root=root, frontmatter_topic_key=topic_key, **common,
            )
        else:
            raise RuntimeError(f"unknown ingest kind: {kind}")
        total += ingest_pipeline(
            ingester, stack["embedder"], stack["store"],
            recreate=False, metadata=stack["metadata"],
        )
    return {"chunks": total, "files": saved, "skipped": skipped}


def _format_citations(citations: list[dict]) -> str:
    lines: list[str] = []
    for c in citations:
        lines.append(
            f"[{c.get('n')}] {c.get('source_path')} :: {c.get('section')}  "
            f"score={c.get('score')}"
        )
        text = (c.get("text") or "").strip()
        if text:
            lines.append(text[:500])
        lines.append("")
    return "\n".join(lines).rstrip()


class _LoadThread(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, config: dict | None, parent=None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        try:
            cfg = self._config if self._config is not None else load_config()
            embedder = make_embedder(cfg)
            store = make_store(cfg)
            reranker = make_reranker(cfg)
            generator = make_generator(cfg)
            metadata = make_metadata(cfg)
            try:
                topics = _topics(store)
            except Exception as exc:
                log.warning("topics: %s", exc)
                topics = []
            self.loaded.emit({
                "config": cfg,
                "embedder": embedder,
                "store": store,
                "reranker": reranker,
                "generator": generator,
                "metadata": metadata,
                "topics": topics,
            })
        except Exception as exc:
            log.exception("load failed")
            self.failed.emit(str(exc))


class _JobThread(QThread):
    """One ask or ingest. The window starts a new thread per job."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._fn())
        except Exception as exc:
            log.exception("job failed")
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.setWindowTitle("rag")
        self.resize(900, 640)
        self._config = config
        self._stack: dict | None = None
        self._load: _LoadThread | None = None
        self._job: _JobThread | None = None
        self._picked: list[Path] = []

        tabs = QTabWidget()
        tabs.addTab(self._build_ask(), "Ask")
        tabs.addTab(self._build_ingest(), "Ingest")
        tabs.addTab(self._build_library(), "Library")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Loading models…")
        self._set_ready(False)

    def _build_ask(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        row = QHBoxLayout()
        self.session = QComboBox()
        self.session.addItem("— no session —", "")
        self.session.currentIndexChanged.connect(self._on_session_changed)
        self.new_session_btn = QPushButton("New session")
        self.new_session_btn.clicked.connect(self._new_session)
        self.topic = QComboBox()
        self.topic.addItem("all topics", "")
        row.addWidget(QLabel("Session"))
        row.addWidget(self.session, 1)
        row.addWidget(self.new_session_btn)
        row.addWidget(QLabel("Topic"))
        row.addWidget(self.topic, 1)
        layout.addLayout(row)

        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Ask a question. Citations appear under the answer.")
        layout.addWidget(self.transcript, 1)

        ask_row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Ask the rag…")
        self.query.returnPressed.connect(self._ask)
        self.ask_btn = QPushButton("Ask")
        self.ask_btn.clicked.connect(self._ask)
        ask_row.addWidget(self.query, 1)
        ask_row.addWidget(self.ask_btn)
        layout.addLayout(ask_row)
        return page

    def _build_ingest(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Markdown, PDF, EPUB, or Python files."))
        self.file_list = QPlainTextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setPlaceholderText("No files selected.")
        layout.addWidget(self.file_list, 1)
        row = QHBoxLayout()
        self.pick_btn = QPushButton("Choose files")
        self.pick_btn.clicked.connect(self._pick_files)
        self.ingest_btn = QPushButton("Ingest")
        self.ingest_btn.clicked.connect(self._ingest)
        self.ingest_status = QLabel("")
        row.addWidget(self.pick_btn)
        row.addWidget(self.ingest_btn)
        row.addWidget(self.ingest_status, 1)
        layout.addLayout(row)
        return page

    def _build_library(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_library)
        layout.addWidget(self.refresh_btn)
        self.library = QPlainTextEdit()
        self.library.setReadOnly(True)
        layout.addWidget(self.library, 1)
        return page

    def start_load(self) -> None:
        self._load = _LoadThread(self._config, self)
        self._load.loaded.connect(self._on_loaded)
        self._load.failed.connect(self._on_load_failed)
        self._load.start()

    def _set_ready(self, ready: bool) -> None:
        # One job at a time. The worker is the only thread touching Qdrant
        # and the metadata connection until it finishes.
        for widget in (
            self.ask_btn,
            self.ingest_btn,
            self.query,
            self.session,
            self.topic,
            self.new_session_btn,
            self.pick_btn,
            self.refresh_btn,
        ):
            widget.setEnabled(ready)

    def _on_loaded(self, stack: object) -> None:
        self._stack = stack  # type: ignore[assignment]
        topics = self._stack.get("topics") or []
        self.topic.clear()
        self.topic.addItem("all topics", "")
        for topic in topics:
            self.topic.addItem(topic, topic)
        self._reload_sessions()
        self._refresh_library()
        self._set_ready(True)
        self.statusBar().showMessage("Ready")

    def _on_load_failed(self, message: str) -> None:
        self.statusBar().showMessage("Models failed to load")
        QMessageBox.critical(self, "rag", message)

    def _pipeline_cfg(self) -> dict:
        return (self._stack or {}).get("config", {}).get("pipeline") or {}

    def _reload_sessions(self, select: str | None = None) -> None:
        current = select if select is not None else self.session.currentData()
        self.session.blockSignals(True)
        self.session.clear()
        self.session.addItem("— no session —", "")
        md = (self._stack or {}).get("metadata")
        if md is not None:
            for row in md.get_sessions(limit=30):
                title = row.get("title") or row["id"]
                self.session.addItem(f"{title}  ({row.get('turn_count', 0)})", row["id"])
        idx = self.session.findData(current or "")
        self.session.setCurrentIndex(idx if idx >= 0 else 0)
        self.session.blockSignals(False)

    def _new_session(self) -> None:
        md = (self._stack or {}).get("metadata")
        if md is None:
            self.statusBar().showMessage("metadata is disabled")
            return
        sid = uuid.uuid4().hex[:12]
        md.create_session(sid, title="new")
        self._reload_sessions(select=sid)
        self.transcript.clear()

    def _on_session_changed(self) -> None:
        sid = self.session.currentData() or ""
        self.transcript.clear()
        md = (self._stack or {}).get("metadata")
        if not sid or md is None:
            return
        blocks: list[str] = []
        for turn in md.get_turns(sid):
            blocks.append(f"Q: {turn.get('query', '')}\n\n{turn.get('answer', '')}")
            cites = turn.get("citations") or []
            if isinstance(cites, list) and cites:
                blocks.append(_format_citations(cites))
            blocks.append("")
        self.transcript.setPlainText("\n".join(blocks).strip())

    def _ask(self) -> None:
        if not self._stack or self._busy():
            return
        query = self.query.text().strip()
        if not query:
            self.statusBar().showMessage("Query is empty")
            return
        topic = self.topic.currentData() or None
        sid = self.session.currentData() or None
        cfg = self._pipeline_cfg()
        stack = self._stack

        def work() -> dict:
            result = ask_pipeline(
                query,
                embedder=stack["embedder"],
                store=stack["store"],
                reranker=stack["reranker"],
                generator=stack["generator"],
                top_k_dense=cfg.get("top_k_dense", 20),
                top_k_final=cfg.get("top_k_final", 5),
                topic=topic,
                metadata=stack["metadata"],
                hybrid=cfg.get("hybrid", False),
                dense_weight=cfg.get("dense_weight", 0.5),
            )
            return {
                "query": query,
                "answer": result.answer,
                "citations": result.citations,
                "session_id": sid,
            }

        self._run_job(work, self._on_asked, "Asking…")

    def _on_asked(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        block = f"Q: {data.get('query', '')}\n\n{data.get('answer', '')}"
        cites = _format_citations(data.get("citations") or [])
        if cites:
            block = f"{block}\n\n{cites}"
        existing = self.transcript.toPlainText().strip()
        self.transcript.setPlainText(f"{existing}\n\n{block}".strip() if existing else block)
        self.query.clear()
        sid = data.get("session_id")
        md = (self._stack or {}).get("metadata")
        if sid and md is not None:
            md.record_turn(
                session_id=sid,
                query=data.get("query") or "",
                answer=data.get("answer") or "",
                citations=data.get("citations") or [],
            )
            self._reload_sessions(select=sid)
        self._refresh_library()
        self.statusBar().showMessage("Ready")

    def _pick_files(self) -> None:
        names, _selected = QFileDialog.getOpenFileNames(
            self,
            "Choose files",
            "",
            "Notes and books (*.md *.markdown *.pdf *.epub *.py)",
        )
        if not names:
            return
        self._picked = [Path(n) for n in names]
        self.file_list.setPlainText("\n".join(p.name for p in self._picked))
        self.ingest_status.setText(f"{len(self._picked)} file(s)")

    def _ingest(self) -> None:
        if not self._stack or self._busy() or not self._picked:
            if not self._picked:
                self.statusBar().showMessage("No files selected")
            return
        paths = list(self._picked)
        stack = self._stack

        def work() -> dict:
            result = _ingest_files(paths, stack)
            try:
                result["topics"] = _topics(stack["store"])
            except Exception as exc:
                log.warning("topics: %s", exc)
                result["topics"] = stack.get("topics") or []
            return result

        self._run_job(work, self._on_ingested, "Ingesting…")

    def _on_ingested(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.topic.clear()
        self.topic.addItem("all topics", "")
        for topic in data.get("topics") or []:
            self.topic.addItem(topic, topic)
        skipped = data.get("skipped") or []
        msg = f"Ingested {len(data.get('files') or [])} file(s), {data.get('chunks', 0)} chunks"
        if skipped:
            msg += f". Skipped: {', '.join(skipped)}"
        self.ingest_status.setText(msg)
        self._refresh_library()
        self.statusBar().showMessage("Ready")

    def _refresh_library(self) -> None:
        md = (self._stack or {}).get("metadata")
        if md is None:
            self.library.setPlainText("metadata is disabled in config.yaml")
            return
        stats = md.get_stats()
        lines = [
            f"sources {stats.get('total_sources', 0)}   "
            f"citations {stats.get('total_citations', 0)}   "
            f"eval runs {stats.get('total_eval_runs', 0)}",
        ]
        if stats.get("top_cited_source"):
            lines.append(
                f"most cited: {stats['top_cited_source']}  ({stats.get('top_cited_count', 0)})"
            )
        lines.append("")
        lines.append("Sources")
        for row in md.get_sources(limit=30):
            lines.append(
                f"  {row.get('source_path')}  topic={row.get('topic')}  "
                f"chunks={row.get('chunk_count')}"
            )
        lines.append("")
        lines.append("Citations")
        for row in md.get_citations(limit=20):
            q = (row.get("query") or "").strip()
            if len(q) > 80:
                q = q[:77] + "…"
            lines.append(f"  [{row.get('rank')}] {row.get('source_path')}  {q}")
        lines.append("")
        lines.append("Eval runs")
        for row in md.get_eval_runs(limit=10):
            lines.append(
                f"  {row.get('ran_at')}  recall@5={row.get('recall_at_5')}  "
                f"mrr={row.get('mrr')}"
            )
        self.library.setPlainText("\n".join(lines))

    def _busy(self) -> bool:
        return self._job is not None and self._job.isRunning()

    def _run_job(self, fn, on_ok, status: str) -> None:
        self._set_ready(False)
        self.statusBar().showMessage(status)
        job = _JobThread(fn, self)
        self._job = job
        job.finished_ok.connect(on_ok)
        job.failed.connect(self._on_job_failed)
        job.finished.connect(lambda: self._set_ready(self._stack is not None))
        job.start()

    def _on_job_failed(self, message: str) -> None:
        self.statusBar().showMessage("Failed")
        QMessageBox.warning(self, "rag", message)

    def closeEvent(self, event) -> None:
        for thread in (self._job, self._load):
            if thread is not None and thread.isRunning():
                thread.wait()
        event.accept()


def run(config: dict | None = None) -> None:
    """CLI entry: open the window and block until it closes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = QApplication([])
    window = MainWindow(config)
    window.show()
    window.start_load()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    run()
