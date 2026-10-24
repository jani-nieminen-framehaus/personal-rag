#!/usr/bin/env bash
# =============================================================================
# start_all.sh — rag boot orchestrator (Linux / macOS).
#
# Idempotent. Checks Ollama and Qdrant and starts the ones that are down.
# Then opens the desktop window (`python -m cli gui`). The window loads
# the models itself, so this path does not also start `rag serve`.
# `--no-browser` starts the FastAPI server instead and does not open
# the window.
#
# Qdrant's URL comes from config.yaml (store.url), not a hardcoded port.
#
# Configuration (overridable via env vars):
#   RAG_REPO          path to the rag repo (default: parent of this script)
#   RAG_VENV          path to the venv (default: $RAG_REPO/.venv)
#   RAG_PYTHON        python inside the venv (default: $RAG_VENV/bin/python)
#   RAG_HOST          GUI bind host (default: 127.0.0.1)
#   RAG_PORT          GUI bind port (default: 8420)
#   OLLAMA_URL        Ollama health probe (default: http://localhost:11434/api/tags)
#   OLLAMA_BIN        ollama binary (default: ollama on PATH)
#   QDRANT_BIN        qdrant binary (default: qdrant on PATH)
#   WAIT_TIMEOUT_S    max seconds to wait for the rag server (default: 120)
# =============================================================================
set -euo pipefail

NO_BROWSER=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-browser) NO_BROWSER=1 ;;
        *)
            echo "start_all: unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RAG_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${RAG_VENV:-$REPO_ROOT/.venv}"
PYTHON="${RAG_PYTHON:-$VENV_DIR/bin/python}"
BIND_HOST="${RAG_HOST:-127.0.0.1}"
PORT="${RAG_PORT:-8420}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/api/tags}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
QDRANT_BIN="${QDRANT_BIN:-qdrant}"
WAIT_TIMEOUT="${WAIT_TIMEOUT_S:-120}"
GUI_URL="http://127.0.0.1:${PORT}"
STATE_FILE="${HOME}/.rag/state.json"
LOG_DIR="${HOME}/.rag"
QDRANT_STORAGE="${XDG_DATA_HOME:-$HOME/.local/share}/rag/qdrant"

cd "$REPO_ROOT"

stage() {
    echo ""
    echo "  [$1] $2"
}

have_bin() {
    if [[ "$1" == */* ]]; then
        [[ -x "$1" ]]
    else
        command -v "$1" >/dev/null 2>&1
    fi
}

# HTTP 2xx/3xx counts as up. Connection errors and 4xx/5xx do not.
healthy() {
    "$PYTHON" -c '
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=10) as resp:
        code = getattr(resp, "status", 200)
        sys.exit(0 if 200 <= code < 400 else 1)
except Exception:
    sys.exit(1)
' "$1"
}

wait_healthy() {
    local url="$1" timeout_s="$2" deadline
    deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        if healthy "$url"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

rag_server_running() {
    [[ -f "$STATE_FILE" ]] || return 1
    local pid
    pid="$("$PYTHON" -c '
import json, sys
from pathlib import Path
try:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)
pid = data.get("pid") or ""
print(pid)
' "$STATE_FILE" 2>/dev/null || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

# store.url from config.yaml, plus the port Qdrant should bind if we start it.
read_qdrant() {
    "$PYTHON" - "$REPO_ROOT/config.yaml" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlparse

url = "http://localhost:6333"
try:
    import yaml
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
    found = ((cfg.get("store") or {}).get("url")) or url
    url = str(found).strip() or url
except Exception as exc:
    print(f"warn: could not read store.url ({exc}); using {url}", file=sys.stderr)
if "://" not in url:
    url = "http://" + url
parsed = urlparse(url)
port = parsed.port or 6333
print(url.rstrip("/"))
print(port)
PY
}

spawn_detached() {
    # $1 is the log file. Remaining args are the command.
    # setsid -f forks first: a background job is already a process-group
    # leader, and setsid() fails for those. The grandchild is reparented
    # to init, so closing the terminal does not kill it.
    local log="$1"
    shift
    setsid -f "$@" >>"$log" 2>&1 </dev/null &
}

if [[ ! -x "$PYTHON" ]]; then
    echo "  [fatal] venv not found at $VENV_DIR" >&2
    echo "  run ./rag once (or: python3 -m venv .venv && pip install -r requirements.txt)" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# -- 1. rag server ----------------------------------------------------------
# The desktop window loads the embedder itself. Starting `rag serve` on
# that path would load the same weights a second time.

stage 1 "rag server"
STARTED_SERVER=0
if [[ "$NO_BROWSER" -eq 0 ]]; then
    echo "    skipped (desktop window loads the models)"
    if rag_server_running; then
        echo "    [warn] a rag server is already running at $GUI_URL and also holds the models" >&2
    fi
else
    if rag_server_running; then
        echo "    already running -> $GUI_URL"
    else
        echo "    not running, starting..."
        export RAG_PORT="$PORT"
        export RAG_HOST="$BIND_HOST"
        spawn_detached "$LOG_DIR/serve.log" "$PYTHON" -m cli serve
        STARTED_SERVER=1
    fi
fi

# -- 2. Ollama --------------------------------------------------------------

stage 2 "Ollama (generator backend)"
if healthy "$OLLAMA_URL"; then
    echo "    already running -> $OLLAMA_URL"
elif ! have_bin "$OLLAMA_BIN"; then
    echo "    [warn] '$OLLAMA_BIN' not found. Install Ollama or set OLLAMA_BIN." >&2
else
    echo "    not reachable, attempting to start..."
    spawn_detached "$LOG_DIR/ollama.log" "$OLLAMA_BIN" serve
    if wait_healthy "$OLLAMA_URL" 30; then
        echo "    ready -> $OLLAMA_URL"
    else
        echo "    [warn] Ollama did not respond within 30s. See $LOG_DIR/ollama.log" >&2
    fi
fi

# -- 3. Qdrant --------------------------------------------------------------

stage 3 "Qdrant (vector store)"
qdrant_meta="$(read_qdrant)"
QDRANT_URL="$(printf '%s\n' "$qdrant_meta" | sed -n '1p')"
QDRANT_PORT="$(printf '%s\n' "$qdrant_meta" | sed -n '2p')"
if [[ -z "$QDRANT_URL" || ! "$QDRANT_PORT" =~ ^[0-9]+$ ]]; then
    echo "    [warn] could not parse store.url; probing http://localhost:6333" >&2
    QDRANT_URL="http://localhost:6333"
    QDRANT_PORT=6333
fi

if healthy "$QDRANT_URL"; then
    echo "    already running -> $QDRANT_URL"
elif ! have_bin "$QDRANT_BIN"; then
    echo "    [warn] '$QDRANT_BIN' not found. Install Qdrant or set QDRANT_BIN." >&2
    echo "    store.url is $QDRANT_URL" >&2
else
    echo "    not reachable, attempting to start..."
    mkdir -p "$QDRANT_STORAGE"
    spawn_detached "$LOG_DIR/qdrant.log" \
        env \
        "QDRANT__SERVICE__HTTP_PORT=$QDRANT_PORT" \
        "QDRANT__STORAGE__STORAGE_PATH=$QDRANT_STORAGE" \
        "$QDRANT_BIN"
    if wait_healthy "$QDRANT_URL" 30; then
        echo "    ready -> $QDRANT_URL"
    else
        echo "    [warn] Qdrant did not respond within 30s. See $LOG_DIR/qdrant.log" >&2
    fi
fi

# -- wait for the server we just spawned ------------------------------------

if [[ "$STARTED_SERVER" -eq 1 ]]; then
    echo ""
    echo "  waiting for $GUI_URL/api/health (up to ${WAIT_TIMEOUT}s)"
    if ! wait_healthy "$GUI_URL/api/health" "$WAIT_TIMEOUT"; then
        echo "    [fatal] rag server did not become ready within ${WAIT_TIMEOUT}s" >&2
        echo "    check $LOG_DIR/serve.log and $STATE_FILE" >&2
        exit 1
    fi
    echo "    ready -> $GUI_URL"
fi

# -- 4. open the desktop window ---------------------------------------------

stage 4 "open GUI"
if [[ "$NO_BROWSER" -eq 0 ]]; then
    echo "    opening the desktop window"
    exec "$PYTHON" -m cli gui
fi

echo "    skip (--no-browser). Server: $GUI_URL"
echo ""
echo "  rag stack ready."
echo "  url : $GUI_URL"
echo "  tip : rag status / rag stats"
echo ""
exit 0
