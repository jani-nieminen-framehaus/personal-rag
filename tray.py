"""System tray icon for the rag GUI.

Run via:
    rag tray

The tray icon sits in the Windows notification area. Left-click (or
right-click > Open rag GUI) opens the browser to the running GUI.
Right-click also exposes Run eval, Status, and Quit.

The icon image is generated programmatically with Pillow at startup
so we don't need to ship a binary asset.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parent
GUI_URL = "http://localhost:8420"


# -- icon image --------------------------------------------------------------

def _make_icon() -> Image.Image:
    """Generate a 64x64 tray icon. Rounded square + stylized "R"."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Rounded square with dark fill + accent border
    draw.rounded_rectangle(
        [2, 2, 61, 61], radius=10,
        fill=(20, 24, 32, 255), outline=(88, 166, 255, 255), width=2,
    )
    # Stylized "R" — left vertical stroke
    draw.rectangle([16, 14, 23, 50], fill=(230, 237, 243, 255))
    # Top loop of the R
    draw.pieslice([22, 14, 44, 36], 180, 360, fill=(230, 237, 243, 255))
    draw.rectangle([40, 30, 44, 35], fill=(20, 24, 32, 255))  # cut the loop
    # Leg of the R
    draw.line([(24, 32), (44, 50)], fill=(230, 237, 243, 255), width=6)
    return img


# -- menu actions ------------------------------------------------------------

def _run_subprocess_in_thread(args: list[str], on_done, timeout: int = 120) -> None:
    """Spawn a subprocess off the tray thread, call on_done(output_str) when done."""
    def work() -> None:
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=timeout,
            )
            output = (result.stdout or result.stderr or "(no output)").strip()
        except subprocess.TimeoutExpired:
            output = f"timed out after {timeout}s"
        except Exception as e:
            output = f"error: {e}"
        on_done(output)
    threading.Thread(target=work, daemon=True).start()


def _on_open_gui(icon, item) -> None:
    webbrowser.open(GUI_URL)


def _on_run_eval(icon, item) -> None:
    def notify(output: str) -> None:
        # Truncate for the balloon; long output would be cut off anyway.
        short = "\n".join(output.splitlines()[:8])
        icon.notify(short, title="rag eval")
    _run_subprocess_in_thread(
        [sys.executable, str(REPO_ROOT / "cli.py"), "eval"],
        on_done=notify, timeout=180,
    )


def _on_show_status(icon, item) -> None:
    def notify(output: str) -> None:
        short = "\n".join(output.splitlines()[:6])
        icon.notify(short, title="rag status")
    _run_subprocess_in_thread(
        [sys.executable, str(REPO_ROOT / "cli.py"), "status"],
        on_done=notify, timeout=10,
    )


def _on_quit(icon, item) -> None:
    icon.stop()


# -- entry point -------------------------------------------------------------

def _build_menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem("Open rag GUI", _on_open_gui, default=True),
        pystray.MenuItem("Run eval", _on_run_eval),
        pystray.MenuItem("Status", _on_show_status),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _on_quit),
    )


def run() -> None:
    """CLI entry: blocks until the user quits the tray icon."""
    icon = pystray.Icon(
        name="rag",
        icon=_make_icon(),
        title="rag — click to open the GUI",
        menu=_build_menu(),
    )
    icon.run()


if __name__ == "__main__":
    run()
