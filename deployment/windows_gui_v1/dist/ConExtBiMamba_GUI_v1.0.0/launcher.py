from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
STATE = RUNS / "gui_server_state.json"
LOG = LOGS / "gui_launcher.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def healthy(url: str, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def open_url(url: str) -> bool:
    """Open the GUI with the native Windows URL handler, with a portable fallback."""
    try:
        if os.name == "nt":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
    except OSError:
        pass
    return bool(webbrowser.open(url, new=2))


def existing_url() -> str | None:
    if not STATE.is_file():
        return None
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        url = str(state["url"])
    except Exception:
        return None
    return url if healthy(url) else None


def available_port(start: int = 8501, stop: int = 8599) -> int:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free local port was found in {start}-{stop}")


def open_when_ready(url: str) -> None:
    for _ in range(120):
        if healthy(url):
            open_url(url)
            return
        time.sleep(0.25)


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    current = existing_url()
    if current:
        open_url(current)
        return 0

    port = available_port()
    url = f"http://127.0.0.1:{port}"
    state = {
        "pid": os.getpid(),
        "port": port,
        "url": url,
        "launcher": str(Path(__file__).resolve()),
        "started_utc": utc_now(),
    }
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"

    from streamlit.web import bootstrap

    opener = threading.Thread(target=open_when_ready, args=(url,), daemon=True)
    opener.start()
    with LOG.open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"[{utc_now()}] Starting ConExtBiMamba GUI at {url}")
        try:
            bootstrap.run(
                str(APP),
                False,
                [],
                {
                    "server_address": "127.0.0.1",
                    "server_port": port,
                    "server_headless": True,
                    "server_fileWatcherType": "none",
                    "server_maxUploadSize": 1024,
                },
            )
        except Exception:
            import traceback

            traceback.print_exc(file=log)
            return 1
        finally:
            try:
                saved = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
                if int(saved.get("pid", -1)) == os.getpid():
                    STATE.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[{utc_now()}] GUI server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
