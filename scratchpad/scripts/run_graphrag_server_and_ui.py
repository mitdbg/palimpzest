from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def _kill_process_group(p: subprocess.Popen[bytes] | subprocess.Popen[str], sig: signal.Signals) -> None:
    if p.poll() is not None:
        return
    try:
        pgid = os.getpgid(p.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def main() -> int:
    root = _repo_root()

    server_host = _env("SERVER_HOST", "127.0.0.1")
    server_port = _env("SERVER_PORT", "8002")
    ui_host = _env("UI_HOST", "127.0.0.1")
    ui_port = _env("UI_PORT", "5173")

    default_snapshot = root / "CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json"
    snapshot_path = Path(_env("PZ_GRAPH_SNAPSHOT_PATH", str(default_snapshot)))
    if not snapshot_path.is_absolute():
        snapshot_path = (root / snapshot_path).resolve()

    ui_dir = root / "ui"
    if not ui_dir.exists():
        raise SystemExit(f"UI directory not found: {ui_dir}")

    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm not found on PATH")

    if not snapshot_path.exists():
        raise SystemExit(
            "Graph snapshot not found. Set PZ_GRAPH_SNAPSHOT_PATH or run:\n"
            f"  {sys.executable} scratchpad/scripts/ingest_cms_standard.py\n"
            f"Expected snapshot at: {snapshot_path}"
        )

    server_env = os.environ.copy()
    server_env["PZ_GRAPH_SNAPSHOT_PATH"] = str(snapshot_path)

    ui_env = os.environ.copy()
    ui_env["VITE_API_URL"] = f"http://{server_host}:{server_port}"
    ui_env["VITE_WS_URL"] = f"ws://{server_host}:{server_port}"

    # Start each child in its own process group so we can reliably kill the whole tree.
    print(f"Starting API server on http://{server_host}:{server_port} (snapshot: {snapshot_path})")
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "palimpzest.server.graphrag_app:app",
        "--host",
        server_host,
        "--port",
        server_port,
        "--reload",
    ]
    server = subprocess.Popen(
        server_cmd,
        cwd=str(root),
        env=server_env,
        preexec_fn=os.setsid,
    )

    print(f"Starting UI on http://{ui_host}:{ui_port}")
    ui_cmd = [npm, "run", "dev", "--", "--host", ui_host, "--port", ui_port]
    ui = subprocess.Popen(
        ui_cmd,
        cwd=str(ui_dir),
        env=ui_env,
        preexec_fn=os.setsid,
    )

    print(f"Server PID={server.pid}, UI PID={ui.pid}")
    print("Press Ctrl+C to stop both.")

    stopping = False

    def request_stop(signum: int, _frame) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        sig = signal.Signals(signum)
        print(f"\nReceived {sig.name}; stopping children...")
        _kill_process_group(ui, signal.SIGTERM)
        _kill_process_group(server, signal.SIGTERM)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    exit_code = 0
    try:
        while True:
            s = server.poll()
            u = ui.poll()
            if s is not None or u is not None:
                # If one dies, stop the other and exit non-zero.
                if not stopping:
                    print("\nOne process exited; stopping the other...")
                _kill_process_group(ui, signal.SIGTERM)
                _kill_process_group(server, signal.SIGTERM)
                exit_code = s if s is not None else (u if u is not None else 1)
                exit_code = 1 if exit_code == 0 else int(exit_code)
                break
            time.sleep(0.25)
    finally:
        # Escalate if needed.
        time.sleep(0.5)
        _kill_process_group(ui, signal.SIGKILL)
        _kill_process_group(server, signal.SIGKILL)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
