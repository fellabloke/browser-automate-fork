"""Deprecated one-shot Windows Chrome CDP launcher.

``Start-Agent.ps1`` is the primary entry point and performs this launch plus the
WSL agent invocation. This compatibility utility only starts/reuses the same
dedicated automation Chrome and exits; it is not a bridge daemon.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CDP_ENDPOINT = "http://127.0.0.1:9222"
CDP_VERSION_URL = f"{CDP_ENDPOINT}/json/version"


def find_chrome() -> Path | None:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    return next((path for path in candidates if path.is_file()), None)


def cdp_ready() -> bool:
    try:
        with urllib.request.urlopen(CDP_VERSION_URL, timeout=2) as response:
            payload = json.load(response)
        return str(payload.get("webSocketDebuggerUrl", "")).startswith("ws")
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main() -> int:
    print("windows_chrome_bridge.py is deprecated; use Start-Agent.ps1 for the full stack.")
    if cdp_ready():
        print(f"Automation Chrome is already ready at {CDP_ENDPOINT}; reusing it.")
        return 0

    chrome_path = find_chrome()
    if chrome_path is None:
        print("[ERROR] Chrome was not found in a standard Windows install location.")
        return 1

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        print("[ERROR] LOCALAPPDATA is unavailable; cannot create the automation profile.")
        return 1
    profile_path = Path(local_app_data) / "AgentFirstBrowse" / "ChromeProfile"
    profile_path.mkdir(parents=True, exist_ok=True)

    args = [
        str(chrome_path),
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_path}",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-crash-restore-bubble",
        "--disable-session-crashed-bubble",
        "--new-window",
        "about:blank",
    ]
    creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        process = subprocess.Popen(args, creationflags=creation_flags)
    except OSError as exc:
        print(f"[ERROR] Chrome startup failed: {exc}")
        return 1
    print(f"Started dedicated automation Chrome (PID {process.pid}); waiting for CDP...")

    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if cdp_ready():
            print(f"Chrome CDP is ready at {CDP_ENDPOINT}. This utility will now exit.")
            return 0
        if process.poll() is not None:
            print(f"[ERROR] Chrome exited with code {process.returncode} before CDP was ready.")
            return 1
        time.sleep(0.25)

    print(f"[ERROR] {CDP_VERSION_URL} did not become ready within 25 seconds.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
