"""Canonical command-line entry point for the current browser agent."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from agent_first_browse.browser.runtime import manual_login_mode
from agent_first_browse.logging import get_logger

logger = get_logger("agent_first_browse")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent First Browse — LangGraph-based Browser Agent"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("login", help="Open browser for manual login.")
    sub.add_parser("probe-cdp", help="Verify LOCAL_CDP_ENDPOINT and exit.")
    run_parser = sub.add_parser("run", help="Run the brain with a task objective.")
    run_parser.add_argument(
        "objective",
        nargs="?",
        default=(
            "1. Go to https://the-internet.herokuapp.com/login\n"
            "2. Type 'tomsmith' into the username field.\n"
            "3. Type 'SuperSecretPassword!' into the password field.\n"
            "4. Click the Login button.\n"
            "5. Wait to see the secure area success message and then finish."
        ),
        help="The task objective for the agent.",
    )
    run_parser.add_argument(
        "--objective-base64",
        metavar="BASE64",
        help="UTF-8/base64 objective transport used by the Windows launcher.",
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "login":
        asyncio.run(manual_login_mode())
    elif args.command == "probe-cdp":
        endpoint = os.getenv("LOCAL_CDP_ENDPOINT", "").strip()
        if not endpoint:
            parser.error("LOCAL_CDP_ENDPOINT is required for probe-cdp")
        try:
            with urllib.request.urlopen(f"{endpoint.rstrip('/')}/json/version", timeout=5) as response:
                payload = json.load(response)
            websocket_url = payload.get("webSocketDebuggerUrl", "")
            if not str(websocket_url).startswith("ws"):
                raise RuntimeError("response has no valid webSocketDebuggerUrl")
            print(websocket_url)
        except Exception as exc:
            logger.error("WSL CDP probe failed for %s: %s", endpoint, exc)
            raise SystemExit(1) from exc
    elif args.command == "run":
        from agent_first_browse.agent.graph import run_brain

        objective = args.objective
        if args.objective_base64:
            try:
                objective = base64.b64decode(args.objective_base64, validate=True).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
                parser.error(f"invalid --objective-base64 value: {exc}")
        configured_headless = os.getenv("BROWSER_HEADLESS", "false").lower() in {
            "1", "true", "yes",
        }
        logger.info("🧠 Agent First Browse — LangGraph Architecture")
        logger.info("Objective: %s", objective[:200])
        asyncio.run(
            run_brain(
                objective,
                headless=getattr(args, "headless", False) or configured_headless,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
