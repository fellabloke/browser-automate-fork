"""Compatibility launcher for the canonical :mod:`agent_first_browse.cli`."""

from agent_first_browse.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
