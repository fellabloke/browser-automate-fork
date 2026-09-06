#!/usr/bin/env python3
"""Check that the canonical promotion package imports and compiles."""

from agent_first_browse.promotion.browser_promoter.graph import build_graph


def main() -> int:
    graph = build_graph()
    print("promotion graph imports and compiles:", sorted(graph.get_graph().nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
