"""CLI entrypoint for the crawler.

Real implementation lands in TICKET-002 (Phase 1). For now this exits
with a clear message so `make crawl` is wired up but does not silently
do nothing.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("[run_crawler] crawler not implemented yet — see TICKET-002 / Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
