"""Standalone environment check (mirrors `quant-platform validate-environment`)."""

from __future__ import annotations

import sys


def main() -> int:
    from quant_platform.cli import validate_environment

    try:
        validate_environment()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
