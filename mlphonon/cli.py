from __future__ import annotations

import argparse
from pathlib import Path

from .config import format_report, load_config, validate_config


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate an MLPhonon project configuration.")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--no-path-check",
        action="store_true",
        help="Validate structure and values without requiring local input files.",
    )
    args = parser.parse_args()
    report = validate_config(load_config(args.config), check_paths=not args.no_path_check)
    print(format_report(report))
    if not report.ok:
        raise SystemExit(2)
