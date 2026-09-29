#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlphonon.config import format_report, load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an MLPhonon workflow config.")
    parser.add_argument("config")
    parser.add_argument("--no-path-check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    report = validate_config(config, check_paths=not args.no_path_check)
    print(format_report(report))
    raise SystemExit(0 if report.ok else 2)


if __name__ == "__main__":
    main()
