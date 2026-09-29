#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlphonon.config import load_config


def cleanup_standard(project_root: Path, patterns: list[str]) -> int:
    removed = 0
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply standard MLPhonon cleanup.")
    parser.add_argument("config")
    parser.add_argument("--standard", action="store_true", help="Delete standard sparse cache patterns.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["project"]["root"])
    cleanup = config.get("cleanup", {})
    patterns = cleanup.get("delete_patterns", ["_sparse*", "_spaese*"])
    if cleanup.get("mode", "standard") != "standard":
        raise SystemExit("Only cleanup.mode=standard is implemented in v1.")

    if args.dry_run:
        count = 0
        for path in project_root.rglob("*"):
            if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
                print(path)
                count += 1
        print(f"Dry run matched {count} files.")
        return

    if not args.standard:
        raise SystemExit("Pass --standard to delete standard cache files.")
    removed = cleanup_standard(project_root, patterns)
    print(f"Removed {removed} standard cache files from {project_root}.")


if __name__ == "__main__":
    main()
