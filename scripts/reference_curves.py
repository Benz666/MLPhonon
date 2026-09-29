#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlphonon.config import load_config, validate_reference_csv


def label_from_path(path: Path) -> str:
    return path.stem.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configured VCA reference curves.")
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config)
    refs = config.get("references", {})
    if not refs.get("enabled", False):
        print("references.enabled is false")
        return
    for item in refs.get("curves", []):
        if isinstance(item, dict):
            path = Path(item["path"])
            label = item.get("label", label_from_path(path))
        else:
            path = Path(item)
            label = label_from_path(path)
        rows, header = validate_reference_csv(path)
        print(f"{label}: {path} rows={rows} header={header}")
    for derived in refs.get("derived", []):
        label = derived.get("label", "derived")
        files = [Path(p) for p in derived.get("files", [])]
        print(f"{label}: {derived.get('operation')} of {len(files)} files")
        for path in files:
            rows, header = validate_reference_csv(path)
            print(f"  {path} rows={rows} header={header}")


if __name__ == "__main__":
    main()

