from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_step_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config")
    parser.add_argument("--model")
    parser.add_argument("--material")
    parser.add_argument("--sequential", action="store_true")
    return parser.parse_args()
